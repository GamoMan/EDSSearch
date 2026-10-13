using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using NRedisStack;
using NRedisStack.RedisStackCommands;
using NRedisStack.Search;
using NRedisStack.Search.Literals.Enums;
using StackExchange.Redis;

/// <summary>
/// Redis with the search module (Redis Stack, or Redis 8+).
///   eds:doc:&lt;path&gt;   hash, indexed by RediSearch: title, keywords, description, content + v (the vector)
///   eds:meta:&lt;path&gt;  hash outside the index: image, date, hash
///   eds:paths         set of all indexed paths
///   eds:fp            fingerprint of the model / schema the index was built with
/// Hybrid search (text + vector) is done here by merging the two result lists with Reciprocal Rank Fusion.
/// Text search sees Latin words only (Redis cannot segment Thai); Thai is found through the vector.
/// A rebuild drops the index and builds it again in place, so search is empty/partial while it runs.
/// </summary>
public sealed class RedisStore : ISearchStore
{
    const string Idx = "eds:idx", Pre = "eds:doc:", Meta = "eds:meta:", PathsKey = "eds:paths", FpKey = "eds:fp";

    readonly StoreContext ctx;
    readonly EdsOptions o;
    readonly ILogger log;
    readonly IDatabase db;
    volatile bool needsRebuild;

    public string Name => "Redis";
    public bool NeedsRebuild => needsRebuild;

    public RedisStore(StoreContext ctx)
    {
        this.ctx = ctx; o = ctx.Options; log = ctx.Log;
        db = ConnectionMultiplexer.Connect(o.Redis).GetDatabase();
    }

    public int Init()
    {
        bool exists = true;
        try { db.FT().Info(Idx); } catch (RedisServerException) { exists = false; }
        if (!exists)
        {
            CreateIndex();
            log.LogInformation("Created the Redis search index '{Idx}' - run POST /api/reindex to fill it", Idx);
        }
        else
        {
            needsRebuild = (string?)db.StringGet(FpKey) != ctx.Fingerprint;
            if (needsRebuild)
                log.LogWarning("The Redis index was built with another embedding model or schema. Search falls back to text-only until " +
                               "POST /api/reindex?rebuild=true (or the next job) has rebuilt it");
        }
        return (int)db.SetLength(PathsKey);
    }

    void CreateIndex()
    {
        var w = o.Weights;
        var schema = new Schema()
            .AddTextField("title", w.Title)
            .AddTextField("keywords", w.Keywords)
            .AddTextField("description", w.Description)
            .AddTextField("content", w.Content);
        if (ctx.VectorDim is int dim)
            schema.AddVectorField("v", Schema.VectorField.VectorAlgo.FLAT, new Dictionary<string, object>
            {
                ["TYPE"] = "FLOAT32",
                ["DIM"] = dim,
                ["DISTANCE_METRIC"] = "IP"       // the doc vector is a weighted sum, see OnnxEmbedder.DocVector
            });
        db.FT().Create(Idx, new FTCreateParams().On(IndexDataType.HASH).Prefix(Pre), schema);
        db.StringSet(FpKey, ctx.Fingerprint);
        needsRebuild = false;
    }

    // drops the index with its documents, and the bookkeeping outside the index
    void ClearAll()
    {
        try { db.FT().DropIndex(Idx, dd: true); } catch (RedisServerException) { }
        foreach (var p in db.SetMembers(PathsKey)) db.KeyDelete(Meta + (string)p!);
        db.KeyDelete(PathsKey);
    }

    // what this app really sees in Redis: server, database, modules and the state of the search index
    public object Status()
    {
        string index; long? indexedDocs = null;
        try { indexedDocs = db.FT().Info(Idx).NumDocs; index = "present"; }
        catch (Exception e) { index = "missing: " + e.Message; }

        string modules;
        try { modules = db.Execute("MODULE", "LIST").ToString() ?? ""; }
        catch (Exception e) { modules = "cannot list: " + e.Message; }

        return new
        {
            endpoint = string.Join(",", db.Multiplexer.GetEndPoints().Select(x => x.ToString())),
            database = db.Database, index = Idx, indexState = index, indexedDocs,
            storedPaths = db.SetLength(PathsKey), modules
        };
    }

    static byte[] Bytes(float[] v) => MemoryMarshal.AsBytes(v.AsSpan()).ToArray();

    public void Upsert(IReadOnlyList<StoreDoc> docs)
    {
        foreach (var sd in docs)
        {
            var d = sd.Doc;
            // the vector DB hash holds only what is searched / embedded (the path is the key itself)
            db.HashSet(Pre + d.Path, new[]
            {
                new HashEntry("title", d.Title), new HashEntry("keywords", d.Keywords),
                new HashEntry("description", d.Description), new HashEntry("content", d.Content)
            });
            if (sd.Vector is { } v) db.HashSet(Pre + d.Path, "v", Bytes(v));
            else db.HashDelete(Pre + d.Path, "v");
            // bookkeeping in a separate key, outside the index
            db.HashSet(Meta + d.Path, new[]
            {
                new HashEntry("image", d.Image), new HashEntry("date", d.Date), new HashEntry("hash", d.Hash)
            });
            db.SetAdd(PathsKey, d.Path);
        }
    }

    public void Delete(IEnumerable<string> paths)
    {
        foreach (var p in paths)
        {
            db.KeyDelete(Pre + p);
            db.KeyDelete(Meta + p);
            db.SetRemove(PathsKey, p);
        }
    }

    public Dictionary<string, (string date, string hash)> Existing()
    {
        var paths = db.SetMembers(PathsKey).Select(x => (string)x!).ToList();
        var batch = db.CreateBatch();
        var tasks = paths.Select(p => batch.HashGetAsync(Meta + p, new RedisValue[] { "date", "hash" })).ToList();
        batch.Execute();
        Task.WaitAll(tasks.Cast<Task>().ToArray());
        var map = new Dictionary<string, (string date, string hash)>();
        for (int i = 0; i < paths.Count; i++)
        {
            var r = tasks[i].Result;
            map[paths[i]] = ((string?)r[0] ?? "", (string?)r[1] ?? "");
        }
        return map;
    }

    public string? GetField(string path, string field) => (string?)db.HashGet(Pre + path, field);

    public List<RawHit> Search(SearchRequest r)
    {
        string? prefix = r.Lang is { Length: 2 } && Regex.IsMatch(r.Lang, "^[a-z]{2}$") ? $"/{r.Lang}/" : null;
        int fetch = r.Fetch;
        // the index has no language field: the filter is applied afterwards, so ask for more candidates
        if (prefix is not null) fetch = Math.Max(fetch, Math.Min(Math.Max((int)db.SetLength(PathsKey), fetch), 500));

        List<(string path, string? title, string? desc, double score)>? text = null, vec = null;

        if (r.Text is not null)
        {
            // Latin words only: Redis does not segment Thai
            var words = Regex.Matches(r.Text.ToLowerInvariant(), @"[\p{L}\p{N}]{2,}").Select(m => m.Value)
                .Where(w => !w.Any(ch => ch >= '\u0E00' && ch <= '\u0E7F')).Distinct().ToList();
            if (words.Count > 0)
            {
                var tq = new Query("(" + string.Join("|", words) + ")")
                    .SetWithScores().ReturnFields("title", "description").Limit(0, fetch);
                text = db.FT().Search(Idx, tq).Documents
                    .Select(d => (Path(d), (string?)d["title"], (string?)d["description"], d.Score)).ToList();
            }
        }
        if (r.Vector is not null)
        {
            double wsum = o.Weights.Title + o.Weights.Keywords + o.Weights.Description + o.Weights.Content;
            var vq = new Query($"*=>[KNN {fetch} @v $B AS dist]")
                .AddParam("B", Bytes(r.Vector))
                .ReturnFields("title", "description", "dist")
                .SetSortBy("dist").Limit(0, fetch).Dialect(2);
            // IP distance = 1 - inner product; / weight sum = weighted average cosine (0..1)
            vec = db.FT().Search(Idx, vq).Documents
                .Select(d => (Path(d), (string?)d["title"], (string?)d["description"], Math.Max(0, 1 - (double)d["dist"]) / wsum))
                .Where(h => h.Item4 > 0).ToList();
        }

        List<(string path, string? title, string? desc, double score)> merged;
        if (text is not null && vec is not null)   // hybrid: Reciprocal Rank Fusion
        {
            var map = new Dictionary<string, (string? t, string? d, double s)>();
            foreach (var list in new[] { text, vec })
                for (int i = 0; i < list.Count; i++)
                {
                    var h = list[i];
                    double rr = 1.0 / (60 + i + 1);
                    map[h.path] = map.TryGetValue(h.path, out var e) ? (e.t ?? h.title, e.d ?? h.desc, e.s + rr) : (h.title, h.desc, rr);
                }
            merged = map.Select(kv => (kv.Key, kv.Value.t, kv.Value.d, kv.Value.s)).OrderByDescending(x => x.s).ToList();
        }
        else merged = text ?? vec ?? new List<(string path, string? title, string? desc, double score)>();

        if (prefix is not null) merged = merged.Where(h => h.path.StartsWith(prefix)).ToList();
        merged = merged.Take(r.Fetch).ToList();

        // image / date live in the meta key: one pipelined round trip for all hits
        var batch = db.CreateBatch();
        var meta = merged.Select(h => batch.HashGetAsync(Meta + h.path, new RedisValue[] { "image", "date" })).ToList();
        batch.Execute();
        Task.WaitAll(meta.Cast<Task>().ToArray());
        return merged.Select((h, i) => new RawHit(h.path, h.title, h.desc, (string?)meta[i].Result[0], (string?)meta[i].Result[1], h.score)).ToList();
    }

    static string Path(Document d) => d.Id.StartsWith(Pre) ? d.Id[Pre.Length..] : d.Id;   // the key is eds:doc:<path>

    // ---------- rebuild: drop and build again in place
    public IRebuildSession BeginRebuild()
    {
        ClearAll();
        CreateIndex();
        return new Rebuild(this);
    }

    sealed class Rebuild : IRebuildSession
    {
        readonly RedisStore s;
        public Rebuild(RedisStore s) { this.s = s; }
        public void Add(IReadOnlyList<StoreDoc> docs) => s.Upsert(docs);
        public Task CommitAsync(int expectedDocs) => Task.CompletedTask;
        public void Abort() { }
    }
}