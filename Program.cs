using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using NRedisStack;
using NRedisStack.RedisStackCommands;
using NRedisStack.Search;
using NRedisStack.Search.Literals.Enums;
using StackExchange.Redis;

var builder = WebApplication.CreateBuilder(args);
var cfg = builder.Configuration.GetSection("Eds").Get<EdsOptions>() ?? new EdsOptions();

builder.Services.AddSingleton(cfg);
builder.Services.AddSingleton<IConnectionMultiplexer>(_ => ConnectionMultiplexer.Connect(cfg.Redis));
builder.Services.AddSingleton(_ =>
{
    var h = new HttpClientHandler();
    if (!string.IsNullOrWhiteSpace(cfg.Proxy))
    {
        h.Proxy = new WebProxy(cfg.Proxy);
        h.UseProxy = true;
        h.DefaultProxyCredentials = CredentialCache.DefaultCredentials;
    }
    return new HttpClient(h) { Timeout = TimeSpan.FromSeconds(30) };
});
builder.Services.AddSingleton<SearchService>();

var app = builder.Build();
app.Services.GetRequiredService<SearchService>().Init();

app.UseDefaultFiles();
app.UseStaticFiles();

// Optional API key for POST (reindex / update)
app.Use(async (ctx, next) =>
{
    if (ctx.Request.Method == "POST" && cfg.ApiKey.Length > 0 &&
        ctx.Request.Headers["X-Api-Key"] != cfg.ApiKey)
    {
        ctx.Response.StatusCode = 401;
        return;
    }
    await next();
});

static async Task<IResult> Guard(Func<Task<object>> f, int ok = 200)
{
    try { return Results.Json(await f(), statusCode: ok); }
    catch (InvalidOperationException e) { return Results.Json(new { error = e.Message }, statusCode: 409); }
}

// default: start in background and return 202 (poll /api/progress); add ?wait=true to block until finished
app.MapPost("/api/reindex", (SearchService s, bool? wait) =>
    Guard(() => s.ReindexAsync(wait ?? false), (wait ?? false) ? 200 : 202));
app.MapPost("/api/update", (SearchService s, UpdateRequest? req, bool? wait) =>
    Guard(() => s.UpdateAsync(req?.Paths, req?.Deep ?? false, wait ?? false), (wait ?? false) ? 200 : 202));
app.MapGet("/api/progress", (SearchService s) => Results.Ok(s.Progress()));
app.MapGet("/api/search", (SearchService s, string q, int? k, string? mode, string? lang, bool? summary) =>
    Results.Ok(s.Search(q, Math.Clamp(k ?? 10, 1, 50), mode ?? "vector", lang, summary ?? true)));
app.MapGet("/api/summary", (SearchService s, string q, string? mode, string? lang) =>
    Results.Json(s.SummaryFor(q, mode ?? "vector", lang)));
app.MapGet("/api/status", (SearchService s) => Results.Ok(s.Status()));
app.MapGet("/api/embed-test", (SearchService s, string text) => Results.Ok(s.EmbedTest(text)));

app.Run();

// ------------------------------------------------------------------ models
public class Weights { public double Title { get; set; } = 1; public double Keywords { get; set; } = .6; public double Description { get; set; } = .4; public double Content { get; set; } = .2; }
public class SummaryOptions
{
    public int TopDocs { get; set; } = 5;          // summarise from the best N results
    public int PerDoc { get; set; } = 40;          // candidate sentences taken from each page
    public int MaxSentences { get; set; } = 4;
}
public class EdsOptions
{
    public string BaseUrl { get; set; } = "";
    public string[] IndexPaths { get; set; } = Array.Empty<string>();   // e.g. /en/query-index.json, /th/query-index.json
    public string Redis { get; set; } = "localhost:6379";
    public int Dim { get; set; } = 2048;
    public int Concurrency { get; set; } = 8;
    public string Proxy { get; set; } = "";
    public string ApiKey { get; set; } = "";
    public Weights Weights { get; set; } = new();
    public EmbeddingOptions Embedding { get; set; } = new();
    public SummaryOptions Summary { get; set; } = new();
}
public record UpdateRequest(string[]? Paths, bool Deep = false);
public record Entry(string Path, string Title, string Date, string Description, string Image, string Keywords, string Robots, string Content);
public record Doc(string Path, string Title, string Date, string Description, string Image, string Keywords, string Content, string Hash);

// ------------------------------------------------------------------ service
public class SearchService
{
    const string Idx = "eds:idx", Pre = "eds:doc:", PathsKey = "eds:paths", DfKey = "eds:df", NKey = "eds:n", FpKey = "eds:fp";

    readonly EdsOptions o;
    readonly IDatabase db;
    readonly HttpClient http;
    readonly ILogger<SearchService> log;
    readonly SemaphoreSlim gate = new(1, 1);
    readonly OnnxEmbedder? emb;          // null = hashed TF-IDF mode
    int[] df;
    int n;

    // job progress (read by /api/progress while a job runs in the background)
    string job = "", state = "idle", phase = "";
    int done, total;
    DateTime? started, finished;
    string? error;
    object? result;

    public SearchService(EdsOptions o, IConnectionMultiplexer mux, HttpClient http, ILogger<SearchService> log)
    {
        this.o = o; this.http = http; this.log = log; db = mux.GetDatabase(); df = new int[o.Dim];
        emb = o.Embedding.Provider.Equals("onnx", StringComparison.OrdinalIgnoreCase)
            ? new OnnxEmbedder(o.Embedding, log) : null;
    }

    int VecDim => emb?.Dim ?? o.Dim;
    string Fingerprint => emb is null ? $"hash:{o.Dim}" : $"onnx:{emb.Dim}:{System.IO.Path.GetFileName(o.Embedding.ModelPath)}";

    void SetPhase(string p, int t = 0)
    {
        phase = p; Volatile.Write(ref total, t); Volatile.Write(ref done, 0);
        log.LogInformation("{Job}: {Phase}{Total}", job, p, t > 0 ? $" ({t})" : "");
    }

    public object Progress()
    {
        int d = Volatile.Read(ref done), t = Volatile.Read(ref total);
        var el = started is null ? 0 : ((finished ?? DateTime.UtcNow) - started.Value).TotalSeconds;
        return new
        {
            job, state, phase, done = d, total = t,
            percent = t > 0 ? Math.Round(100.0 * d / t, 1) : 0,
            elapsedSeconds = Math.Round(el, 1), error, result
        };
    }

    async Task<object> RunJob(string name, Func<Task<object>> work, bool wait)
    {
        if (!await gate.WaitAsync(0)) throw new InvalidOperationException("An index job is already running");
        job = name; state = "running"; error = null; result = null; started = DateTime.UtcNow; finished = null;
        SetPhase("starting");
        var t = Task.Run(async () =>
        {
            try { result = await work(); state = "done"; }
            catch (Exception e) { error = e.Message; state = "failed"; log.LogError(e, "{Job} failed", name); }
            finally { finished = DateTime.UtcNow; gate.Release(); }
            return Progress();
        });
        return wait ? await t : Progress();
    }

    // ---------- startup
    public void Init()
    {
        bool exists = true;
        try { db.FT().Info(Idx); } catch (RedisServerException) { exists = false; }
        if (exists && (string?)db.StringGet(FpKey) != Fingerprint)
        {
            // vector settings changed (provider / model / dimension): old vectors are unusable
            log.LogWarning("Vector settings changed ({Fp}); dropping the old index. Run POST /api/reindex.", Fingerprint);
            try { db.FT().DropIndex(Idx, dd: true); } catch (RedisServerException) { }
            db.KeyDelete(new RedisKey[] { PathsKey, DfKey, NKey });
            exists = false;
        }
        if (!exists) CreateIndex();
        LoadStats();
    }

    void CreateIndex()
    {
        var schema = new Schema()
            .AddTextField("path", 1.0, noStem: true)
            .AddTagField("lang")
            .AddTextField("title", o.Weights.Title)
            .AddTextField("keywords", o.Weights.Keywords)
            .AddTextField("description", o.Weights.Description)
            .AddTextField("content", o.Weights.Content)
            .AddVectorField("v", Schema.VectorField.VectorAlgo.FLAT, new Dictionary<string, object>
            {
                ["TYPE"] = "FLOAT32",
                ["DIM"] = VecDim,
                ["DISTANCE_METRIC"] = "IP"
            });
        db.FT().Create(Idx, new FTCreateParams().On(IndexDataType.HASH).Prefix(Pre), schema);
        db.StringSet(FpKey, Fingerprint);
    }

    void LoadStats()
    {
        var blob = db.StringGet(DfKey);
        df = new int[o.Dim];
        if (blob.HasValue)
        {
            var bytes = (byte[])blob!;
            MemoryMarshal.Cast<byte, int>(bytes).Slice(0, Math.Min(o.Dim, bytes.Length / 4)).CopyTo(df);
        }
        n = (int)db.StringGet(NKey);
    }

    void SaveStats()
    {
        db.StringSet(DfKey, MemoryMarshal.AsBytes(df.AsSpan()).ToArray());
        db.StringSet(NKey, n);
    }

    public object Status() => new { docs = n, provider = emb is null ? "hash" : "onnx", dim = VecDim, weights = o.Weights };

    // sanity check for the embedding model: token ids (compare with the HF tokenizer), dim and norm
    public object EmbedTest(string text)
    {
        if (emb is null) return new { provider = "hash", note = "Eds:Embedding:Provider is not onnx" };
        var ids = emb.DebugIds(text);
        var v = emb.QueryVector(text);
        return new { provider = "onnx", tokens = ids.Length, ids = ids.Take(64), dim = v.Length,
                     norm = Math.Round(Math.Sqrt(v.Sum(x => (double)x * x)), 4), head = v.Take(5) };
    }

    // ---------- tokenizing (no LLM): Latin words as-is, Thai as char 2/3-grams
    static IEnumerable<string> Tokens(string text)
    {
        var run = new StringBuilder();
        int kind = 0;
        foreach (var ch in text.ToLowerInvariant() + " ")
        {
            int k = ch >= '\u0E00' && ch <= '\u0E7F' ? 2 : char.IsLetterOrDigit(ch) ? 1 : 0;
            if (k != kind)
            {
                foreach (var t in Flush(run.ToString(), kind)) yield return t;
                run.Clear();
                kind = k;
            }
            if (k != 0) run.Append(ch);
        }
    }

    static IEnumerable<string> Flush(string r, int kind)
    {
        if (kind == 1 && r.Length >= 2) yield return r;
        else if (kind == 2)
        {
            if (r.Length == 1) yield return r;
            for (int size = 2; size <= 3; size++)
                for (int i = 0; i + size <= r.Length; i++)
                    yield return r.Substring(i, size);
        }
    }

    int Bucket(string t)
    {
        uint h = 2166136261;
        foreach (var c in t) { h ^= c; h *= 16777619; }
        return (int)(h % (uint)o.Dim);
    }

    Dictionary<int, int> Counts(string? text)
    {
        var d = new Dictionary<int, int>();
        if (string.IsNullOrWhiteSpace(text)) return d;
        foreach (var t in Tokens(text)) { var b = Bucket(t); d[b] = d.GetValueOrDefault(b) + 1; }
        return d;
    }

    double Idf(int b) => Math.Log((n + 1.0) / (df[b] + 1.0)) + 1.0;

    // add weight * L2-normalised TF-IDF of one field into v
    void AddField(float[] v, Dictionary<int, int> counts, double weight)
    {
        if (counts.Count == 0) return;
        var tmp = new Dictionary<int, double>();
        double norm = 0;
        foreach (var (b, c) in counts) { var w = (1 + Math.Log(c)) * Idf(b); tmp[b] = w; norm += w * w; }
        norm = Math.Sqrt(norm);
        foreach (var (b, w) in tmp) v[b] += (float)(weight * w / norm);
    }

    // doc vec = sum_f weight_f * unit(tfidf_f);  query vec = unit(tfidf)
    // => inner product = sum_f weight_f * cosine(query, field_f)
    float[] DocVector(Dictionary<int, int>[] c)
    {
        var v = new float[o.Dim];
        AddField(v, c[0], o.Weights.Title);
        AddField(v, c[1], o.Weights.Keywords);
        AddField(v, c[2], o.Weights.Description);
        AddField(v, c[3], o.Weights.Content);
        return v;
    }

    static byte[] Bytes<T>(T[] a) where T : struct => MemoryMarshal.AsBytes(a.AsSpan()).ToArray();

    // ---------- fetching
    async Task<List<Entry>> FetchIndexAsync()
    {
        var list = new List<Entry>();
        const int limit = 500;
        foreach (var indexPath in o.IndexPaths)
        for (int offset = 0; ; offset += limit)
        {
            var url = $"{o.BaseUrl.TrimEnd('/')}{indexPath}?limit={limit}&offset={offset}";
            using var json = JsonDocument.Parse(await http.GetStringAsync(url));
            var root = json.RootElement;
            var data = root.GetProperty("data");
            foreach (var e in data.EnumerateArray())
                list.Add(new Entry(S(e, "path"), S(e, "title"), S(e, "date"), S(e, "description"),
                    S(e, "image"), S(e, "keywords"), S(e, "robots"), S(e, "content")));
            int total = root.TryGetProperty("total", out var t) ? t.GetInt32() : 0;
            if (data.GetArrayLength() == 0 || offset + limit >= total) break;
        }
        return list.Where(e => e.Path.Length > 0).GroupBy(e => e.Path).Select(g => g.First()).ToList();
    }

    static string S(JsonElement e, string name) =>
        !e.TryGetProperty(name, out var p) ? "" :
        p.ValueKind == JsonValueKind.String ? p.GetString() ?? "" :
        p.ValueKind is JsonValueKind.Number ? p.GetRawText() : "";

    // "/th/about" -> "th", "/en/" -> "en"; anything else -> "other"
    static string Lang(string path)
    {
        var m = Regex.Match(path, @"^/([a-z]{2})(/|$)");
        return m.Success ? m.Groups[1].Value : "other";
    }

    string PlainUrl(string path)
    {
        var p = path.EndsWith('/') ? path + "index" : path;
        return o.BaseUrl.TrimEnd('/') + p + ".plain.html";
    }

    static string HtmlToText(string html)
    {
        html = Regex.Replace(html, @"<(script|style)[\s\S]*?</\1>", " ", RegexOptions.IgnoreCase);
        html = Regex.Replace(html, @"</(p|div|h[1-6]|li|tr|section|ul|ol|table|blockquote)>|<br\s*/?>", "\n", RegexOptions.IgnoreCase);
        html = Regex.Replace(html, @"<[^>]+>", " ");
        html = Regex.Replace(WebUtility.HtmlDecode(html), @"[ \t\r\f\v]+", " ");
        return Regex.Replace(html, @"\s*\n\s*", "\n").Trim();      // one newline between blocks
    }

    async Task<Doc?> ToDocAsync(Entry e)
    {
        if (e.Robots.Contains("noindex", StringComparison.OrdinalIgnoreCase)) return null;
        string content = e.Content;
        try { content = HtmlToText(await http.GetStringAsync(PlainUrl(e.Path))); }
        catch (HttpRequestException) { /* fall back to the content column */ }
        var hash = Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(
            string.Join("\u0001", e.Title, e.Keywords, e.Description, content))));
        return new Doc(e.Path, e.Title, e.Date, e.Description, e.Image, e.Keywords, content, hash);
    }

    async Task<List<Doc>> BuildDocsAsync(IEnumerable<Entry> entries)
    {
        using var sem = new SemaphoreSlim(o.Concurrency);
        var tasks = entries.Select(async e =>
        {
            await sem.WaitAsync();
            try { return await ToDocAsync(e); }
            finally
            {
                sem.Release();
                int cnt = Interlocked.Increment(ref done);
                if (cnt % 25 == 0 || cnt == total) log.LogInformation("{Job}: {Phase} {Done}/{Total}", job, phase, cnt, total);
            }
        });
        return (await Task.WhenAll(tasks)).Where(d => d != null).Select(d => d!).ToList();
    }

    // ---------- writing to Redis
    Dictionary<int, int>[] FieldCounts(Doc d) =>
        new[] { Counts(d.Title), Counts(d.Keywords), Counts(d.Description), Counts(d.Content) };

    static ushort[] BucketsOf(Dictionary<int, int>[] counts) =>
        counts.SelectMany(c => c.Keys).Distinct().Select(b => (ushort)b).ToArray();

    void Write(Doc d, float[] vec, ushort[] buckets)
    {
        var flat = Regex.Replace(d.Content, @"\s+", " ");
        var snippet = flat.Length > 240 ? flat[..240] + "…" : flat;
        db.HashSet(Pre + d.Path, new[]
        {
            new HashEntry("path", d.Path), new HashEntry("lang", Lang(d.Path)), new HashEntry("title", d.Title),
            new HashEntry("keywords", d.Keywords), new HashEntry("description", d.Description),
            new HashEntry("content", d.Content), new HashEntry("snippet", snippet),
            new HashEntry("image", d.Image), new HashEntry("date", d.Date), new HashEntry("hash", d.Hash),
            new HashEntry("v", Bytes(vec)), new HashEntry("b", Bytes(buckets))
        });
        db.SetAdd(PathsKey, d.Path);
    }

    void Remove(string path)
    {
        var b = db.HashGet(Pre + path, "b");
        if (b.HasValue)
            foreach (var x in MemoryMarshal.Cast<byte, ushort>((byte[])b!))
                if (x < df.Length) df[x] = Math.Max(0, df[x] - 1);
        if (db.KeyDelete(Pre + path)) n = Math.Max(0, n - 1);
        db.SetRemove(PathsKey, path);
    }

    void Upsert(Doc d)
    {
        if (db.KeyExists(Pre + d.Path)) Remove(d.Path);
        n++;
        if (emb is not null) { Write(d, emb.DocVector(d, o.Weights), Array.Empty<ushort>()); return; }
        var counts = FieldCounts(d);
        foreach (var b in counts.SelectMany(c => c.Keys).Distinct()) df[b]++;
        Write(d, DocVector(counts), BucketsOf(counts));
    }

    void RecreateIndex()
    {
        try { db.FT().DropIndex(Idx, dd: true); } catch (RedisServerException) { }
        db.KeyDelete(new RedisKey[] { PathsKey, DfKey, NKey });
        CreateIndex();
    }

    // ---------- API: reindex (full rebuild)
    public Task<object> ReindexAsync(bool wait) => RunJob("reindex", ReindexCore, wait);

    async Task<object> ReindexCore()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        SetPhase("fetching query-index");
        var entries = await FetchIndexAsync();
        SetPhase("fetching pages", entries.Count);
        var docs = await BuildDocsAsync(entries);

        if (emb is null)
        {
            // hashed TF-IDF: IDF needs every document first -> compute all vectors, then recreate + write
            SetPhase("computing vectors", docs.Count);
            var prepared = new List<(Doc d, float[] vec, ushort[] b)>(docs.Count);
            var counts = docs.Select(FieldCounts).ToList();
            var newDf = new int[o.Dim];
            foreach (var c in counts)
                foreach (var k in c.SelectMany(x => x.Keys).Distinct()) newDf[k]++;
            df = newDf; n = docs.Count;
            for (int i = 0; i < docs.Count; i++)
            { prepared.Add((docs[i], DocVector(counts[i]), BucketsOf(counts[i]))); Interlocked.Increment(ref done); }

            SetPhase("recreating index");
            RecreateIndex();

            SetPhase("writing to Redis", prepared.Count);
            foreach (var (d, vec, b) in prepared) { Write(d, vec, b); Interlocked.Increment(ref done); }
        }
        else
        {
            // embeddings are slow per page, so each page is written as soon as it is embedded:
            // it shows up in Redis immediately, is searchable while the job runs, and progress = pages written
            SetPhase("recreating index");
            RecreateIndex();
            df = new int[o.Dim]; n = 0;
            SetPhase("embedding + writing", docs.Count);
            foreach (var d in docs)
            {
                Write(d, emb.DocVector(d, o.Weights), Array.Empty<ushort>());
                n++;
                int cnt = Interlocked.Increment(ref done);
                if (cnt % 20 == 0 || cnt == docs.Count)
                    log.LogInformation("reindex: embedded {Done}/{Total} pages ({Sec:F0}s)", cnt, docs.Count, sw.Elapsed.TotalSeconds);
            }
        }
        SaveStats();

        return new { indexed = n, skipped = entries.Count - n, seconds = Math.Round(sw.Elapsed.TotalSeconds, 1) };
    }

    // ---------- API: update (incremental)
    public Task<object> UpdateAsync(string[]? paths, bool deep, bool wait) =>
        RunJob("update", () => UpdateCore(paths, deep), wait);

    async Task<object> UpdateCore(string[]? paths, bool deep)
    {
        if (n == 0) return await ReindexCore();      // nothing indexed yet

        SetPhase("fetching query-index");
        var entries = await FetchIndexAsync();
        var byPath = entries.ToDictionary(e => e.Path);
        var existing = db.SetMembers(PathsKey).Select(x => (string)x!).ToHashSet();

        List<string> check, removed;
        if (paths is { Length: > 0 })
        {
            check = paths.Where(byPath.ContainsKey).ToList();
            removed = paths.Where(p => !byPath.ContainsKey(p) && existing.Contains(p)).ToList();
        }
        else
        {
            removed = existing.Except(byPath.Keys).ToList();
            check = deep
                ? byPath.Keys.ToList()
                : byPath.Keys.Where(p => !existing.Contains(p) || (string?)db.HashGet(Pre + p, "date") != byPath[p].Date).ToList();
        }

        SetPhase("fetching pages", check.Count);
        var docs = await BuildDocsAsync(check.Select(p => byPath[p]));
        SetPhase(emb is null ? "writing to Redis" : "embedding + writing", docs.Count);
        var okPaths = docs.Select(d => d.Path).ToHashSet();
        removed.AddRange(check.Where(p => !okPaths.Contains(p) && existing.Contains(p))); // became noindex

        int added = 0, updated = 0, unchanged = 0;
        foreach (var d in docs)
        {
            Interlocked.Increment(ref done);
            bool had = existing.Contains(d.Path);
            if (had && deep && (string?)db.HashGet(Pre + d.Path, "hash") == d.Hash) { unchanged++; continue; }
            Upsert(d);
            if (had) updated++; else added++;
        }
        foreach (var p in removed.Distinct()) Remove(p);
        SaveStats();

        return new { added, updated, removed = removed.Distinct().Count(), unchanged, total = n };
    }

    // ---------- API: search
    public record Hit(string? path, string url, string? title, string? description, string? snippet,
                      string? image, string? date, double score);

    List<Hit> SearchHits(string q, int k, string mode, string? lang)
    {
        // optional language filter, e.g. lang=th
        var filter = lang is { Length: 2 } && Regex.IsMatch(lang, "^[a-z]{2}$") ? $"@lang:{{{lang}}}" : null;
        if (string.IsNullOrWhiteSpace(q) || n == 0) return new List<Hit>();
        var fields = new[] { "path", "title", "description", "image", "date", "snippet" };
        double wsum = o.Weights.Title + o.Weights.Keywords + o.Weights.Description + o.Weights.Content;

        Hit ToHit(Document d, double score) => new((string?)d["path"], o.BaseUrl.TrimEnd('/') + (string?)d["path"],
            (string?)d["title"], (string?)d["description"], (string?)d["snippet"],
            (string?)d["image"], (string?)d["date"], score);

        if (mode == "text")   // RediSearch full-text, field weights from the schema (Latin words only)
        {
            var words = Tokens(q).Where(t => !(t[0] >= '\u0E00' && t[0] <= '\u0E7F')).Distinct().ToList();
            if (words.Count == 0) return new List<Hit>();
            var tq = new Query((filter is null ? "" : filter + " ") + "(" + string.Join("|", words) + ")")
                .SetWithScores().ReturnFields(fields).Limit(0, k);
            return db.FT().Search(Idx, tq).Documents.Select(d => ToHit(d, Math.Round(d.Score, 4))).ToList();
        }

        float[] qv;
        if (emb is not null) qv = emb.QueryVector(q);
        else
        {
            var qc = Counts(q);
            if (qc.Count == 0) return new List<Hit>();
            qv = new float[o.Dim];
            AddField(qv, qc, 1.0);
        }

        var query = new Query($"{(filter is null ? "*" : "(" + filter + ")")}=>[KNN {k} @v $B AS dist]")
            .AddParam("B", Bytes(qv))
            .ReturnFields(fields.Append("dist").ToArray())
            .SetSortBy("dist")
            .Limit(0, k)
            .Dialect(2);
        var res = db.FT().Search(Idx, query);
        // IP distance = 1 - inner product; divide by the weight sum -> 0..1 weighted cosine
        return res.Documents
            .Select(d => ToHit(d, Math.Round(Math.Max(0, 1 - (double)d["dist"]) / wsum, 4)))
            .Where(h => h.score > 0).ToList();
    }

    public object Search(string q, int k, string mode, string? lang = null, bool summary = true)
    {
        var hits = SearchHits(q, k, mode, lang);
        return new { mode, summary = summary ? Summarize(q, hits) : null, results = hits };
    }

    // same top results as /api/search, so the [n] citations match the result numbers
    public object? SummaryFor(string q, string mode, string? lang) =>
        Summarize(q, SearchHits(q, o.Summary.TopDocs, mode, lang));

    // ---------- extractive summary: no generation, only sentences that exist on the pages, each with its source
    record Cand(int Doc, int Pos, string Text);
    public record SummarySentence(string text, int source);

    static readonly Regex SentSplit = new(@"(?<=[.!?…])\s+|\n+", RegexOptions.Compiled);

    static IEnumerable<string> Sentences(string text)
    {
        var seen = new HashSet<string>();
        foreach (var raw in SentSplit.Split(text))
        {
            var seg = raw.Trim();
            if (seg.Length == 0) continue;
            foreach (var piece in CutLong(seg, 220))
                if (piece.Length >= 25 && seen.Add(piece)) yield return piece;
        }
    }

    // Thai has no sentence punctuation: cut long segments at spaces into ~220 char pieces
    static IEnumerable<string> CutLong(string seg, int max)
    {
        if (seg.Length <= 320) { yield return seg; yield break; }
        var sb = new StringBuilder();
        foreach (var w in seg.Split(' '))
        {
            if (sb.Length > 0 && sb.Length + w.Length + 1 > max) { yield return sb.ToString(); sb.Clear(); }
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(w);
        }
        if (sb.Length > 0) yield return sb.ToString();
    }

    static double Dot(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i];
        return s;
    }

    static string Clip(string s, int n) => s.Length <= n ? s : s[..n] + "…";

    double SparseCos(Dictionary<int, int> a, Dictionary<int, int> b)
    {
        double dot = 0, na = 0, nb = 0;
        foreach (var (k, c) in a)
        {
            var w = (1 + Math.Log(c)) * Idf(k);
            na += w * w;
            if (b.TryGetValue(k, out var c2)) dot += w * (1 + Math.Log(c2)) * Idf(k);
        }
        foreach (var (k, c) in b) { var w = (1 + Math.Log(c)) * Idf(k); nb += w * w; }
        return na > 0 && nb > 0 ? dot / Math.Sqrt(na * nb) : 0;
    }

    object? Summarize(string q, List<Hit> hits)
    {
        var so = o.Summary;
        var cands = new List<Cand>();
        for (int i = 0; i < Math.Min(so.TopDocs, hits.Count); i++)
        {
            var key = Pre + hits[i].path;
            var desc = ((string?)db.HashGet(key, "description") ?? "").Trim();
            if (desc.Length >= 25) cands.Add(new Cand(i, -1, desc));
            int pos = 0;
            foreach (var sent in Sentences((string?)db.HashGet(key, "content") ?? "").Take(so.PerDoc))
                cands.Add(new Cand(i, pos++, sent));
        }
        if (cands.Count == 0) return null;

        double[] sims;
        float[][]? vecs = null;
        if (emb is not null)
        {
            var qv = emb.QueryVector(q);
            vecs = emb.PassageVectors(cands.Select(c => c.Text).ToList());
            sims = vecs.Select(v => Dot(qv, v)).ToArray();
        }
        else
        {
            var qc = Counts(q);
            sims = cands.Select(c => SparseCos(qc, Counts(c.Text))).ToArray();
        }

        double Score(int i) => sims[i] * (1 - 0.04 * cands[i].Doc);      // small bonus for higher-ranked pages
        var order = Enumerable.Range(0, cands.Count).OrderByDescending(Score).ToList();
        double best = Score(order[0]);
        if (best <= 0) return null;
        double floor = emb is null ? best * 0.75 : best - 0.06;          // keep only sentences close to the best one

        var picked = new List<int>();
        foreach (var i in order)
        {
            if (picked.Count >= so.MaxSentences || Score(i) < floor) break;
            bool dup = picked.Any(j => cands[j].Text == cands[i].Text || (vecs is not null && Dot(vecs[i], vecs[j]) > 0.93));
            if (!dup) picked.Add(i);
        }

        var sentences = picked.OrderBy(i => cands[i].Doc).ThenBy(i => cands[i].Pos)     // reading order
            .Select(i => new SummarySentence(Clip(cands[i].Text, 320), cands[i].Doc + 1)).ToList();
        return new
        {
            text = string.Join(" ", sentences.Select(x => $"{x.text} [{x.source}]")),
            sentences,
            sources = sentences.Select(x => x.source).Distinct().ToList()
        };
    }
}
