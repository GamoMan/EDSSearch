using System.Text;
using System.Text.RegularExpressions;
using Azure;
using Azure.Core.Pipeline;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;

/// <summary>
/// Azure AI Search. Every query / update goes through an ALIAS (Eds:Azure:IndexName) that points at the current
/// real index (&lt;alias&gt;-&lt;model fingerprint&gt;-&lt;time&gt;). A rebuild creates a new index, fills it, moves the alias
/// and deletes the old one, so search never has an empty period.
/// </summary>
public sealed class AzureStore : ISearchStore
{
    const string VectorField = "contentVector";

    readonly StoreContext ctx;
    readonly EdsOptions o;
    readonly AzureSearchOptions a;
    readonly ILogger log;
    readonly SearchIndexClient indexClient;
    readonly SearchClient search;
    readonly AzureKeyCredential cred;
    readonly SearchClientOptions azOptions;
    string activeIndex = "";
    volatile bool needsRebuild;

    public string Name => "Azure AI Search";
    public bool NeedsRebuild => needsRebuild;

    public AzureStore(StoreContext ctx)
    {
        this.ctx = ctx; o = ctx.Options; a = o.Azure; log = ctx.Log;
        if (string.IsNullOrWhiteSpace(a.Endpoint) || string.IsNullOrWhiteSpace(a.ApiKey))
            throw new InvalidOperationException("Eds:Store is \"azure\": Eds:Azure:Endpoint and Eds:Azure:ApiKey must be set in appsettings.json");
        cred = new AzureKeyCredential(a.ApiKey);
        // the Azure SDK goes through the same routing proxy (rules: Eds:Proxy:Hosts, e.g. *.search.windows.net)
        azOptions = new SearchClientOptions
        {
            Transport = new HttpClientTransport(new HttpClient(new HttpClientHandler { Proxy = ctx.Proxy, UseProxy = true })
                                                { Timeout = TimeSpan.FromSeconds(100) })
        };
        indexClient = new SearchIndexClient(new Uri(a.Endpoint), cred, azOptions);
        search = ClientFor(a.IndexName);
    }

    SearchClient ClientFor(string indexOrAlias) => new(new Uri(a.Endpoint), indexOrAlias, cred, azOptions);
    string NewIndexName() => $"{a.IndexName}-{ctx.Fingerprint}-{DateTime.UtcNow:yyyyMMddHHmmss}";

    // "/th/about" -> "th", "/en/" -> "en"; anything else -> "other"
    static string Lang(string path)
    {
        var m = Regex.Match(path, @"^/([a-z]{2})(/|$)");
        return m.Success ? m.Groups[1].Value : "other";
    }

    // document key: Azure keys may only contain letters, digits, '-', '_' and '=' -> base64url of the path
    static string Key(string path) =>
        Convert.ToBase64String(Encoding.UTF8.GetBytes(path)).TrimEnd('=').Replace('+', '-').Replace('/', '_');

    public int Init()
    {
        string? current = null;
        try { current = indexClient.GetAlias(a.IndexName).Value.Indexes.FirstOrDefault(); }
        catch (RequestFailedException e) when (e.Status == 404) { }

        if (current is null)
        {
            current = NewIndexName();
            CreateIndex(current);
            indexClient.CreateOrUpdateAlias(new SearchAlias(a.IndexName, new[] { current }));
            log.LogInformation("Created Azure AI Search index '{Index}' and alias '{Alias}' - run POST /api/reindex to fill it", current, a.IndexName);
        }
        activeIndex = current;
        needsRebuild = !current.StartsWith($"{a.IndexName}-{ctx.Fingerprint}-", StringComparison.Ordinal);
        if (needsRebuild)
            log.LogWarning("Index '{Index}' was built with another embedding model or schema. Search falls back to text-only until " +
                           "POST /api/reindex?rebuild=true (or the next job) has built a new index", current);
        try { return (int)search.GetDocumentCount().Value; }
        catch (Exception e) { log.LogWarning("Could not read the document count: {Msg}", e.Message); return 0; }
    }

    // title / keywords / description / content are the searchable fields; the vector is made from the same four.
    // path, lang, image, date, hash are plain (not searchable) fields that the app needs to show links and detect changes.
    void CreateIndex(string name)
    {
        var an = new LexicalAnalyzerName(a.Analyzer);
        var fields = new List<SearchField>
        {
            new SimpleField("id", SearchFieldDataType.String) { IsKey = true },                       // base64url(path)
            new SimpleField("path", SearchFieldDataType.String) { IsFilterable = true },
            new SimpleField("lang", SearchFieldDataType.String) { IsFilterable = true },              // /th/... -> th (server-side filter)
            new SearchableField("title") { AnalyzerName = an },
            new SearchableField("keywords") { AnalyzerName = an },
            new SearchableField("description") { AnalyzerName = an },
            new SearchableField("content") { AnalyzerName = an },
            new SimpleField("image", SearchFieldDataType.String),
            new SimpleField("date", SearchFieldDataType.String),
            new SimpleField("hash", SearchFieldDataType.String)
        };
        VectorSearch? vs = null;
        if (ctx.VectorDim is int dim)
        {
            fields.Add(new SearchField(VectorField, SearchFieldDataType.Collection(SearchFieldDataType.Single))
            {
                IsSearchable = true,
                IsHidden = true,                                   // not returned in results (saves bandwidth)
                VectorSearchDimensions = dim,
                VectorSearchProfileName = "vec"
            });
            vs = new VectorSearch
            {
                Profiles = { new VectorSearchProfile("vec", "hnsw") },
                Algorithms = { new HnswAlgorithmConfiguration("hnsw") { Parameters = new HnswParameters { Metric = VectorSearchAlgorithmMetric.Cosine } } }
            };
        }
        var index = new SearchIndex(name, fields) { VectorSearch = vs, DefaultScoringProfile = "fieldWeights" };
        // the field weights 1 / .6 / .4 / .2 for the text (BM25) part of every query
        index.ScoringProfiles.Add(new ScoringProfile("fieldWeights")
        {
            TextWeights = new TextWeights(new Dictionary<string, double>
            {
                ["title"] = o.Weights.Title, ["keywords"] = o.Weights.Keywords,
                ["description"] = o.Weights.Description, ["content"] = o.Weights.Content
            })
        });
        indexClient.CreateIndex(index);
    }

    public object Status()
    {
        long? docs = null;
        try { docs = search.GetDocumentCount().Value; } catch { }
        return new { endpoint = a.Endpoint, alias = a.IndexName, activeIndex, documentsInAzure = docs, analyzer = a.Analyzer };
    }

    SearchDocument ToSearchDoc(StoreDoc sd)
    {
        var d = sd.Doc;
        var doc = new SearchDocument
        {
            ["id"] = Key(d.Path), ["path"] = d.Path, ["lang"] = Lang(d.Path),
            ["title"] = d.Title, ["keywords"] = d.Keywords, ["description"] = d.Description, ["content"] = d.Content,
            ["image"] = d.Image, ["date"] = d.Date, ["hash"] = d.Hash
        };
        if (sd.Vector is { } v) doc[VectorField] = v;
        return doc;
    }

    void Upload(SearchClient c, IReadOnlyList<StoreDoc> docs)
    {
        foreach (var chunk in docs.Chunk(100))
            c.IndexDocuments(IndexDocumentsBatch.MergeOrUpload(chunk.Select(ToSearchDoc).ToList()),
                             new IndexDocumentsOptions { ThrowOnAnyError = true });
    }

    public void Upsert(IReadOnlyList<StoreDoc> docs) => Upload(search, docs);

    public void Delete(IEnumerable<string> paths)
    {
        foreach (var chunk in paths.Select(Key).Chunk(500))
            search.IndexDocuments(IndexDocumentsBatch.Delete("id", chunk), new IndexDocumentsOptions { ThrowOnAnyError = true });
    }

    public Dictionary<string, (string date, string hash)> Existing()
    {
        var so = new SearchOptions { Size = 1000 };
        so.Select.Add("path"); so.Select.Add("date"); so.Select.Add("hash");
        var map = new Dictionary<string, (string date, string hash)>();
        foreach (var r in search.Search<SearchDocument>("*", so).Value.GetResults())
        {
            var p = r.Document.GetString("path");
            if (!string.IsNullOrEmpty(p)) map[p] = (r.Document.GetString("date") ?? "", r.Document.GetString("hash") ?? "");
        }
        return map;
    }

    // Azure's simple query syntax treats + - | " * ( ) as operators: turn them into spaces
    static string CleanQuery(string q) =>
        Regex.Replace(q, @"[\+\-\|\*\(\)\\""~\^\{\}\[\]:/<>=!&]", " ").Trim();

    public List<RawHit> Search(SearchRequest r)
    {
        string? text = null;
        if (r.Text is not null)
        {
            text = CleanQuery(r.Text);
            if (text.Length == 0) text = null;
        }
        if (text is null && r.Vector is null) return new List<RawHit>();

        var so = new SearchOptions { Size = r.Fetch };
        foreach (var f in new[] { "path", "title", "description", "image", "date" }) so.Select.Add(f);
        if (r.Lang is { Length: 2 } && Regex.IsMatch(r.Lang, "^[a-z]{2}$")) so.Filter = $"lang eq '{r.Lang}'";
        if (text is not null)
            foreach (var f in new[] { "title", "keywords", "description", "content" }) so.SearchFields.Add(f);
        if (r.Vector is not null)
            so.VectorSearch = new VectorSearchOptions
            {
                Queries = { new VectorizedQuery(r.Vector) { KNearestNeighborsCount = r.Fetch, Fields = { VectorField } } }
            };

        var hits = new List<RawHit>();
        foreach (var x in search.Search<SearchDocument>(text, so).Value.GetResults())   // text null = vector only
            hits.Add(new RawHit(x.Document.GetString("path") ?? "", x.Document.GetString("title"),
                x.Document.GetString("description"), x.Document.GetString("image"), x.Document.GetString("date"), x.Score ?? 0));
        return hits;
    }

    public string? GetField(string path, string field)
    {
        try
        {
            var go = new GetDocumentOptions();
            go.SelectedFields.Add(field);
            return search.GetDocument<SearchDocument>(Key(path), go).Value.GetString(field);
        }
        catch (RequestFailedException e) when (e.Status == 404) { return null; }
    }

    // ---------- rebuild: new index next to the live one, alias switch at the end
    public IRebuildSession BeginRebuild()
    {
        var name = NewIndexName();
        CreateIndex(name);
        return new Rebuild(this, name);
    }

    sealed class Rebuild : IRebuildSession
    {
        readonly AzureStore s;
        readonly string name;
        readonly SearchClient target;
        bool swapped;

        public Rebuild(AzureStore s, string name) { this.s = s; this.name = name; target = s.ClientFor(name); }

        public void Add(IReadOnlyList<StoreDoc> docs) => s.Upload(target, docs);

        public async Task CommitAsync(int expectedDocs)
        {
            // documents become searchable a moment after the upload: wait (up to 60 s) until they are all counted
            for (int i = 0; i < 30; i++)
            {
                try { if (target.GetDocumentCount().Value >= expectedDocs) break; } catch { }
                await Task.Delay(2000);
            }
            s.indexClient.CreateOrUpdateAlias(new SearchAlias(s.a.IndexName, new[] { name }));
            swapped = true;
            s.activeIndex = name;
            s.needsRebuild = false;
            foreach (var old in s.indexClient.GetIndexNames().Where(x => x != name && x.StartsWith(s.a.IndexName + "-")).ToList())
            {
                try { s.indexClient.DeleteIndex(old, cancellationToken: default); s.log.LogInformation("Deleted old index {Index}", old); }
                catch (Exception e) { s.log.LogWarning("Could not delete old index {Index}: {Msg}", old, e.Message); }
            }
        }

        public void Abort()
        {
            if (!swapped)   // the build failed: remove the half-filled index, the live one is untouched
            try { s.indexClient.DeleteIndex(name, cancellationToken: default); } catch { }
        }
    }
}