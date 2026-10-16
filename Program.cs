using System.Collections.Concurrent;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

var builder = WebApplication.CreateBuilder(args);
AppPaths.Root = builder.Environment.ContentRootPath;      // models/... in appsettings.json are relative to the app folder
var cfg = builder.Configuration.GetSection("Eds").Get<EdsOptions>() ?? new EdsOptions();

builder.Services.AddSingleton(cfg);
// every outgoing fetch (query-index.json, pages) goes through RoutingProxy, which decides per host
// whether the corporate proxy is used (Eds:Proxy section, switchable at runtime with /api/proxy)
builder.Services.AddSingleton<RoutingProxy>();
builder.Services.AddSingleton(sp =>
{
    var c = new HttpClient(new HttpClientHandler
    {
        Proxy = sp.GetRequiredService<RoutingProxy>(),
        UseProxy = true
    }) { Timeout = TimeSpan.FromSeconds(30) };
    c.DefaultRequestHeaders.UserAgent.TryParseAdd(cfg.Fetch.UserAgent);
    return c;
});
builder.Services.AddSingleton<SearchService>();
builder.Services.AddSingleton<JobScheduler>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<JobScheduler>());   // runs the scheduled jobs

var app = builder.Build();
app.Services.GetRequiredService<SearchService>().Init();

app.UseDefaultFiles();
app.UseStaticFiles();

// tell the developer straight away if the web page cannot be found (a missing wwwroot gives a blank 404 page)
var indexFile = System.IO.Path.Combine(app.Environment.WebRootPath ?? "(no wwwroot folder)", "index.html");
app.Logger.LogInformation("Web page: {File} -> {State}", indexFile, File.Exists(indexFile) ? "found" : "MISSING");
app.MapGet("/", () => Results.Text(
    $"index.html not found.\nExpected at: {indexFile}\nContent root: {app.Environment.ContentRootPath}\n" +
    "Put wwwroot/index.html next to Program.cs and run dotnet run from that folder.", "text/plain"));

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
    catch (KeyNotFoundException e) { return Results.Json(new { error = e.Message }, statusCode: 404); }
}

// default: start in background and return 202 (poll /api/progress); add ?wait=true to block until finished
// reindex = compare every page with what is indexed and change only the differences (search keeps working);
// ?rebuild=true builds a brand-new index next to the current one and switches to it when it is complete
// (needed after changing the embedding model; search keeps using the old index meanwhile)
app.MapPost("/api/reindex", (SearchService s, bool? wait, bool? rebuild) =>
    Guard(() => s.ReindexAsync(wait ?? false, rebuild ?? false), (wait ?? false) ? 200 : 202));
// update: new / changed (date) / removed pages; body {"deep":true} compares content, {"urls":[...]} or {"paths":[...]} only those pages
app.MapPost("/api/update", (SearchService s, UpdateRequest? req, bool? wait) =>
    Guard(() => s.UpdateAsync((req?.Paths ?? Array.Empty<string>()).Concat(req?.Urls ?? Array.Empty<string>()).ToArray(),
                              req?.Deep ?? false, wait ?? false), (wait ?? false) ? 200 : 202));
// one page: POST /api/update/url?url=https://www.example.com/th/about   (or just /th/about)
app.MapPost("/api/update/url", (SearchService s, string url, bool? wait) =>
    Guard(() => s.UpdateAsync(new[] { url }, false, wait ?? false), (wait ?? false) ? 200 : 202));
// proxy: GET state, POST ?enabled=true|false switches it at runtime (no value = back to config), GET /test?url= shows the route
app.MapGet("/api/proxy", (RoutingProxy p) => Results.Ok(p.State()));
app.MapPost("/api/proxy", (RoutingProxy p, bool? enabled) => { p.SetOverride(enabled); return Results.Ok(p.State()); });
app.MapGet("/api/proxy/test", (RoutingProxy p, string url) => Results.Ok(p.Test(url)));
app.MapGet("/api/jobs", (JobScheduler j) => Results.Ok(j.List()));
app.MapPost("/api/jobs/{name}/run", (JobScheduler j, string name) => Guard(() => j.RunNowAsync(name), 202));
app.MapGet("/api/progress", (SearchService s) => Results.Ok(s.Progress()));
app.MapGet("/api/search", (SearchService s, string q, int? k, string? mode, string? lang, bool? summary) =>
    Results.Ok(s.Search(q, Math.Clamp(k ?? 10, 1, 50), mode ?? "hybrid", lang, summary ?? false)));   // summary=true -> extractive summary inside the JSON
// whole answer as JSON (LLM if enabled and loaded, otherwise extractive) - waits until generation has finished
app.MapGet("/api/summary", (SearchService s, string q, string? mode, string? lang) =>
    Results.Json(s.SummaryFor(q, mode ?? "hybrid", lang)));

// the same answer streamed token by token (Server-Sent Events) - used by the web page
app.MapGet("/api/summary/stream", async (HttpContext ctx, SearchService s, string q, string? mode, string? lang) =>
{
    ctx.Response.Headers.ContentType = "text/event-stream";
    ctx.Response.Headers.CacheControl = "no-cache";
    async Task Send(object o)
    {
        await ctx.Response.WriteAsync("data: " + JsonSerializer.Serialize(o) + "\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }
    try
    {
        await Send(new { type = "meta", mode = s.LlmEnabled ? "llm" : "extractive" });
        foreach (var piece in s.StreamAnswer(q, mode ?? "hybrid", lang, ctx.RequestAborted))
            await Send(new { type = "token", text = piece });
        await Send(new { type = "done" });
    }
    catch (OperationCanceledException) { }          // browser closed / started a new search
    catch (Exception e) { await Send(new { type = "error", message = e.Message }); }
});
app.MapGet("/api/status", (SearchService s) => Results.Ok(s.Status()));
app.MapGet("/api/embed-test", (SearchService s, string text, string? text2) => Results.Ok(s.EmbedTest(text, text2)));

app.Run();

// ------------------------------------------------------------------ models
public class Weights { public double Title { get; set; } = 1; public double Keywords { get; set; } = .6; public double Description { get; set; } = .4; public double Content { get; set; } = .2; }

// Azure AI Search connection (Eds:Azure), used when Eds:Store is "azure"
public class AzureSearchOptions
{
    public string Endpoint { get; set; } = "";             // https://<your-service>.search.windows.net
    public string ApiKey { get; set; } = "";               // admin key (creates indexes, uploads and deletes documents)
    public string IndexName { get; set; } = "eds-pages";   // ALIAS name. The real indexes are <alias>-<model fingerprint>-<time>
    public string Analyzer { get; set; } = "th.microsoft"; // text analyzer: segments Thai words, also fine for English
}

// how politely pages and query-index.json are requested (the server answers 429 Too Many Requests when we are too fast)
public class FetchOptions
{
    public int Concurrency { get; set; } = 4;         // pages fetched at the same time
    public int MinIntervalMs { get; set; } = 200;     // minimum gap between two requests, all threads together (200 = 5 per second)
    public int MaxRetries { get; set; } = 6;          // retries for 429 / 502 / 503 / 504 / timeouts
    public int RetryBaseSeconds { get; set; } = 2;    // wait 2, 4, 8, 16 ... seconds, or the server's Retry-After when it sends one
    public int MaxWaitSeconds { get; set; } = 120;    // never wait longer than this for one retry
    public string UserAgent { get; set; } = "EdsSearch-Indexer/1.0";
}
// extra rules for text that must not be indexed or shown (the built-in rules already handle section-metadata, paths, GUIDs, style names)
public class CleaningOptions
{
    public string[] ExtraDropClasses { get; set; } = Array.Empty<string>();    // more block class names to remove, e.g. "cookie-banner"
    public string[] ExtraLinePatterns { get; set; } = Array.Empty<string>();   // regex: a whole line of text that matches is dropped
    public string[] ExtraRemovePatterns { get; set; } = Array.Empty<string>(); // regex: the matching text is cut out, the rest of the line stays
}
public class DemoteRule
{
    public string PathPattern { get; set; } = "";      // regex on the page path, e.g. "/(news|press)(/|$)"
    public double Factor { get; set; } = 0.9;          // score is multiplied by this (1 = no change, lower = pushed further down)
}
public class RankingOptions { public List<DemoteRule> Demote { get; set; } = new(); }

public class SummaryOptions
{
    public int TopDocs { get; set; } = 5;          // summarise from the best N results
    public int PerDoc { get; set; } = 40;          // candidate sentences taken from each page
    public int MaxSentences { get; set; } = 4;
}
public class EdsOptions
{
    public string BaseUrl { get; set; } = "";            // where pages are FETCHED from (e.g. https://main--site--org.aem.live)
    public string PublicBaseUrl { get; set; } = "";      // domain of the LINKS in search results (e.g. https://www.example.com); empty = BaseUrl
    public string TimeZone { get; set; } = "Asia/Bangkok";   // time zone of the job cron expressions
    public List<JobConfig> Jobs { get; set; } = new();
    public string[] IndexPaths { get; set; } = Array.Empty<string>();   // e.g. /en/query-index.json, /th/query-index.json
    public string Store { get; set; } = "redis";         // where the index lives: "redis" or "azure"
    public string Redis { get; set; } = "localhost:6379";   // used when Store = redis
    public AzureSearchOptions Azure { get; set; } = new();  // used when Store = azure
    public string ApiKey { get; set; } = "";             // protects the POST endpoints of THIS app (header X-Api-Key)
    public Weights Weights { get; set; } = new();
    public EmbeddingOptions Embedding { get; set; } = new();
    public SummaryOptions Summary { get; set; } = new();
    public LlmOptions Llm { get; set; } = new();
    public RankingOptions Ranking { get; set; } = new();
    public FetchOptions Fetch { get; set; } = new();
    public CleaningOptions Cleaning { get; set; } = new();
}
public record UpdateRequest(string[]? Paths, bool Deep = false, string[]? Urls = null);
public record Entry(string Path, string Title, string Date, string Description, string Image, string Keywords, string Robots, string Content);
public record Doc(string Path, string Title, string Date, string Description, string Image, string Keywords, string Content, string Hash);

// ------------------------------------------------------------------ service
public class SearchService
{
    readonly EdsOptions o;
    readonly HttpClient http;
    readonly ILogger<SearchService> log;
    readonly IConfiguration conf;        // read again on every use -> PublicBaseUrl changes without a restart
    readonly SemaphoreSlim gate = new(1, 1);
    readonly OnnxEmbedder? emb;          // null = no vectors: the index is text-only
    readonly LlmSummarizer? llm;         // null = extractive summary only
    readonly List<(Regex re, double factor)> demote;   // pages whose score is lowered (e.g. news)
    readonly string[] dropClasses;       // page blocks removed before indexing
    readonly Regex[] extraJunk;          // extra "this line is not content" patterns (Eds:Cleaning)
    readonly Regex[] extraRemove;        // extra "cut this text out" patterns (Eds:Cleaning)
    readonly ISearchStore store;         // Redis or Azure AI Search (Eds:Store)
    const int BatchSize = 50;            // pages per write to the store
    int n;                               // number of documents (kept up to date by the jobs)

    // polite fetching: one shared pace for all threads, a global pause after 429, pages that could not be fetched
    readonly object paceLock = new();
    long nextSlot, pauseUntil;                                    // Environment.TickCount64 milliseconds
    readonly ConcurrentDictionary<string, byte> failed = new();

    // job progress (read by /api/progress while a job runs in the background)
    string job = "", state = "idle", phase = "";
    int done, total;
    DateTime? started, finished;
    string? error;
    object? result;

    public SearchService(EdsOptions o, HttpClient http, ILogger<SearchService> log, IConfiguration conf, RoutingProxy proxy)
    {
        this.o = o; this.http = http; this.log = log; this.conf = conf;
        emb = o.Embedding.Provider.Equals("onnx", StringComparison.OrdinalIgnoreCase)
            ? new OnnxEmbedder(o.Embedding, log) : null;
        dropClasses = BuiltInDropClasses.Concat(o.Cleaning.ExtraDropClasses).ToArray();
        extraJunk = o.Cleaning.ExtraLinePatterns.Where(p => p.Length > 0)
            .Select(p => new Regex(p, RegexOptions.IgnoreCase | RegexOptions.Compiled)).ToArray();
        extraRemove = o.Cleaning.ExtraRemovePatterns.Where(p => p.Length > 0)
            .Select(p => new Regex(p, RegexOptions.IgnoreCase | RegexOptions.Compiled)).ToArray();
        demote = o.Ranking.Demote.Where(r => r.PathPattern.Length > 0)
            .Select(r => (new Regex(r.PathPattern, RegexOptions.IgnoreCase | RegexOptions.Compiled), r.Factor)).ToList();
        if (o.Llm.Enabled)
        {
            try { llm = new LlmSummarizer(o.Llm, log); }
            catch (Exception e)   // missing model must not stop search: fall back to the extractive summary
            { log.LogWarning("LLM summary disabled: {Msg}", e.Message); }
        }

        var ctx = new StoreContext(o, emb?.Dim, Fingerprint, log, proxy);
        store = o.Store.Trim().ToLowerInvariant() switch
        {
            "redis" => new RedisStore(ctx),
            "azure" => new AzureStore(ctx),
            _ => throw new InvalidOperationException($"Eds:Store must be \"redis\" or \"azure\" (it is \"{o.Store}\")")
        };
    }

    // domain used for the links in search results; falls back to the fetch origin
    string PublicBase
    {
        get
        {
            var p = conf["Eds:PublicBaseUrl"];
            if (string.IsNullOrWhiteSpace(p)) p = conf["Eds:BaseUrl"];
            return (p ?? o.BaseUrl).TrimEnd('/');
        }
    }

    // identifies what the vectors in the index were made with (model file + size); a different value means "rebuild"
    string Fingerprint
    {
        get
        {
            var s = (emb is null ? "text" : $"onnx:{emb.Dim}:{System.IO.Path.GetFileName(o.Embedding.ModelPath)}") + ":s3";
            return Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(s)))[..8].ToLowerInvariant();
        }
    }

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
        failed.Clear();
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

    // ---------- startup: open (or create) the index in the chosen store
    public void Init()
    {
        n = store.Init();
        log.LogInformation("Search store: {Store}; {N} documents; vectors {Vec}", store.Name, n,
            emb is null ? "off (text only)" : $"on ({emb.Dim} dim)");
    }

    public object Status() => new
    {
        store = store.Name, documents = n, needsRebuild = store.NeedsRebuild,
        vectors = emb is null ? "off (text only)" : $"on ({emb.Dim} dim)",
        weights = o.Weights, details = store.Status()
    };

    // sanity check for the embedding model: token ids (compare with the HF tokenizer), dim and norm
    public object EmbedTest(string text, string? text2 = null)
    {
        if (emb is null) return new { provider = "none", note = "Eds:Embedding:Provider is not onnx" };
        var ids = emb.DebugIds(text);
        var v = emb.QueryVector(text);
        // text2 (optional): cosine between the two texts - a quick quality check, e.g. a Thai phrase vs its English meaning
        double? sim = text2 is null ? null : Math.Round(Dot(v, emb.QueryVector(text2)), 4);
        return new { provider = "onnx", tokens = ids.Length, ids = ids.Take(64), dim = v.Length,
                     norm = Math.Round(Math.Sqrt(v.Sum(x => (double)x * x)), 4), head = v.Take(5), similarity = sim };
    }

    // ---------- tokenizing: Latin words as-is, Thai as char 2/3-grams (only used to rank sentences for the summary without vectors)
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

    // ---------- polite HTTP: pacing, retry on 429 / 5xx, Retry-After
    async Task Pace()
    {
        long wait;
        lock (paceLock)
        {
            long now = Environment.TickCount64;
            long slot = Math.Max(Math.Max(now, nextSlot), pauseUntil);
            nextSlot = slot + Math.Max(0, o.Fetch.MinIntervalMs);
            wait = slot - now;
        }
        if (wait > 0) await Task.Delay((int)Math.Min(wait, int.MaxValue));
    }

    // after a 429 every thread waits, not only the one that was refused
    void PauseAll(TimeSpan d)
    {
        lock (paceLock) pauseUntil = Math.Max(pauseUntil, Environment.TickCount64 + (long)d.TotalMilliseconds);
    }

    TimeSpan RetryDelay(HttpResponseMessage? resp, int attempt)
    {
        var f = o.Fetch;
        TimeSpan? server = null;
        if (resp?.Headers.RetryAfter is { } ra)
            server = ra.Delta ?? (ra.Date is { } dt ? (TimeSpan?)(dt - DateTimeOffset.UtcNow) : null);
        var backoff = TimeSpan.FromSeconds(Math.Min(f.MaxWaitSeconds, f.RetryBaseSeconds * Math.Pow(2, attempt)));
        var d = server is { } s && s > TimeSpan.Zero ? s : backoff;
        d += TimeSpan.FromMilliseconds(Random.Shared.Next(0, 1000));          // jitter
        var max = TimeSpan.FromSeconds(f.MaxWaitSeconds);
        return d > max ? max : d;
    }

    async Task<HttpResponseMessage> GetAsync(string url)
    {
        var f = o.Fetch;
        for (int attempt = 0; ; attempt++)
        {
            await Pace();
            string why;
            TimeSpan wait;
            try
            {
                var resp = await http.GetAsync(url);
                int code = (int)resp.StatusCode;
                if (code is not (429 or 502 or 503 or 504) || attempt >= f.MaxRetries) return resp;
                why = $"HTTP {code}";
                wait = RetryDelay(resp, attempt);
                resp.Dispose();
            }
            catch (Exception e) when ((e is HttpRequestException || e is TaskCanceledException) && attempt < f.MaxRetries)
            {
                why = e.GetType().Name + ": " + e.Message;
                wait = RetryDelay(null, attempt);
            }
            log.LogWarning("{Url}: {Why} - waiting {Sec:F0}s before retry {N}/{Max}", url, why, wait.TotalSeconds, attempt + 1, f.MaxRetries);
            PauseAll(wait);
        }
    }

    async Task<string> GetStringAsync(string url)
    {
        using var resp = await GetAsync(url);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }

    // ---------- reading query-index.json and the pages
    async Task<List<Entry>> FetchIndexAsync()
    {
        var list = new List<Entry>();
        const int limit = 500;
        foreach (var indexPath in o.IndexPaths)
        for (int offset = 0; ; offset += limit)
        {
            var url = $"{o.BaseUrl.TrimEnd('/')}{indexPath}?limit={limit}&offset={offset}";
            using var json = JsonDocument.Parse(await GetStringAsync(url));
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

    string PlainUrl(string path)
    {
        var p = path.EndsWith('/') ? path + "index" : path;
        return o.BaseUrl.TrimEnd('/') + p + ".plain.html";
    }

    // parts of an EDS page that are settings, not content: the "section-metadata" table (style, is-subnav-section ...)
    // zero-width space/joiners, left-to-right / right-to-left marks, soft hyphen, BOM: invisible, never content
    static readonly Regex Invisible = new(@"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\u00AD\uFEFF]", RegexOptions.Compiled);
    static readonly string[] BuiltInDropClasses = { "section-metadata", "metadata" };
    static readonly Regex DivTag = new(@"<(/?)div\b[^>]*>", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    // removes every <div class="...name..."> ... </div> with everything inside (nested divs are counted)
    static string RemoveBlocks(string html, IEnumerable<string> classNames)
    {
        foreach (var cls in classNames)
        {
            var open = new Regex($@"<div\b[^>]*\bclass\s*=\s*[""'][^""']*\b{Regex.Escape(cls)}\b[^""']*[""'][^>]*>", RegexOptions.IgnoreCase);
            Match m;
            while ((m = open.Match(html)).Success)
            {
                int depth = 1, end = html.Length;
                foreach (Match t in DivTag.Matches(html, m.Index + m.Length))
                {
                    depth += t.Groups[1].Length == 0 ? 1 : -1;
                    if (depth == 0) { end = t.Index + t.Length; break; }
                }
                html = html.Remove(m.Index, end - m.Index).Insert(m.Index, "\n");
            }
        }
        return html;
    }

    // page HTML -> readable text: only what a visitor reads. Settings blocks, bare links / paths, ids and
    // style names (bg-truthful-blue, is-subnav-section false, GUIDs ...) are left out.
    string HtmlToText(string html)
    {
        html = Regex.Replace(html, @"<(script|style|noscript|template)[\s\S]*?</\1>", " ", RegexOptions.IgnoreCase);
        html = RemoveBlocks(html, dropClasses);
        html = Regex.Replace(html, @"<a\b[^>]*>\s*(?:https?://[^<\s]+|/[^<\s]*)\s*</a>", " ", RegexOptions.IgnoreCase);   // link whose text is only its address
        html = Regex.Replace(html, @"</(p|div|h[1-6]|li|tr|section|ul|ol|table|blockquote)>|<br\s*/?>", "\n", RegexOptions.IgnoreCase);
        html = Regex.Replace(html, @"<[^>]+>", " ");
        html = Regex.Replace(Decode(html), @"[ \t\r\f\v\u00A0\u2000-\u200A\u202F\u205F\u3000]+", " ");   // incl. &nbsp;
        html = Invisible.Replace(html, "");                                                               // zero-width and direction marks
        html = Regex.Replace(html, @"\s*\n\s*", "\n").Trim();      // one newline between blocks
        return string.Join("\n", html.Split('\n').Select(StripSettings).Where(l => l.Length > 0 && !IsJunk(l)));
    }

    // lines that are not content: a URL or path, a GUID, style / setting names, or (mostly) symbols and numbers
    static readonly Regex PathLike = new(@"^(https?://\S+|www\.\S+|/[\w\-./%~+@:]+/?)$", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    static readonly Regex GuidLike = new(@"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    static readonly Regex ConfigLike = new(@"\b(?:is|has|show|hide)-[a-z0-9]+(?:-[a-z0-9]+)*\b|\bbg-[a-z0-9\-]+|\bfull-bleed[\w\-]*|^\s*(?:style|variant|layout|theme|class|id)\s+[a-z0-9\-, ]+$", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    static readonly Regex CssToken = new(@"^[a-z][a-z0-9]*(-[a-z0-9]+)+$", RegexOptions.Compiled);

    // Settings text that is mixed INTO real text (the section-metadata table flattened into one line):
    //   "is-subnav-section false style bg-truthful-blue, full-bleed-bg <real text> true is-subnav-section false style pad-top-30, ..."
    // The settings are cut out and the real text stays. Marked with \u0001 first so that a true/false next to a setting goes with it
    // (a "true" inside a normal sentence is kept).
    static readonly Regex SettingKey = new(@"\b(?:is|has|show|hide)-[a-z0-9]+(?:-[a-z0-9]+)*\b(?:\s+(?:true|false)(?![\w-]))?", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    static readonly Regex StyleList = new(@"\bstyle\s+(?:[a-z][a-z0-9]*(?:-[a-z0-9]+)+\s*,?\s*)+", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    static readonly Regex StyleName = new(@"(?<![\w-])(?:bg|pad|margin|center|central|underline|full-bleed)(?:-[a-z0-9]+)+,?", RegexOptions.Compiled);
    static readonly Regex SettingMark = new(@"(?:(?<![\w-])(?:true|false)(?![\w-])\s*)?(?:\u0001\s*)+(?:(?<![\w-])(?:true|false)(?![\w-]))?", RegexOptions.Compiled);

    string StripSettings(string s)
    {
        s = SettingKey.Replace(s, "\u0001");
        s = StyleList.Replace(s, "\u0001");
        s = StyleName.Replace(s, "\u0001");
        s = SettingMark.Replace(s, " ");
        foreach (var r in extraRemove) s = r.Replace(s, " ");
        return Regex.Replace(s, @"[ \t]{2,}", " ").Trim();
    }

    // readable one-line text: invisible characters, entities and tags cleaned, settings cut out
    string Tidy(string? s) => StripSettings(CleanLine(s));

    // content -> one readable line (for the snippet under a result), also for pages indexed before the cleaning existed
    string ReadableText(string? content) =>
        string.Join(" ", (content ?? "").Split('\n').Select(Tidy).Where(l => l.Length > 0 && !IsJunk(l)));

    // built-in rules + the patterns from Eds:Cleaning:ExtraLinePatterns
    bool IsJunk(string s) => IsNoise(s) || extraJunk.Any(r => r.IsMatch(s));

    static bool IsNoise(string s)
    {
        s = s.Trim();
        if (s.Length == 0 || PathLike.IsMatch(s) || s.Contains("/content/") || GuidLike.IsMatch(s) || ConfigLike.IsMatch(s)) return true;
        int letters = s.Count(char.IsLetter), nonSpace = s.Count(c => !char.IsWhiteSpace(c));
        if (letters == 0 || (nonSpace >= 15 && letters < nonSpace * 0.5)) return true;
        // lowercase-with-hyphen words (css class names) make up most of the line
        var words = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int css = words.Count(w => CssToken.IsMatch(w.Trim(',', ';')));
        return css >= 2 && css * 2 >= words.Length;
    }

    // &quot; &amp;quot; &#39; &nbsp; ... -> real characters. Repeated, because some sources are escaped twice.
    static string Decode(string s)
    {
        for (int i = 0; i < 3 && s.Contains('&'); i++)
        {
            var d = WebUtility.HtmlDecode(s);
            if (d == s) break;
            s = d;
        }
        return s;
    }

    // one-line text for titles, descriptions, keywords, snippets: readable characters, no tags, single spaces
    static string CleanLine(string? s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        s = Decode(s);
        s = Regex.Replace(s, @"</?[a-zA-Z][^>]*>", " ");                  // stray html tags
        s = Invisible.Replace(s, "");
        return Regex.Replace(s, @"\s+", " ").Trim();                       // \s also matches the non-breaking space
    }

    async Task<Doc?> ToDocAsync(Entry e)
    {
        if (e.Robots.Contains("noindex", StringComparison.OrdinalIgnoreCase)) return null;
        if (Regex.IsMatch(e.Title, @"^\s*404(\D|$)")) return null;     // error pages ("404", "404 - Page not found"): not indexed, page not even fetched
        string content = HtmlToText(e.Content);
        try { content = HtmlToText(await GetStringAsync(PlainUrl(e.Path))); }
        catch (HttpRequestException ex) when (ex.StatusCode == HttpStatusCode.NotFound) { /* no .plain.html: use the content column */ }
        catch (Exception ex) when (ex is HttpRequestException || ex is TaskCanceledException)
        {
            // still failing after the retries: keep what is indexed for this page instead of indexing it half-empty
            failed[e.Path] = 1;
            log.LogWarning("{Path}: could not be fetched ({Msg}); left unchanged", e.Path, ex.Message);
            return null;
        }
        string title = Tidy(e.Title), keywords = Tidy(e.Keywords), description = Tidy(e.Description);
        var hash = Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(
            string.Join("\u0001", title, keywords, description, content))));
        return new Doc(e.Path, title, e.Date, description, e.Image, keywords, content, hash);
    }

    async Task<List<Doc>> BuildDocsAsync(IEnumerable<Entry> entries)
    {
        using var sem = new SemaphoreSlim(Math.Max(1, o.Fetch.Concurrency));
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

    // ---------- writing to the store
    StoreDoc ToStoreDoc(Doc d)
    {
        float[]? v = null;
        if (emb is not null)
        {
            var x = emb.DocVector(d, o.Weights);
            if (x.Any(f => f != 0)) v = x;      // an all-zero vector (page without any text) is not stored
        }
        return new StoreDoc(d, v);
    }

    // ---------- API: reindex
    public Task<object> ReindexAsync(bool wait, bool rebuild = false) => RunJob("reindex", () => ReindexCore(rebuild), wait);

    async Task<object> ReindexCore(bool rebuild)
    {
        // an index already exists and fits the model: compare every page with it (content hash) and change only what differs.
        // Nothing is deleted up front, so search keeps working while this runs.
        if (!rebuild && !store.NeedsRebuild && n > 0) return await UpdateCore(null, deep: true);
        return await BuildNewIndex();
    }

    // full build. Azure: a NEW index next to the live one, the alias moves when it is complete (no empty period).
    // Redis: the index is dropped and built again in place.
    async Task<object> BuildNewIndex()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        SetPhase("fetching query-index");
        var entries = await FetchIndexAsync();
        SetPhase("fetching pages", entries.Count);
        var docs = await BuildDocsAsync(entries);

        SetPhase("preparing the index");
        var session = store.BeginRebuild();
        try
        {
            SetPhase(emb is null ? "writing" : "embedding + writing", docs.Count);
            var batch = new List<StoreDoc>();
            foreach (var d in docs)
            {
                batch.Add(ToStoreDoc(d));
                int cnt = Interlocked.Increment(ref done);
                if (batch.Count >= BatchSize) { session.Add(batch); batch = new List<StoreDoc>(); }
                if (cnt % 20 == 0 || cnt == docs.Count)
                    log.LogInformation("reindex: {Done}/{Total} pages ({Sec:F0}s)", cnt, docs.Count, sw.Elapsed.TotalSeconds);
            }
            if (batch.Count > 0) session.Add(batch);
            SetPhase("finishing");
            await session.CommitAsync(docs.Count);
        }
        catch { session.Abort(); throw; }

        n = docs.Count;
        return new { store = store.Name, indexed = n, skipped = entries.Count - n, failed = failed.Count, seconds = Math.Round(sw.Elapsed.TotalSeconds, 1) };
    }

    // ---------- API: update (incremental)
    public Task<object> UpdateAsync(string[]? paths, bool deep, bool wait) =>
        RunJob("update", () => UpdateCore(paths, deep), wait);

    async Task<object> UpdateCore(string[]? paths, bool deep)
    {
        // nothing indexed yet, or the index was built with another model: it has to be built from scratch
        if (n == 0 || store.NeedsRebuild) return await BuildNewIndex();

        SetPhase("fetching query-index");
        var entries = await FetchIndexAsync();
        var byPath = entries.ToDictionary(e => e.Path);
        SetPhase("reading the index");
        var existing = store.Existing();

        paths = NormalizePaths(paths);
        List<string> check, removed;
        var src = byPath;                                  // where each page's metadata comes from
        if (paths is { Length: > 0 })
        {
            // pages named by the caller: use the query-index entry if there is one, otherwise read
            // title / description / keywords from the page itself (query-index can lag behind a fresh publish)
            src = new Dictionary<string, Entry>();
            foreach (var p in paths)
            {
                if (byPath.TryGetValue(p, out var e)) src[p] = e;
                else if (await FetchEntryFromPageAsync(p) is { } pe) src[p] = pe;
            }
            check = src.Keys.ToList();
            removed = paths.Where(p => !src.ContainsKey(p) && existing.ContainsKey(p)).ToList();   // page is gone (404)
        }
        else
        {
            removed = existing.Keys.Except(byPath.Keys).ToList();
            check = deep
                ? byPath.Keys.ToList()
                : byPath.Keys.Where(p => !existing.ContainsKey(p) || existing[p].date != byPath[p].Date).ToList();
        }

        SetPhase("fetching pages", check.Count);
        var docs = await BuildDocsAsync(check.Select(p => src[p]));
        SetPhase(emb is null ? "writing" : "embedding + writing", docs.Count);
        var okPaths = docs.Select(d => d.Path).ToHashSet();
        removed.AddRange(check.Where(p => !okPaths.Contains(p) && existing.ContainsKey(p) && !failed.ContainsKey(p))); // became noindex (not: could not be fetched)

        int added = 0, updated = 0, unchanged = 0;
        var batch = new List<StoreDoc>();
        foreach (var d in docs)
        {
            Interlocked.Increment(ref done);
            bool had = existing.ContainsKey(d.Path);
            if (had && deep && existing[d.Path].hash == d.Hash) { unchanged++; continue; }
            batch.Add(ToStoreDoc(d));
            if (had) updated++; else added++;
            if (batch.Count >= BatchSize) { store.Upsert(batch); batch = new List<StoreDoc>(); }
        }
        if (batch.Count > 0) store.Upsert(batch);

        var gone = removed.Distinct().Where(existing.ContainsKey).ToList();
        store.Delete(gone);
        n = Math.Max(0, existing.Count + added - gone.Count);

        return new { added, updated, removed = gone.Count, unchanged, failed = failed.Count, total = n };
    }

    // ---------- URLs / paths from the caller
    static string ToPath(string s)
    {
        s = s.Trim();
        if (s.Length == 0) return "";
        if (Uri.TryCreate(s, UriKind.Absolute, out var u) && (u.Scheme == "http" || u.Scheme == "https"))
            s = Uri.UnescapeDataString(u.AbsolutePath);                    // the domain is ignored
        else
        {
            var cut = s.IndexOfAny(new[] { '?', '#' });
            if (cut >= 0) s = s[..cut];
        }
        return s.StartsWith('/') ? s : "/" + s;
    }

    static string[]? NormalizePaths(IEnumerable<string>? items) =>
        items?.Select(ToPath).Where(p => p.Length > 0).Distinct().ToArray();

    static string MetaTag(string html, string key)
    {
        var k = Regex.Escape(key);
        var m = Regex.Match(html, $@"<meta\s+[^>]*?(?:name|property)\s*=\s*[""']{k}[""'][^>]*?content\s*=\s*[""']([^""']*)[""']", RegexOptions.IgnoreCase);
        if (!m.Success)
            m = Regex.Match(html, $@"<meta\s+[^>]*?content\s*=\s*[""']([^""']*)[""'][^>]*?(?:name|property)\s*=\s*[""']{k}[""']", RegexOptions.IgnoreCase);
        return m.Success ? m.Groups[1].Value : "";
    }

    // metadata read from the page's own HTML; null when the page does not exist (404 / 410)
    async Task<Entry?> FetchEntryFromPageAsync(string path)
    {
        using var resp = await GetAsync(o.BaseUrl.TrimEnd('/') + path);
        if (resp.StatusCode is HttpStatusCode.NotFound or HttpStatusCode.Gone) return null;
        resp.EnsureSuccessStatusCode();
        var html = await resp.Content.ReadAsStringAsync();
        var title = Regex.Match(html, @"<title[^>]*>([\s\S]*?)</title>", RegexOptions.IgnoreCase) is { Success: true } t ? t.Groups[1].Value : "";
        var date = (resp.Content.Headers.LastModified ?? DateTimeOffset.UtcNow).ToString("yyyy-MM-dd");
        return new Entry(path, title, date, MetaTag(html, "description"), MetaTag(html, "og:image"),
                         MetaTag(html, "keywords"), MetaTag(html, "robots"), "");
    }

    // ---------- scheduled jobs (see JobScheduler)
    public Task<object> RunScheduledAsync(JobConfig j, bool wait)
    {
        var name = "scheduled:" + j.Name;
        return j.Action.Equals("reindex", StringComparison.OrdinalIgnoreCase)
            ? RunJob(name, () => ReindexCore(j.Rebuild), wait)
            : RunJob(name, () => UpdateCore(NormalizePaths((j.Paths ?? Array.Empty<string>()).Concat(j.Urls ?? Array.Empty<string>())), j.Deep), wait);
    }

    public (string state, string? error, object? result, double seconds) Outcome() =>
        (state, error, result, started is null ? 0 : Math.Round(((finished ?? DateTime.UtcNow) - started.Value).TotalSeconds, 1));

    // ---------- API: search
    public record Hit(string? path, string url, string? title, string? description, string? snippet,
                      string? image, string? date, double score);

    // vector / hybrid need embeddings and an index built with the same model; otherwise text-only
    string ResolveMode(string? mode)
    {
        mode = (mode ?? "hybrid").Trim().ToLowerInvariant();
        if (mode is not ("vector" or "text" or "hybrid")) mode = "hybrid";
        return mode != "text" && (emb is null || store.NeedsRebuild) ? "text" : mode;
    }

    List<Hit> SearchHits(string q, int k, string mode, string? lang, out string usedMode)
    {
        usedMode = ResolveMode(mode);
        if (string.IsNullOrWhiteSpace(q)) return new List<Hit>();
        int fetch = demote.Count > 0 ? Math.Min(Math.Max(k * 4, 50), 200) : k;     // demoted pages may drop out of the top k

        var request = new SearchRequest(
            usedMode != "vector" ? q : null,                       // words (BM25 / full-text)
            usedMode != "text" ? emb!.QueryVector(q) : null,       // meaning (vector)
            fetch, lang);

        var hits = store.Search(request)
            .Select(r => new Hit(r.Path, PublicBase + r.Path, Tidy(r.Title), NoJunk(Tidy(r.Description)),
                                 null, r.Image, r.Date, Math.Round(r.Score, 4)))
            .ToList();
        return Finish(hits, k);
    }

    string NoJunk(string s) => IsJunk(s) ? "" : s;     // a description that is only settings text is not shown

    double DemoteFactor(string path) =>
        demote.Where(r => r.re.IsMatch(path)).Select(r => r.factor).DefaultIfEmpty(1.0).Min();

    // demote (e.g. news), top k, then a snippet from the content for pages without a description
    List<Hit> Finish(List<Hit> hits, int k)
    {
        if (demote.Count > 0)   // lower the score, then re-sort - even a best match moves down
            hits = hits.Select(h => h with { score = Math.Round(h.score * DemoteFactor(h.path!), 4) })
                       .OrderByDescending(h => h.score).ToList();
        return hits.Take(k).Select(h =>
        {
            if (!string.IsNullOrWhiteSpace(h.description)) return h;
            var flat = ReadableText(GetField(h.path!, "content"));
            return h with { snippet = flat.Length > 240 ? flat[..240] + "…" : flat };
        }).ToList();
    }

    string? GetField(string path, string field) => store.GetField(path, field);

    public object Search(string q, int k, string mode, string? lang = null, bool summary = true)
    {
        var hits = SearchHits(q, k, mode, lang, out var used);
        return new { mode = used, summary = summary ? Summarize(q, hits) : null, results = hits };
    }

    // same top results as /api/search, so the [n] citations match the result numbers
    public object? SummaryFor(string q, string mode, string? lang)
    {
        var text = string.Concat(StreamAnswer(q, mode, lang, CancellationToken.None));
        return text.Length == 0 ? null : new { mode = llm is null ? "llm-disabled (extractive)" : "llm", text };
    }

    // ---------- extractive summary: no generation, only sentences that exist on the pages, each with its source
    record Cand(int Doc, int Pos, string Text);
    public record SummarySentence(string text, int source);

    static readonly Regex SentSplit = new(@"(?<=[.!?…])\s+|\n+", RegexOptions.Compiled);

    IEnumerable<string> Sentences(string text)
    {
        var seen = new HashSet<string>();
        foreach (var raw in SentSplit.Split(text))
        {
            var seg = Tidy(raw);
            if (seg.Length == 0) continue;
            foreach (var piece in CutLong(seg, 220))
                if (piece.Length >= 25 && !IsJunk(piece) && seen.Add(piece)) yield return piece;
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

    public record ExtractiveSummary(string text, List<SummarySentence> sentences, List<int> sources);

    // best passages for the query from the top pages: the page description and sentences of the page content,
    // nothing else (no paths, links or titles)
    List<Cand> Pick(string q, List<Hit> hits, int max, bool loose)
    {
        var so = o.Summary;
        var cands = new List<Cand>();
        for (int i = 0; i < Math.Min(so.TopDocs, hits.Count); i++)
        {
            // candidates are only the page's description and its page content (sentences)
            var desc = CleanLine(hits[i].description);
            if (desc.Length >= 25 && !IsJunk(desc)) cands.Add(new Cand(i, -1, desc));
            int pos = 0;
            foreach (var sent in Sentences(GetField(hits[i].path!, "content") ?? "").Take(so.PerDoc))
                cands.Add(new Cand(i, pos++, sent));
        }
        if (cands.Count == 0) return new List<Cand>();

        double[] sims;
        float[][]? vecs = null;
        if (emb is not null)
        {
            var qv = emb.QueryVector(q);
            vecs = emb.PassageVectors(cands.Select(c => c.Text).ToList());
            sims = vecs.Select(v => Dot(qv, v)).ToArray();
        }
        else   // no vectors: share of the query words that appear in the sentence
        {
            var qt = Tokens(q).ToHashSet();
            sims = cands.Select(c => qt.Count == 0 ? 0 : (double)Tokens(c.Text).Distinct().Count(qt.Contains) / qt.Count).ToArray();
        }

        double Score(int i) => sims[i] * (1 - 0.04 * cands[i].Doc);      // small bonus for higher-ranked pages
        var order = Enumerable.Range(0, cands.Count).OrderByDescending(Score).ToList();
        double best = Score(order[0]);
        if (best <= 0) return new List<Cand>();
        double floor = loose ? (emb is null ? best * 0.5 : best - 0.10)
                             : (emb is null ? best * 0.75 : best - 0.06);   // keep passages close to the best one

        var picked = new List<int>();
        foreach (var i in order)
        {
            if (picked.Count >= max || Score(i) < floor) break;
            bool dup = picked.Any(j => cands[j].Text == cands[i].Text || (vecs is not null && Dot(vecs[i], vecs[j]) > 0.93));
            if (!dup) picked.Add(i);
        }
        return picked.OrderBy(i => cands[i].Doc).ThenBy(i => cands[i].Pos).Select(i => cands[i]).ToList();   // reading order
    }

    ExtractiveSummary? Summarize(string q, List<Hit> hits)
    {
        var picked = Pick(q, hits, o.Summary.MaxSentences, false);
        if (picked.Count == 0) return null;
        var sentences = picked.Select(c => new SummarySentence(Clip(c.Text, 320), c.Doc + 1)).ToList();
        return new ExtractiveSummary(
            string.Join(" ", sentences.Select(x => $"{x.text} [{x.source}]")),
            sentences,
            sentences.Select(x => x.source).Distinct().ToList());
    }

    // ---------- LLM answer: writes a new, short answer from the best passages, with [n] citations
    public bool LlmEnabled => llm is not null;

    string BuildPrompt(string q, List<Hit> hits, List<Cand> evidence)
    {
        var sources = new StringBuilder();
        foreach (var g in evidence.GroupBy(c => c.Doc))
        {
            sources.Append($"[{g.Key + 1}] {hits[g.Key].title}\n");
            foreach (var c in g) sources.Append("- ").Append(Clip(c.Text, 400)).Append('\n');
            sources.Append('\n');
        }
        bool thai = q.Any(ch => ch >= '\u0E00' && ch <= '\u0E7F');
        var system =
            "You are the search assistant of a bank website. Answer the question using ONLY the numbered sources provided. " +
            $"Write a short, natural answer in {(thai ? "Thai" : "English")}: 2-4 sentences, or a few short bullet points when listing steps or options. " +
            "Explain in your own words instead of copying sentences. " +
            "After each statement, add the number of the source that supports it in square brackets, like [1] or [2]. " +
            "Never invent facts, numbers, rates or fees that are not in the sources. " +
            "If the sources do not answer the question, say so briefly.";
        return llm!.Format(system, $"Sources:\n{sources}\nQuestion: {q}");
    }

    // pieces of the answer as they are produced; extractive fallback returns the whole text in one piece
    public IEnumerable<string> StreamAnswer(string q, string mode, string? lang, CancellationToken ct)
    {
        var hits = SearchHits(q, o.Summary.TopDocs, mode, lang, out _);
        if (hits.Count == 0) yield break;

        if (llm is null)
        {
            var s = Summarize(q, hits);
            if (s is not null) yield return s.text;
            yield break;
        }

        var evidence = Pick(q, hits, o.Llm.Evidence, true);
        if (evidence.Count == 0) yield break;
        foreach (var piece in llm.Generate(BuildPrompt(q, hits, evidence), ct))
            yield return piece;
    }
}