/// <summary>
/// Where the search index lives. Two implementations, chosen with Eds:Store in appsettings.json:
///   "redis"  -> RedisStore  (Redis Stack / Redis 8 with the search module)
///   "azure"  -> AzureStore  (Azure AI Search)
/// Everything else (fetching pages, embeddings, jobs, summary, API) is shared and does not know which one is used.
/// </summary>
public record StoreDoc(Doc Doc, float[]? Vector);                                        // a page + its vector (null = none)
public record RawHit(string Path, string? Title, string? Description, string? Image, string? Date, double Score);
// Text = words to look for (null = vector search only), Vector = query vector (null = text search only), both = hybrid
public record SearchRequest(string? Text, float[]? Vector, int Fetch, string? Lang);
public record StoreContext(EdsOptions Options, int? VectorDim, string Fingerprint, ILogger Log, RoutingProxy Proxy);

/// <summary>A full rebuild in progress: Add pages, then Commit (make it the live index) or Abort.</summary>
public interface IRebuildSession
{
    void Add(IReadOnlyList<StoreDoc> docs);
    Task CommitAsync(int expectedDocs);
    void Abort();
}

public interface ISearchStore
{
    string Name { get; }
    bool NeedsRebuild { get; }                                  // index was built with another model / schema
    int Init();                                                 // open or create the index; returns the number of documents in it
    object Status();
    Dictionary<string, (string date, string hash)> Existing();  // path -> (date column, content hash) of everything indexed
    void Upsert(IReadOnlyList<StoreDoc> docs);                  // add or replace
    void Delete(IEnumerable<string> paths);
    IRebuildSession BeginRebuild();
    List<RawHit> Search(SearchRequest request);
    string? GetField(string path, string field);                // e.g. "content"
}