using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

public class EmbeddingOptions
{
    public string Provider { get; set; } = "hash";                    // "onnx" | "hash" (hashed TF-IDF)
    public string ModelPath { get; set; } = "models/model.onnx";
    public string TokenizerPath { get; set; } = "models/sentencepiece.bpe.model";
    public int MaxTokens { get; set; } = 512;                         // model limit incl. <s> and </s>
    public int MaxChunks { get; set; } = 6;                           // content is split into windows, at most this many
    public int BatchSize { get; set; } = 16;
    public int Threads { get; set; } = 0;                             // CPU threads for inference; 0 = half of the cores
    public string QueryPrefix { get; set; } = "query: ";              // e5 models need these prefixes
    public string PassagePrefix { get; set; } = "passage: ";
}

/// <summary>
/// Runs an XLM-R based sentence-embedding model (multilingual-e5-*) in-process.
/// Vectors are L2-normalised, so inner product == cosine.
/// </summary>
public sealed class OnnxEmbedder : IDisposable
{
    // XLM-R / fairseq special ids
    const int BosId = 0, PadId = 1, EosId = 2, UnkId = 3;

    readonly EmbeddingOptions o;
    readonly InferenceSession session;
    readonly SentencePieceTokenizer tok;
    readonly bool needsTokenTypes, pooled;
    readonly string outName;
    readonly int[] queryPrefix, passagePrefix;

    public int Dim { get; }

    public OnnxEmbedder(EmbeddingOptions o, ILogger log)
    {
        this.o = o;
        foreach (var f in new[] { o.ModelPath, o.TokenizerPath })
            if (!File.Exists(f))
                throw new FileNotFoundException(
                    $"Embedding file not found: {Path.GetFullPath(f)}. See models/README.md, " +
                    "or set Eds:Embedding:Provider to \"hash\".");

        using (var fs = File.OpenRead(o.TokenizerPath))
            tok = SentencePieceTokenizer.Create(fs, false, false);   // we add <s> / </s> ourselves

        var so = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = o.Threads > 0 ? o.Threads : Math.Max(1, Environment.ProcessorCount / 2),
            InterOpNumThreads = 1
        };
        session = new InferenceSession(o.ModelPath, so);

        needsTokenTypes = session.InputMetadata.ContainsKey("token_type_ids");
        outName = session.OutputMetadata.ContainsKey("sentence_embedding") ? "sentence_embedding"
                : session.OutputMetadata.ContainsKey("last_hidden_state") ? "last_hidden_state"
                : session.OutputMetadata.Keys.First();
        pooled = session.OutputMetadata[outName].Dimensions.Length == 2;   // already [batch, dim]?

        queryPrefix = RawIds(o.QueryPrefix.Trim());
        passagePrefix = RawIds(o.PassagePrefix.Trim());

        // warm-up run also tells us the embedding size, so it never has to be configured by hand
        Dim = Run(new List<int[]> { Build(passagePrefix, RawIds("test")) })[0].Length;
        log.LogInformation("Embedding model loaded: {Model} (dim {Dim}, output '{Out}', pooled={Pooled})",
            Path.GetFileName(o.ModelPath), Dim, outName, pooled);
    }

    // ---------- tokenizing (sentencepiece ids -> XLM-R ids, same mapping as the HF tokenizer)
    int[] RawIds(string text) =>
        tok.EncodeToIds(text).Select(i => i == 0 ? UnkId : i + 1).ToArray();

    static int[] Build(int[] prefix, int[] body)
    {
        var s = new int[prefix.Length + body.Length + 2];
        s[0] = BosId;
        prefix.CopyTo(s, 1);
        body.CopyTo(s, 1 + prefix.Length);
        s[^1] = EosId;
        return s;
    }

    // split long text into windows that fit the model
    IEnumerable<int[]> Windows(int[] ids, int[] prefix, int maxChunks)
    {
        int room = Math.Max(1, o.MaxTokens - 2 - prefix.Length);
        for (int i = 0, c = 0; i < ids.Length && c < maxChunks; i += room, c++)
            yield return Build(prefix, ids[i..Math.Min(i + room, ids.Length)]);
    }

    /// <summary>Token ids as the model sees them (without the e5 prefix) - compare with the Python tokenizer.</summary>
    public int[] DebugIds(string text) => Build(Array.Empty<int>(), RawIds(text));

    // ---------- public API
    public float[] QueryVector(string q)
    {
        var seq = Windows(RawIds(q), queryPrefix, 1).FirstOrDefault();
        return seq is null ? new float[Dim] : Run(new List<int[]> { seq })[0];
    }

    /// <summary>One unit vector per text (first window only) - used to rank sentences for the summary.</summary>
    public float[][] PassageVectors(IReadOnlyList<string> texts)
    {
        var res = new float[texts.Count][];
        var seqs = new List<int[]>();
        var map = new List<int>();
        for (int i = 0; i < texts.Count; i++)
        {
            var seq = Windows(RawIds(texts[i]), passagePrefix, 1).FirstOrDefault();
            if (seq is null) res[i] = new float[Dim];
            else { seqs.Add(seq); map.Add(i); }
        }
        var vecs = Run(seqs);
        for (int j = 0; j < vecs.Length; j++) res[map[j]] = vecs[j];
        return res;
    }

    /// <summary>
    /// Doc vector = (sum_f w_f * unit(embed_f)) * (sumW / sum of w_f for fields that exist).
    /// With a unit query vector, inner product / sumW is the weighted AVERAGE cosine over the fields
    /// the page actually has, so pages without keywords are not penalised.
    /// </summary>
    public float[] DocVector(Doc d, Weights w)
    {
        var texts = new[] { d.Title, d.Keywords, d.Description, d.Content };
        var wts = new[] { w.Title, w.Keywords, w.Description, w.Content };

        var seqs = new List<int[]>();
        var owner = new List<int>();
        for (int f = 0; f < 4; f++)
        {
            if (string.IsNullOrWhiteSpace(texts[f])) continue;
            foreach (var s in Windows(RawIds(texts[f]), passagePrefix, f == 3 ? o.MaxChunks : 1))
            { seqs.Add(s); owner.Add(f); }
        }

        var v = new float[Dim];
        if (seqs.Count == 0) return v;

        var vecs = Run(seqs);
        double wsum = wts.Sum(), present = 0;
        for (int f = 0; f < 4; f++)
        {
            var mine = Enumerable.Range(0, seqs.Count).Where(i => owner[i] == f).ToList();
            if (mine.Count == 0) continue;
            var m = new float[Dim];
            foreach (var i in mine)
                for (int k = 0; k < Dim; k++) m[k] += vecs[i][k];
            Normalize(m);
            present += wts[f];
            for (int k = 0; k < Dim; k++) v[k] += (float)(wts[f] * m[k]);
        }
        if (present > 0)
        {
            var scale = (float)(wsum / present);
            for (int k = 0; k < Dim; k++) v[k] *= scale;
        }
        return v;
    }

    // ---------- inference (batched, padded to the longest sequence in each batch)
    float[][] Run(List<int[]> seqs)
    {
        var result = new float[seqs.Count][];
        var order = Enumerable.Range(0, seqs.Count).OrderBy(i => seqs[i].Length).ToArray();   // similar lengths together

        for (int start = 0; start < order.Length; start += o.BatchSize)
        {
            var idx = order.Skip(start).Take(o.BatchSize).ToArray();
            int B = idx.Length, L = idx.Max(i => seqs[i].Length);

            var ids = new DenseTensor<long>(new[] { B, L });
            var mask = new DenseTensor<long>(new[] { B, L });
            for (int b = 0; b < B; b++)
            {
                var s = seqs[idx[b]];
                for (int t = 0; t < L; t++)
                {
                    ids[b, t] = t < s.Length ? s[t] : PadId;
                    mask[b, t] = t < s.Length ? 1 : 0;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ids),
                NamedOnnxValue.CreateFromTensor("attention_mask", mask)
            };
            if (needsTokenTypes)
                inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(new[] { B, L })));

            using var res = session.Run(inputs);
            var flat = res.First(r => r.Name == outName).AsTensor<float>().ToArray();

            for (int b = 0; b < B; b++)
            {
                float[] v;
                if (pooled)
                {
                    int D = flat.Length / B;
                    v = new float[D];
                    Array.Copy(flat, b * D, v, 0, D);
                }
                else   // mean pooling over real tokens (what e5 expects)
                {
                    int D = flat.Length / (B * L), len = seqs[idx[b]].Length;
                    v = new float[D];
                    for (int t = 0; t < len; t++)
                        for (int k = 0; k < D; k++) v[k] += flat[(b * L + t) * D + k];
                    for (int k = 0; k < D; k++) v[k] /= len;
                }
                Normalize(v);
                result[idx[b]] = v;
            }
        }
        return result;
    }

    static void Normalize(float[] v)
    {
        double n = 0;
        foreach (var x in v) n += (double)x * x;
        n = Math.Sqrt(n);
        if (n > 0) for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / n);
    }

    public void Dispose() => session.Dispose();
}
