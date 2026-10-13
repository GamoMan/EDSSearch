using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

public class EmbeddingOptions
{
    public string Provider { get; set; } = "hash";                    // "onnx" = vectors from the model below; anything else = no vectors (text-only index)
    public string ModelPath { get; set; } = "models/model.onnx";
    public string TokenizerPath { get; set; } = "models/sentencepiece.bpe.model";   // not used when the model has its own tokenizer
    public bool Extensions { get; set; } = true;                      // register onnxruntime-extensions (models with ai.onnx.contrib ops, e.g. BertTokenizer)
    public int MaxTokens { get; set; } = 512;                         // model limit incl. <s> and </s> (separate tokenizer mode)
    public int MaxChars { get; set; } = 500;                          // chunk size in characters (model-with-own-tokenizer mode)
    public int MaxChunks { get; set; } = 6;                           // content is split into windows, at most this many
    public int BatchSize { get; set; } = 16;
    public string Pooling { get; set; } = "mean";                     // how token vectors become one vector: mean (e5) | cls (bge-m3); ignored if the model already outputs a sentence vector
    public string OutputName { get; set; } = "";                      // force a specific model output; empty = auto
    public int Threads { get; set; } = 0;                             // CPU threads for inference; 0 = half of the cores
    public string QueryPrefix { get; set; } = "query: ";              // e5 models need these prefixes; most other models: ""
    public string PassagePrefix { get; set; } = "passage: ";
}

/// <summary>
/// Runs a sentence-embedding model in-process. Two kinds of model are supported:
///  1. token-id models (multilingual-e5-*): input_ids + attention_mask, tokenized here with SentencePiece;
///  2. models with the tokenizer inside the graph (ai.onnx.contrib:BertTokenizer ...): one string input,
///     needs onnxruntime-extensions.
/// Vectors are L2-normalised, so inner product == cosine.
/// </summary>
public sealed class OnnxEmbedder : IDisposable
{
    // XLM-R / fairseq special ids (token-id mode)
    const int BosId = 0, PadId = 1, EosId = 2, UnkId = 3;

    readonly EmbeddingOptions o;
    readonly InferenceSession session;
    readonly SentencePieceTokenizer? tok;      // null when the model has its own tokenizer
    readonly string? textInput;                // name of the string input (model with its own tokenizer)
    readonly bool needsTokenTypes, pooled;
    readonly string outName;
    readonly int[] queryPrefix = Array.Empty<int>(), passagePrefix = Array.Empty<int>();
    // multimodal graphs also declare image/video/audio inputs; for text-only use they get EMPTY tensors
    readonly List<(string name, Type type, int[] dims)> extraInputs = new();

    bool TextMode => textInput is not null;
    bool cls => o.Pooling.Equals("cls", StringComparison.OrdinalIgnoreCase);
    public int Dim { get; }

    public OnnxEmbedder(EmbeddingOptions o, ILogger log)
    {
        this.o = o;
        OnnxNative.Prepare(log);                                  // load the right onnxruntime.dll first, log what is deployed
        var modelPath = AppPaths.Resolve(o.ModelPath);            // relative paths are relative to the app folder
        var tokenizerPath = AppPaths.Resolve(o.TokenizerPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException(
                $"Embedding model not found: {Path.GetFullPath(modelPath)}. See models/README.md, " +
                "or set Eds:Embedding:Provider to \"hash\".");

        var so = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = o.Threads > 0 ? o.Threads : Math.Max(1, Environment.ProcessorCount / 2),
            InterOpNumThreads = 1
        };
        if (o.Extensions)
        {
            // needed for models that contain ai.onnx.contrib ops (BertTokenizer, ...); harmless for other models
            try { so.RegisterOrtExtensions(); }
            catch (Exception e) { log.LogWarning("onnxruntime-extensions could not be registered: {Msg}", e.Message); }
        }
        session = new InferenceSession(modelPath, so);

        string Describe(IReadOnlyDictionary<string, NodeMetadata> m) =>
            string.Join(", ", m.Select(kv => $"{kv.Key}:{kv.Value.ElementType.Name}[{string.Join(',', kv.Value.Dimensions)}]"));
        log.LogInformation("Embedding model inputs: {In}; outputs: {Out}",
            Describe(session.InputMetadata), Describe(session.OutputMetadata));

        var stringInputs = session.InputMetadata.Where(kv => kv.Value.ElementType == typeof(string)).Select(kv => kv.Key).ToList();
        if (stringInputs.Count > 0)
        {
            if (stringInputs.Count > 1 || session.InputMetadata.Count > 1)
                throw new NotSupportedException(
                    "Only models with ONE string input are supported when the tokenizer is inside the model. Inputs: " +
                    Describe(session.InputMetadata));
            textInput = stringInputs[0];
        }
        else
        {
            if (!File.Exists(tokenizerPath))
                throw new FileNotFoundException(
                    $"Tokenizer file not found: {Path.GetFullPath(tokenizerPath)}. See models/README.md.");
            using (var fs = File.OpenRead(tokenizerPath))
                tok = SentencePieceTokenizer.Create(fs, false, false);   // we add <s> / </s> ourselves
            needsTokenTypes = session.InputMetadata.ContainsKey("token_type_ids");
            foreach (var kv in session.InputMetadata)
                if (kv.Key is not ("input_ids" or "attention_mask" or "token_type_ids"))
                    extraInputs.Add((kv.Key, kv.Value.ElementType, kv.Value.Dimensions.Select(d => d < 0 ? 0 : d).ToArray()));
            if (extraInputs.Count > 0)
                log.LogInformation("Model has extra inputs, fed with empty tensors: {Names}", string.Join(", ", extraInputs.Select(x => x.name)));
            queryPrefix = RawIds(o.QueryPrefix.Trim());
            passagePrefix = RawIds(o.PassagePrefix.Trim());
        }

        outName = o.OutputName.Length > 0 && session.OutputMetadata.ContainsKey(o.OutputName) ? o.OutputName
                : new[] { "sentence_embedding", "dense_vecs", "last_hidden_state" }
                    .FirstOrDefault(n => session.OutputMetadata.ContainsKey(n)) ?? session.OutputMetadata.Keys.First();
        pooled = session.OutputMetadata[outName].Dimensions.Length == 2;   // already [batch, dim]?

        // warm-up run also tells us the embedding size, so it never has to be configured by hand
        Dim = RunUnits(Units("test", false, 1).Cast<object>().ToList())[0].Length;
        log.LogInformation("Embedding model loaded: {Model} (dim {Dim}, output '{Out}', pooled={Pooled}, tokenizer={Tok})",
            Path.GetFileName(modelPath), Dim, outName, pooled, TextMode ? "inside the model" : "SentencePiece");
    }

    // ---------- units: token-id windows (int[]) or text chunks (string), depending on the model
    IEnumerable<object> Units(string text, bool query, int maxChunks) => TextMode
        ? TextWindows(text, query ? o.QueryPrefix : o.PassagePrefix, maxChunks)
        : Windows(RawIds(text), query ? queryPrefix : passagePrefix, maxChunks);

    float[][] RunUnits(List<object> units) => TextMode
        ? RunText(units.Cast<string>().ToList())
        : Run(units.Cast<int[]>().ToList());

    // ---------- token-id mode: sentencepiece ids -> XLM-R ids, same mapping as the HF tokenizer
    int[] RawIds(string text) =>
        tok!.EncodeToIds(text).Select(i => i == 0 ? UnkId : i + 1).ToArray();

    static int[] Build(int[] prefix, int[] body)
    {
        var s = new int[prefix.Length + body.Length + 2];
        s[0] = BosId;
        prefix.CopyTo(s, 1);
        body.CopyTo(s, 1 + prefix.Length);
        s[^1] = EosId;
        return s;
    }

    IEnumerable<int[]> Windows(int[] ids, int[] prefix, int maxChunks)
    {
        int room = Math.Max(1, o.MaxTokens - 2 - prefix.Length);
        for (int i = 0, c = 0; i < ids.Length && c < maxChunks; i += room, c++)
            yield return Build(prefix, ids[i..Math.Min(i + room, ids.Length)]);
    }

    // ---------- own-tokenizer mode: we cannot count tokens, so cut by characters (at a space when possible)
    IEnumerable<string> TextWindows(string text, string prefix, int maxChunks)
    {
        text = string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        int room = Math.Max(50, o.MaxChars - prefix.Length);
        for (int i = 0, c = 0; i < text.Length && c < maxChunks; c++)
        {
            int len = Math.Min(room, text.Length - i);
            if (i + len < text.Length)
            {
                int sp = text.LastIndexOf(' ', i + len - 1, len);
                if (sp > i + room / 2) len = sp - i;
            }
            var piece = text.Substring(i, len).Trim();
            if (piece.Length > 0) yield return prefix + piece;
            i += len;
        }
    }

    /// <summary>Token ids as the model sees them (without prefix) - only in token-id mode.</summary>
    public int[] DebugIds(string text) => TextMode ? Array.Empty<int>() : Build(Array.Empty<int>(), RawIds(text));

    // ---------- public API
    public float[] QueryVector(string q)
    {
        var u = Units(q, true, 1).FirstOrDefault();
        return u is null ? new float[Dim] : RunUnits(new List<object> { u })[0];
    }

    /// <summary>One unit vector per text (first window only) - used to rank sentences for the summary.</summary>
    public float[][] PassageVectors(IReadOnlyList<string> texts)
    {
        var res = new float[texts.Count][];
        var units = new List<object>();
        var map = new List<int>();
        for (int i = 0; i < texts.Count; i++)
        {
            var u = Units(texts[i], false, 1).FirstOrDefault();
            if (u is null) res[i] = new float[Dim];
            else { units.Add(u); map.Add(i); }
        }
        var vecs = RunUnits(units);
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

        var units = new List<object>();
        var owner = new List<int>();
        for (int f = 0; f < 4; f++)
        {
            if (string.IsNullOrWhiteSpace(texts[f])) continue;
            foreach (var u in Units(texts[f], false, f == 3 ? o.MaxChunks : 1))
            { units.Add(u); owner.Add(f); }
        }

        var v = new float[Dim];
        if (units.Count == 0) return v;

        var vecs = RunUnits(units);
        double wsum = wts.Sum(), present = 0;
        for (int f = 0; f < 4; f++)
        {
            var mine = Enumerable.Range(0, units.Count).Where(i => owner[i] == f).ToList();
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

    // ---------- inference, token-id mode (batched, padded to the longest sequence in each batch)
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
            foreach (var (name, type, dims) in extraInputs)
                inputs.Add(type == typeof(float) ? NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(dims))
                         : type == typeof(long) ? NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(dims))
                         : throw new NotSupportedException($"Unsupported extra model input '{name}' ({type.Name}); send this message so it can be added."));

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
                else if (cls)   // first token (<s>), what bge-m3 uses
                {
                    int D = flat.Length / (B * L);
                    v = new float[D];
                    Array.Copy(flat, b * L * D, v, 0, D);
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

    // ---------- inference, model-with-own-tokenizer mode: the graph takes raw strings
    float[][] RunText(List<string> texts)
    {
        var result = new float[texts.Count][];
        // a token-level output cannot be mean-pooled correctly across padding -> one text at a time
        int batch = pooled ? Math.Max(1, o.BatchSize) : 1;
        int rank = session.InputMetadata[textInput!].Dimensions.Length;

        for (int start = 0; start < texts.Count; start += batch)
        {
            var chunk = texts.Skip(start).Take(batch).ToArray();
            var t = new DenseTensor<string>(rank >= 2 ? new[] { chunk.Length, 1 } : new[] { chunk.Length });
            for (int b = 0; b < chunk.Length; b++)
            {
                if (rank >= 2) t[b, 0] = chunk[b]; else t[b] = chunk[b];
            }

            using var res = session.Run(new[] { NamedOnnxValue.CreateFromTensor(textInput!, t) });
            var tensor = res.First(r => r.Name == outName).AsTensor<float>();
            var flat = tensor.ToArray();
            int D = tensor.Dimensions[tensor.Dimensions.Length - 1];

            for (int b = 0; b < chunk.Length; b++)
            {
                float[] v = new float[D];
                if (pooled) Array.Copy(flat, b * D, v, 0, D);
                else if (cls) Array.Copy(flat, 0, v, 0, D);   // [1, L, D] -> first token
                else        // [1, L, D] -> mean over L
                {
                    int L = flat.Length / D;
                    for (int p = 0; p < L; p++)
                        for (int k = 0; k < D; k++) v[k] += flat[p * D + k];
                    for (int k = 0; k < D; k++) v[k] /= L;
                }
                Normalize(v);
                result[start + b] = v;
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