using Microsoft.ML.OnnxRuntimeGenAI;

public class LlmOptions
{
    public bool Enabled { get; set; } = false;
    public string ModelPath { get; set; } = "models/llm";     // folder that contains genai_config.json
    public string Template { get; set; } = "chatml";          // chatml (Qwen, ...) | phi3 (Phi-3 / Phi-3.5)
    public int Evidence { get; set; } = 8;                    // passages given to the model
    public int MaxNewTokens { get; set; } = 300;
}

/// <summary>
/// Runs a small instruct model in-process with ONNX Runtime GenAI (CPU).
/// One generation at a time; the caller can cancel (e.g. browser closed) and generation stops.
/// </summary>
public sealed class LlmSummarizer : IDisposable
{
    const int MaxLength = 4096;     // prompt + answer, must not exceed the model's context window

    readonly LlmOptions o;
    readonly Model model;
    readonly Tokenizer tok;
    readonly SemaphoreSlim gate = new(1, 1);

    public LlmSummarizer(LlmOptions o, ILogger log)
    {
        this.o = o;
        OnnxNative.Prepare(log);
        var dir = AppPaths.Resolve(o.ModelPath);
        var cfg = Path.Combine(dir, "genai_config.json");
        if (!File.Exists(cfg))
            throw new FileNotFoundException(
                $"LLM folder not found or incomplete: {Path.GetFullPath(dir)} (needs genai_config.json). See models/README.md.");
        model = new Model(dir);
        tok = new Tokenizer(model);
        log.LogInformation("LLM loaded from {Path} (template {Template})", o.ModelPath, o.Template);
    }

    public string Format(string system, string user) => o.Template.ToLowerInvariant() switch
    {
        "phi3" => $"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
        _ => $"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    };

    public IEnumerable<string> Generate(string prompt, CancellationToken ct)
    {
        gate.Wait(ct);                       // CPU bound: queue concurrent requests
        try
        {
            using var sequences = tok.Encode(prompt);
            using var gp = new GeneratorParams(model);
            gp.SetSearchOption("max_length", MaxLength);
            using var gen = new Generator(model, gp);
            gen.AppendTokenSequences(sequences);
            using var stream = tok.CreateStream();

            int count = 0;
            while (!gen.IsDone() && count++ < o.MaxNewTokens)
            {
                ct.ThrowIfCancellationRequested();
                gen.GenerateNextToken();
                var piece = LastPiece(gen, stream);
                if (piece.Length > 0) yield return piece;
            }
        }
        finally { gate.Release(); }
    }

    // helper keeps the ReadOnlySpan out of the iterator method
    static string LastPiece(Generator gen, TokenizerStream stream) =>
        stream.Decode(gen.GetSequence(0)[^1]);

    public void Dispose() { tok.Dispose(); model.Dispose(); gate.Dispose(); }
}