using System.Diagnostics;
using System.Runtime.InteropServices;

/// <summary>Relative paths in appsettings.json (models/...) are resolved against the app folder, not the process's working directory.</summary>
public static class AppPaths
{
    public static string Root { get; set; } = AppContext.BaseDirectory;      // set to the content root at startup
    public static string Resolve(string p) =>
        string.IsNullOrWhiteSpace(p) || Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(Root, p));
}

/// <summary>
/// Makes sure the onnxruntime.dll that was DEPLOYED WITH THE APP is the one that gets loaded, and logs what is going on.
/// "TypeInitializationException ... NativeMethods ... NullReferenceException" means the managed Microsoft.ML.OnnxRuntime
/// found a native onnxruntime.dll that is older / different (e.g. one from the Windows system folder) or none at all.
/// Loading the right file by its full path first fixes the first case; the log shows the other two.
/// </summary>
public static class OnnxNative
{
    static bool done;
    static readonly object lk = new();

    public static void Prepare(ILogger log)
    {
        lock (lk)
        {
            if (done) return;
            done = true;

            log.LogInformation("Process: {Arch}, {Bits}-bit, {Os}; content root {Root}",
                RuntimeInformation.ProcessArchitecture, Environment.Is64BitProcess ? 64 : 32, RuntimeInformation.OSDescription, AppPaths.Root);
            if (!OperatingSystem.IsWindows()) return;

            var rid = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X64 => "win-x64", Architecture.X86 => "win-x86", Architecture.Arm64 => "win-arm64", _ => null
            };

            var sys = Path.Combine(Environment.SystemDirectory, "onnxruntime.dll");
            if (File.Exists(sys))
                log.LogWarning("Windows has its own onnxruntime.dll ({Ver}) in {Dir}; it can be loaded instead of the one that comes with the app",
                    Version(sys), Environment.SystemDirectory);

            var candidates = new List<string>();
            if (rid is not null) candidates.Add(Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "onnxruntime.dll"));
            candidates.Add(Path.Combine(AppContext.BaseDirectory, "onnxruntime.dll"));

            var file = candidates.FirstOrDefault(File.Exists);
            if (file is null)
            {
                log.LogError("onnxruntime.dll was not deployed. Looked in: {Paths}. Publish with: dotnet publish -c Release -r win-x64 --self-contained false",
                    string.Join("; ", candidates));
                return;
            }
            try
            {
                NativeLibrary.Load(file);      // by full path: later loads of "onnxruntime" find this module
                log.LogInformation("Loaded onnxruntime.dll {Ver} from {File}", Version(file), file);
            }
            catch (Exception e) { log.LogError("Could not load {File}: {Msg}", file, e.Message); }
        }
    }

    static string Version(string f)
    {
        try { return FileVersionInfo.GetVersionInfo(f).FileVersion ?? "?"; } catch { return "?"; }
    }
}