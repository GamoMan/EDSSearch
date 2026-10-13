using System.Net;
using System.Text.RegularExpressions;

/// <summary>
/// Decides, for every outgoing request (query-index.json and page fetches), whether it goes through the
/// corporate proxy. Reads the Eds:Proxy section from IConfiguration on each request, so editing appsettings.json
/// takes effect without a restart; /api/proxy can also switch it on/off at runtime.
///
///   "Proxy": {
///     "Enabled": true,                       false = every request goes direct
///     "Address": "http://proxy.mycorp.local:8080",   empty = use the system / environment proxy
///     "Mode": "include",                     include: only hosts matching Hosts use the proxy
///                                            exclude: every host EXCEPT those matching Hosts uses the proxy
///     "Hosts": [ ... ],                      empty list = all hosts. Each rule is one of
///                                              www.example.com        exact domain
///                                              *.aem.live  main--*    wildcard (* = any text, ? = one character)
///                                              regex:^main--.*\.aem\.(live|page)$   regular expression (case-insensitive)
///     "UseDefaultCredentials": true,         Windows login; or set User / Password
///     "User": "", "Password": ""
///   }
///
/// (the old single value  "Proxy": "http://..."  still works as the address)
/// </summary>
public class RoutingProxy : IWebProxy
{
    public record Decision(Uri? Proxy, string Reason);

    readonly IConfiguration conf;
    readonly ILogger<RoutingProxy> log;
    int overrideState = -1;               // -1 = follow config, 0 = forced off, 1 = forced on (runtime switch)
    bool? overrideEnabled => Volatile.Read(ref overrideState) switch { 0 => false, 1 => true, _ => null };
    long viaProxy, direct;

    readonly object lk = new();
    string rulesKey = "\0";
    List<Regex> rules = new();
    List<string> badRules = new();

    public RoutingProxy(IConfiguration conf, ILogger<RoutingProxy> log) { this.conf = conf; this.log = log; }

    // ---------- IWebProxy
    public ICredentials? Credentials
    {
        get
        {
            var user = conf["Eds:Proxy:User"];
            if (!string.IsNullOrWhiteSpace(user))
            {
                var pass = conf["Eds:Proxy:Password"] ?? "";
                var i = user.IndexOf('\\');
                return i > 0 ? new NetworkCredential(user[(i + 1)..], pass, user[..i]) : new NetworkCredential(user, pass);
            }
            return conf.GetValue("Eds:Proxy:UseDefaultCredentials", true) ? CredentialCache.DefaultCredentials : null;
        }
        set { }   // credentials come from configuration
    }

    public Uri? GetProxy(Uri destination)
    {
        var d = Decide(destination);
        if (d.Proxy is null) Interlocked.Increment(ref direct); else Interlocked.Increment(ref viaProxy);
        log.LogDebug("{Url} -> {Route} ({Reason})", destination.GetLeftPart(UriPartial.Path), d.Proxy?.Host ?? "direct", d.Reason);
        return d.Proxy;
    }

    public bool IsBypassed(Uri host) => Decide(host).Proxy is null;

    // ---------- the decision
    public Decision Decide(Uri u)
    {
        if (u.IsLoopback) return new(null, "loopback address");

        bool enabled = overrideEnabled ?? conf.GetValue("Eds:Proxy:Enabled", true);
        if (!enabled) return new(null, overrideEnabled is null ? "proxy disabled in config" : "proxy switched off at runtime");

        var addr = Address();
        if (addr.Length == 0)
        {
            // no address configured: behave as before (system / environment proxy, if any)
            var sys = HttpClient.DefaultProxy;
            var su = sys.IsBypassed(u) ? null : sys.GetProxy(u);
            return su is null ? new(null, "no proxy address configured") : new(su, "system default proxy");
        }
        if (!Uri.TryCreate(addr, UriKind.Absolute, out var proxyUri))
            return new(null, $"Proxy:Address '{addr}' is not a valid URL");

        var hosts = Hosts();
        if (hosts.Length == 0) return new(proxyUri, "no host rules: every host uses the proxy");

        bool match = Rules(hosts).Any(r => r.IsMatch(u.Host));
        bool exclude = ModeIsExclude();
        bool use = exclude ? !match : match;
        var reason = exclude
            ? (match ? "host is in the exclude list" : "host is not in the exclude list")
            : (match ? "host matches a proxy rule" : "host matches no proxy rule");
        return new(use ? proxyUri : null, reason);
    }

    // new form: Eds:Proxy:Address; old form: Eds:Proxy as a plain string
    string Address()
    {
        var a = conf["Eds:Proxy:Address"];
        if (string.IsNullOrWhiteSpace(a)) a = conf["Eds:Proxy"];
        return (a ?? "").Trim();
    }

    string[] Hosts() => conf.GetSection("Eds:Proxy:Hosts").Get<string[]>()?
        .Select(h => h.Trim()).Where(h => h.Length > 0).ToArray() ?? Array.Empty<string>();

    bool ModeIsExclude() =>
        (conf["Eds:Proxy:Mode"] ?? "include").Trim().ToLowerInvariant() is "exclude" or "bypass";

    // compiled rules are cached until the list changes
    List<Regex> Rules(string[] hosts)
    {
        var key = string.Join("\n", hosts);
        lock (lk)
        {
            if (key == rulesKey) return rules;
            var list = new List<Regex>();
            var bad = new List<string>();
            foreach (var h in hosts)
            {
                try
                {
                    string pattern =
                        h.StartsWith("regex:", StringComparison.OrdinalIgnoreCase) ? h[6..] :
                        h.Contains('*') || h.Contains('?')
                            ? "^" + Regex.Escape(h).Replace("\\*", ".*").Replace("\\?", ".") + "$"
                            : "^" + Regex.Escape(h) + "$";
                    list.Add(new Regex(pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled));
                }
                catch (ArgumentException e)
                {
                    bad.Add(h);
                    log.LogWarning("Proxy rule '{Rule}' ignored: {Msg}", h, e.Message);
                }
            }
            rules = list; badRules = bad; rulesKey = key;
            return rules;
        }
    }

    // ---------- runtime switch and diagnostics (see /api/proxy)
    public void SetOverride(bool? enabled)
    {
        Volatile.Write(ref overrideState, enabled is null ? -1 : enabled.Value ? 1 : 0);
        log.LogInformation("Proxy runtime override: {State}", enabled is null ? "cleared (config applies)" : enabled.Value ? "ON" : "OFF");
    }

    public object State()
    {
        var addr = Address();
        var hosts = Hosts();
        Rules(hosts);   // refresh invalid-rule list
        bool cfgEnabled = conf.GetValue("Eds:Proxy:Enabled", true);
        return new
        {
            address = Uri.TryCreate(addr, UriKind.Absolute, out var u) ? u.GetLeftPart(UriPartial.Authority) : addr,
            configEnabled = cfgEnabled,
            runtimeOverride = overrideEnabled,
            effectiveEnabled = overrideEnabled ?? cfgEnabled,
            mode = ModeIsExclude() ? "exclude" : "include",
            hosts,
            invalidRules = badRules.ToArray(),
            requestsViaProxy = Interlocked.Read(ref viaProxy),
            requestsDirect = Interlocked.Read(ref direct)
        };
    }

    // "would this URL use the proxy?" - for checking rules without fetching anything
    public object Test(string url)
    {
        if (!Uri.TryCreate(url.Contains("://") ? url : "https://" + url, UriKind.Absolute, out var u))
            return new { url, error = "not a valid URL" };
        var d = Decide(u);
        return new { url, host = u.Host, viaProxy = d.Proxy is not null, proxy = d.Proxy?.GetLeftPart(UriPartial.Authority), reason = d.Reason };
    }
}