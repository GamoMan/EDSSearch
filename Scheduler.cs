using Cronos;

/// <summary>One scheduled job (Eds:Jobs in appsettings.json).</summary>
public class JobConfig
{
    public string Name { get; set; } = "";
    public bool Enabled { get; set; } = true;
    public string Cron { get; set; } = "";             // 5 fields "min hour day month weekday" (6 with a leading seconds field)
    public string Action { get; set; } = "update";     // update | reindex
    public bool Deep { get; set; }                     // update: compare page content instead of the date column
    public bool Rebuild { get; set; }                  // reindex: drop and rebuild from scratch (normally false = sync)
    public string[]? Paths { get; set; }               // update only these pages ...
    public string[]? Urls { get; set; }                // ... given as paths or full URLs
}

/// <summary>
/// Runs the jobs from Eds:Jobs at their cron times (Eds:TimeZone). Index jobs never overlap: if another
/// job (or a manual reindex/update) is running when a job is due, it waits and starts as soon as the index is free.
/// </summary>
public class JobScheduler : BackgroundService
{
    class JobState
    {
        public JobConfig Cfg = new();
        public CronExpression? Cron;
        public DateTime? Next;             // UTC
        public DateTime? LastStart;        // UTC
        public string? LastState, LastError;
        public object? LastResult;
        public double LastSeconds;
        public string? Problem;
    }

    readonly SearchService svc;
    readonly ILogger<JobScheduler> log;
    readonly TimeZoneInfo tz;
    readonly List<JobState> jobs = new();

    public JobScheduler(EdsOptions o, SearchService svc, ILogger<JobScheduler> log)
    {
        this.svc = svc; this.log = log;
        tz = Zone(o.TimeZone);

        foreach (var c in o.Jobs)
        {
            var st = new JobState { Cfg = c };
            if (string.IsNullOrWhiteSpace(c.Name)) st.Problem = "Name is missing";
            else if (!c.Action.Equals("update", StringComparison.OrdinalIgnoreCase) &&
                     !c.Action.Equals("reindex", StringComparison.OrdinalIgnoreCase))
                st.Problem = $"unknown Action '{c.Action}' (use update or reindex)";
            else
            {
                try
                {
                    int fields = c.Cron.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
                    st.Cron = CronExpression.Parse(c.Cron, fields == 6 ? CronFormat.IncludeSeconds : CronFormat.Standard);
                    if (c.Enabled) st.Next = st.Cron.GetNextOccurrence(DateTime.UtcNow, tz);
                }
                catch (CronFormatException e) { st.Problem = "bad Cron: " + e.Message; }
            }
            if (st.Problem is not null) log.LogError("Job '{Name}' ignored: {Problem}", c.Name, st.Problem);
            else log.LogInformation("Job '{Name}' ({Action}) {State}, next run {Next}", c.Name, c.Action,
                c.Enabled ? "scheduled" : "disabled", Local(st.Next));
            jobs.Add(st);
        }
    }

    TimeZoneInfo Zone(string id)
    {
        try { return string.IsNullOrWhiteSpace(id) ? TimeZoneInfo.Local : TimeZoneInfo.FindSystemTimeZoneById(id); }
        catch (Exception e)
        {
            log.LogWarning("Time zone '{Id}' not found ({Msg}); using the server's local time zone", id, e.Message);
            return TimeZoneInfo.Local;
        }
    }

    string? Local(DateTime? utc) =>
        utc is null ? null : TimeZoneInfo.ConvertTimeFromUtc(DateTime.SpecifyKind(utc.Value, DateTimeKind.Utc), tz).ToString("yyyy-MM-dd HH:mm:ss");

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(TimeSpan.FromSeconds(10), ct); }
            catch (OperationCanceledException) { break; }

            foreach (var j in jobs)
            {
                if (j.Problem is not null || !j.Cfg.Enabled || j.Cron is null || j.Next is null || j.Next > DateTime.UtcNow) continue;
                await RunAsync(j);
            }
        }
    }

    async Task RunAsync(JobState j)
    {
        var t0 = DateTime.UtcNow;
        try
        {
            log.LogInformation("Job '{Name}': starting ({Action})", j.Cfg.Name, j.Cfg.Action);
            await svc.RunScheduledAsync(j.Cfg, wait: true);
        }
        catch (InvalidOperationException)
        {
            // another index job is running: stay "due" and try again on the next tick (10 s)
            log.LogDebug("Job '{Name}': waiting, another index job is running", j.Cfg.Name);
            return;
        }

        var (state, error, result, seconds) = svc.Outcome();
        j.LastStart = t0; j.LastState = state; j.LastError = error; j.LastResult = result; j.LastSeconds = seconds;
        j.Next = j.Cron!.GetNextOccurrence(DateTime.UtcNow, tz);      // from now: a long job never causes a pile-up
        log.LogInformation("Job '{Name}': {State} in {Sec}s, next run {Next}", j.Cfg.Name, state, seconds, Local(j.Next));
    }

    /// <summary>Start a job now (POST /api/jobs/{name}/run); runs in the background, follow it with /api/progress.</summary>
    public Task<object> RunNowAsync(string name)
    {
        var j = jobs.FirstOrDefault(x => x.Cfg.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                ?? throw new KeyNotFoundException($"No job named '{name}'");
        if (j.Problem is not null) throw new InvalidOperationException($"Job '{name}' is not valid: {j.Problem}");
        return svc.RunScheduledAsync(j.Cfg, wait: false);
    }

    public object List() => new
    {
        timeZone = tz.Id,
        jobs = jobs.Select(j => new
        {
            name = j.Cfg.Name, action = j.Cfg.Action, cron = j.Cfg.Cron, enabled = j.Cfg.Enabled, problem = j.Problem,
            nextRun = j.Cfg.Enabled ? Local(j.Next) : null,
            lastRun = Local(j.LastStart), lastState = j.LastState, lastSeconds = j.LastSeconds,
            lastResult = j.LastResult, lastError = j.LastError
        })
    };
}