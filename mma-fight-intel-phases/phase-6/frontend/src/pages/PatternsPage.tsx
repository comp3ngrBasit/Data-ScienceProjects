import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { BrainCircuit, CalendarRange, Filter, Network, RefreshCw, Settings2, X } from "lucide-react";
import { patternsApi, type PatternFilters } from "../api/patterns";
import { MetaLine, Warnings, num } from "../components/patterns/common";
import { CombinationsTab, CompareTab, ExplorerTab, FightsPanel, OpponentsTab, SettingsPanel, TendenciesTab, TransitionsTab } from "../components/patterns/tabs";
import { EmptyState, ErrorBlock, LoadingBlock, PageHeader, ProgressBar, Spinner, StatCard } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const TABS = ["Tendencies", "Combinations", "Transitions", "Sequence explorer", "Opponents", "Compare"] as const;
type Tab = (typeof TABS)[number];
const STATE = {
  up_to_date: { label: "Analysed", cls: "text-emerald-300" },
  stale: { label: "Out of date", cls: "text-amber-300" },
  not_analysed: { label: "Not analysed", cls: "text-ink-500" },
  processing: { label: "Analysing…", cls: "text-sky-300" },
} as const;

export default function PatternsPage() {
  const { fighter } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("Tendencies");
  const [panel, setPanel] = useState<"" | "fights" | "settings">("");
  const [filters, setFilters] = useState<PatternFilters>({});
  const [version, setVersion] = useState(0);
  const fkey = JSON.stringify(filters) + version;

  const ov = useData(patternsApi.overview, [version], (d) => (d?.jobs.some((j) => j.status === "queued" || j.status === "processing") ? 1500 : 0));
  const status = useData(() => (fighter ? patternsApi.status(fighter) : Promise.resolve(undefined)), [fighter, version],
    (d) => (d?.state === "processing" ? 1000 : 0));
  const summary = useData(() => (fighter ? patternsApi.summary(fighter, filters) : Promise.resolve(undefined)), [fighter, fkey]);

  // after queuing an analysis, poll until it is no longer running, then refresh everything once
  const [pending, setPending] = useState(false);
  useEffect(() => {
    if (!pending) return;
    const t = window.setInterval(async () => {
      if (!fighter) return;
      const st = await patternsApi.status(fighter).catch(() => undefined);
      if (st && st.state !== "processing") {
        setPending(false);
        setVersion((v) => v + 1);
      }
    }, 800);
    return () => window.clearInterval(t);
  }, [pending, fighter]);

  useEffect(() => {
    if (!fighter && ov.data?.fighters.length) navigate(`/patterns/${ov.data.fighters[0].fighter_key}`, { replace: true });
  }, [fighter, ov.data, navigate]);
  useEffect(() => setFilters({}), [fighter]);

  const current = ov.data?.fighters.find((f) => f.fighter_key === fighter);
  const fightOptions = useMemo(() => {
    const vids = ov.data?.videos ?? [];
    const keys = summary.data?.meta.available_fights ?? current?.fight_keys ?? [];
    return keys.map((k) => {
      const v = vids.filter((x) => x.fight_key === k);
      return { key: k, label: v[0]?.event || v.map((x) => x.title).join(" + ") || k, date: v[0]?.fight_date };
    });
  }, [ov.data, summary.data, current]);

  const analyze = async (force = false) => {
    if (!fighter) return;
    try {
      await patternsApi.analyze(fighter, force);
      toast("info", "Pattern analysis queued");
      status.reload(true);
      setPending(true);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    }
  };
  const analyzeAll = async () => {
    try {
      await patternsApi.analyzeAll();
      toast("info", "Analysis of all fighters queued — unchanged fighters are skipped");
      ov.reload(true);
      setPending(true);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    }
  };

  if (ov.error) return <ErrorBlock message={ov.error} onRetry={() => ov.reload()} />;
  if (!ov.data) return <LoadingBlock />;
  const o = ov.data;
  const allJob = o.jobs.find((j) => j.fighter_key === "*" && (j.status === "queued" || j.status === "processing"));
  const st = status.data;
  const s = summary.data;

  return (
    <div>
      <PageHeader
        title="Fighter Pattern Intelligence"
        subtitle="Combinations, conditional probabilities, Markov transitions, tendencies and opponent analysis — computed only from recognised or annotated actions"
        actions={
          <div className="flex gap-2">
            <button className={cn("btn-secondary", panel === "fights" && "ring-1 ring-blood-500")} onClick={() => setPanel(panel === "fights" ? "" : "fights")}><CalendarRange className="size-4" />Fights & dates</button>
            <button className={cn("btn-secondary", panel === "settings" && "ring-1 ring-blood-500")} onClick={() => setPanel(panel === "settings" ? "" : "settings")}><Settings2 className="size-4" />Settings</button>
            <button className="btn-primary" onClick={analyzeAll} disabled={!!allJob}>{allJob ? <Spinner className="size-4" /> : <RefreshCw className="size-4" />}Analyse all</button>
          </div>
        }
      />
      {allJob && <div className="mb-4 panel px-5 py-3"><div className="flex justify-between text-xs text-sky-200"><span>{allJob.current_stage ?? "Queued"}</span><span className="tabular">{Math.round(allJob.progress * 100)}%</span></div><ProgressBar value={allJob.progress} tone="sky" className="mt-1" /></div>}
      {panel === "fights" && <div className="mb-4"><FightsPanel onChanged={() => setVersion((v) => v + 1)} /></div>}
      {panel === "settings" && <div className="mb-4"><SettingsPanel onSaved={() => setVersion((v) => v + 1)} /></div>}

      {!o.fighters.length ? (
        <EmptyState icon={<BrainCircuit className="size-6" />} title="No action data to analyse yet"
          body={<>Pattern analysis uses Phase 3 output. Run <Link className="text-blood-300 hover:underline" to="/actions">action recognition</Link> on analysed videos (or mark videos fully annotated with verified annotations), then return here. {o.extraction.excluded_unattributed ? `${o.extraction.excluded_unattributed} actions could not be attributed to a fighter or track.` : ""}</>} />
      ) : (
        <div className="grid gap-5 xl:grid-cols-[17rem_minmax(0,1fr)]">
          <aside className="panel h-fit">
            <div className="panel-header"><h3 className="label-caps">Fighters ({o.fighters.length})</h3></div>
            <div className="max-h-[70vh] divide-y divide-ink-700/60 overflow-y-auto">
              {o.fighters.map((f) => (
                <Link key={f.fighter_key} to={`/patterns/${f.fighter_key}`} className={cn("block px-4 py-2.5 hover:bg-ink-800", f.fighter_key === fighter && "bg-ink-800 shadow-[inset_2px_0_0] shadow-blood-500")}>
                  <div className={cn("truncate text-sm", f.is_track ? "text-ink-300" : "text-white")}>{f.name}</div>
                  <div className="mt-0.5 flex justify-between text-[11px] text-ink-500"><span className="tabular">{f.actions} actions · {f.fights} fight{f.fights === 1 ? "" : "s"}</span><span className={STATE[f.analysis].cls}>{STATE[f.analysis].label}</span></div>
                </Link>
              ))}
            </div>
            <div className="border-t border-ink-700/60 px-4 py-3 text-[11px] text-ink-500">
              Unassigned tracks are listed separately per video. Assign tracks to fighter profiles in <Link to="/tracking" className="text-ink-300 hover:underline">Fighter Tracking</Link> to combine a fighter's footage across videos.
            </div>
          </aside>

          <div className="min-w-0 space-y-4">
            {!fighter || !current && !s ? <LoadingBlock /> : (
              <>
                <div className="panel px-5 py-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h2 className="font-display text-3xl font-bold uppercase tracking-wide text-white">{s?.name ?? current?.name}</h2>
                      <div className="mt-1 text-xs text-ink-400">
                        {current?.stance && <span className="mr-3">Stance: {current.stance}</span>}
                        {s?.summary.period ? <span>Period {s.summary.period.from} – {s.summary.period.to}</span> : <span>No fight dates set</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {st && <span className={cn("text-xs", STATE[st.state].cls)}>{STATE[st.state].label}{st.snapshot && st.state !== "processing" && <span className="text-ink-500"> · {new Date(st.snapshot.created_at).toLocaleString()}</span>}</span>}
                      <button className="btn-primary" onClick={() => analyze(st?.state === "up_to_date")} disabled={st?.state === "processing"}>
                        {st?.state === "processing" ? <Spinner className="size-4" /> : <Network className="size-4" />}{st?.state === "up_to_date" ? "Re-run analysis" : "Run analysis"}
                      </button>
                    </div>
                  </div>
                  {st?.state === "processing" && st.job && <ProgressBar value={st.job.progress} tone="sky" className="mt-3" />}
                  {st?.job?.status === "failed" && <p className="mt-2 text-xs text-blood-300">Last analysis failed: {st.job.error_message}</p>}

                  <div className="mt-4 flex flex-wrap items-end gap-3 border-t border-ink-700/60 pt-4">
                    <Filter className="mb-2 size-4 text-ink-500" />
                    <label className="text-xs"><div className="label-caps mb-1">Fight</div>
                      <select className="input h-9 w-64" value={filters.fight?.[0] ?? ""} onChange={(e) => setFilters({ ...filters, fight: e.target.value ? [e.target.value] : undefined })}>
                        <option value="">All fights ({fightOptions.length})</option>
                        {fightOptions.map((f) => <option key={f.key} value={f.key}>{f.label}{f.date ? ` · ${f.date}` : ""}</option>)}
                      </select>
                    </label>
                    <label className="text-xs"><div className="label-caps mb-1">From</div><input type="date" className="input h-9" value={filters.date_from ?? ""} onChange={(e) => setFilters({ ...filters, date_from: e.target.value || undefined })} /></label>
                    <label className="text-xs"><div className="label-caps mb-1">To</div><input type="date" className="input h-9" value={filters.date_to ?? ""} onChange={(e) => setFilters({ ...filters, date_to: e.target.value || undefined })} /></label>
                    <label className="text-xs"><div className="label-caps mb-1">Min. confidence</div>
                      <select className="input h-9 w-32" value={filters.min_confidence ?? ""} onChange={(e) => setFilters({ ...filters, min_confidence: e.target.value ? Number(e.target.value) : undefined })}>
                        <option value="">Settings ({o.settings.min_confidence})</option>{[0.5, 0.6, 0.7, 0.8, 0.9].map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </label>
                    {(filters.fight || filters.date_from || filters.date_to || filters.min_confidence != null) && <button className="btn-ghost h-9 text-xs" onClick={() => setFilters({})}><X className="size-3.5" />Clear</button>}
                  </div>
                </div>

                {summary.error && <ErrorBlock message={summary.error} onRetry={() => summary.reload()} />}
                {s && (
                  <>
                    <Warnings items={s.warnings} />
                    <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
                      <StatCard label="Actions" value={s.summary.actions} hint={Object.entries(s.summary.sources).map(([k, v]) => `${v} ${k}`).join(" · ") || "—"} accent />
                      <StatCard label="Sequences" value={s.summary.sequences} hint={`${s.summary.multi_action_sequences} with ≥2 actions · mean ${num(s.summary.mean_sequence_length, 1)}`} />
                      <StatCard label="Fights" value={s.summary.fights} hint={`${s.summary.videos} video(s) · ${s.summary.minutes_observed} min`} />
                      <StatCard label="Combinations" value={s.summary.cross_fight_patterns + s.summary.single_fight_patterns} hint={`${s.summary.cross_fight_patterns} cross-fight · ${s.summary.single_fight_patterns} single-fight`} />
                      <StatCard label="Reliable probabilities" value={`${s.summary.probabilities_meeting_thresholds}/${s.summary.probabilities_total}`} hint="meet the minimum sample size" />
                    </div>
                    <MetaLine meta={s.meta} />
                    <div className="flex gap-1 overflow-x-auto border-b border-ink-700">
                      {TABS.map((t) => (
                        <button key={t} onClick={() => setTab(t)} className={cn("relative whitespace-nowrap px-3 py-2 text-sm", tab === t ? "text-white" : "text-ink-400 hover:text-ink-100")}>
                          {t}{tab === t && <span className="absolute inset-x-2 -bottom-px h-0.5 bg-blood-500" />}
                        </button>
                      ))}
                    </div>
                    {tab === "Tendencies" && <TendenciesTab fighter={fighter} filters={filters} fkey={fkey} />}
                    {tab === "Combinations" && <CombinationsTab fighter={fighter} filters={filters} fkey={fkey} />}
                    {tab === "Transitions" && <TransitionsTab fighter={fighter} filters={filters} fkey={fkey} states={s.states} labels={s.state_labels} maxOrder={s.config.max_order} />}
                    {tab === "Sequence explorer" && <ExplorerTab fighter={fighter} filters={filters} fkey={fkey} />}
                    {tab === "Opponents" && <OpponentsTab fighter={fighter} filters={filters} fkey={fkey} />}
                    {tab === "Compare" && <CompareTab fighters={o.fighters} filters={filters} fkey={fkey} initial={fighter} />}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
