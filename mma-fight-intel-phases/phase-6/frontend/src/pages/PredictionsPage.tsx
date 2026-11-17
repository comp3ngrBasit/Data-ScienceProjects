import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { BrainCircuit, Check, Play, Plus, RefreshCw, Sparkles, Upload, X } from "lucide-react";
import { actionsApi } from "../api/actions";
import { api, apiUrl } from "../api/client";
import { ARCH_LABEL, predictionsApi, type Prediction, type PredictionSettings, type StoredPrediction } from "../api/predictions";
import { DistBars, Empty, LevelBadge, OverTime, Panel, ReliabilityBlock, ReportView, Warn, ciTxt, pct } from "../components/predictions/parts";
import { ErrorBlock, LoadingBlock, PageHeader, ProgressBar, Spinner } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const TABS = ["Predict", "Video timeline", "Evaluation", "Models"] as const;
type Tab = (typeof TABS)[number];
type Action = { category: string; start?: number | null; end?: number | null };

export default function PredictionsPage() {
  const [tab, setTab] = useState<Tab>("Predict");
  const [ver, setVer] = useState(0);
  const ov = useData(predictionsApi.overview, [ver], (d) => (d?.training_jobs.some((j) => j.status === "queued" || j.status === "processing") ? 1500 : 0));
  const fighters = useData(predictionsApi.fighters, [ver]);
  const tax = useData(actionsApi.taxonomy, []);
  const [fighter, setFighter] = useState("");
  useEffect(() => {
    if (!fighter && fighters.data?.length) setFighter(fighters.data[0].fighter_key);
  }, [fighters.data, fighter]);

  if (ov.error) return <ErrorBlock message={ov.error} onRetry={() => ov.reload()} />;
  if (!ov.data || !fighters.data || !tax.data) return <LoadingBlock />;
  const o = ov.data;
  const running = o.training_jobs.find((j) => j.status === "queued" || j.status === "processing");
  const cats = tax.data.categories.filter((c) => c.group !== "unknown");
  const labelOf = (k: string) => cats.find((c) => c.key === k)?.label ?? k;

  return (
    <div>
      <PageHeader title="AI Next-Move Prediction" subtitle="Predicts a fighter's next action from recent actions, fighter tendencies, opponent behaviour and position — with calibrated probabilities, evidence and reliability" />
      {!o.active_model && (
        <Warn items={["No trained prediction model yet. Open the Models tab and click “Train & evaluate models”. Training uses only recognised / annotated actions from Phases 1–4 (and any imported real sequences)."]} />
      )}
      {o.active_model && (
        <div className="mb-4 flex flex-wrap items-center gap-3 rounded-md border border-ink-700 bg-ink-850 px-4 py-2.5 text-xs text-ink-300">
          <BrainCircuit className="size-4 text-blood-400" />
          Active model <span className="font-mono text-white">{o.active_model.name}</span> ({ARCH_LABEL[o.active_model.architecture]})
          · {o.active_model.dataset_info?.examples} examples · {o.active_model.dataset_info?.fights} fights
          · held-out top-1 {pct(o.active_model.summary.top1_accuracy, 1)} / top-3 {pct(o.active_model.summary.top3_accuracy, 1)}
          {o.active_model.experimental && <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-amber-300">experimental</span>}
        </div>
      )}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <label className="text-xs"><span className="label-caps mr-2">Fighter</span>
          <select className="input h-9 w-72" value={fighter} onChange={(e) => setFighter(e.target.value)}>
            {fighters.data.map((f) => <option key={f.fighter_key} value={f.fighter_key}>{f.name}{f.source === "external" ? " (imported)" : ""} · {f.actions} actions</option>)}
          </select>
        </label>
      </div>
      <div className="mb-4 flex gap-1 border-b border-ink-700">
        {TABS.map((t) => <button key={t} onClick={() => setTab(t)} className={cn("relative px-3 py-2 text-sm", tab === t ? "text-white" : "text-ink-400 hover:text-ink-100")}>{t}{tab === t && <span className="absolute inset-x-2 -bottom-px h-0.5 bg-blood-500" />}</button>)}
      </div>
      {tab === "Predict" && (o.active_model ? <PredictTab fighter={fighter} cats={cats} labelOf={labelOf} /> : <Empty>Train a model first.</Empty>)}
      {tab === "Video timeline" && (o.active_model ? <VideoTab fighter={fighter} /> : <Empty>Train a model first.</Empty>)}
      {tab === "Evaluation" && <EvaluationTab fighter={fighter} cats={cats} />}
      {tab === "Models" && <ModelsTab running={running} onChanged={() => { setVer((v) => v + 1); }} external={o.external_datasets} />}
    </div>
  );
}

// ============================================================================ predict
function PredictTab({ fighter, cats, labelOf }: { fighter: string; cats: { key: string; label: string; group: string }[]; labelOf: (k: string) => string }) {
  const toast = useToast();
  const status = useData(() => (fighter ? predictionsApi.modelStatus(fighter) : Promise.resolve(undefined)), [fighter]);
  const [hist, setHist] = useState<Action[]>([{ category: "left_jab" }, { category: "right_cross" }]);
  const [add, setAdd] = useState("left_hook");
  const [ctx, setCtx] = useState<{ opponent_action: string; distance: string }>({ opponent_action: "", distance: "" });
  const [mode, setMode] = useState<"immediate" | "contextual">("contextual");
  const [res, setRes] = useState<Prediction>();
  const [seqs, setSeqs] = useState<Awaited<ReturnType<typeof predictionsApi.nextSequence>>>();
  const [busy, setBusy] = useState(false);
  const [actual, setActual] = useState("");
  const [evaluated, setEvaluated] = useState<StoredPrediction>();

  const run = async () => {
    setBusy(true);
    setEvaluated(undefined);
    try {
      const context = { opponent_action: ctx.opponent_action || undefined, distance: ctx.distance ? Number(ctx.distance) : undefined };
      const [p, s] = await Promise.all([
        predictionsApi.predict(fighter, { history: hist, context, mode }),
        predictionsApi.nextSequence(fighter, hist.map((h) => h.category), 3, context),
      ]);
      setRes(p);
      setSeqs(s);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const evaluate = async () => {
    if (!res?.prediction_id || !actual) return;
    try {
      setEvaluated(await predictionsApi.evaluate(fighter, res.prediction_id, actual));
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className="grid gap-4 xl:grid-cols-[24rem_minmax(0,1fr)]">
      <Panel title="Current situation">
        <div className="space-y-4 px-5 py-4 text-sm">
          {status.data?.note && <p className="text-xs text-amber-300">{status.data.note}</p>}
          <div>
            <div className="label-caps mb-2">Recent action sequence (oldest → newest)</div>
            <div className="flex flex-wrap items-center gap-1.5">
              {hist.map((h, i) => (
                <span key={i} className="inline-flex items-center gap-1 rounded bg-ink-750 px-2 py-1 text-xs text-ink-100 ring-1 ring-ink-600">
                  {i === hist.length - 1 && <span className="mr-1 rounded bg-blood-500/20 px-1 text-[9px] uppercase text-blood-300">current</span>}
                  {labelOf(h.category)}
                  <button aria-label="Remove" onClick={() => setHist(hist.filter((_, j) => j !== i))} className="text-ink-500 hover:text-white"><X className="size-3" /></button>
                </span>
              ))}
            </div>
            <div className="mt-2 flex gap-2">
              <select className="input h-8 flex-1 text-xs" value={add} onChange={(e) => setAdd(e.target.value)}>{cats.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}</select>
              <button className="btn-secondary h-8 px-2 text-xs" disabled={hist.length >= 10} onClick={() => setHist([...hist, { category: add }])}><Plus className="size-3.5" />Add</button>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs"><div className="label-caps mb-1">Opponent's last action</div>
              <select className="input h-8 w-full text-xs" value={ctx.opponent_action} onChange={(e) => setCtx({ ...ctx, opponent_action: e.target.value })}><option value="">Unknown</option>{cats.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}</select>
            </label>
            <label className="text-xs"><div className="label-caps mb-1">Distance (torso lengths)</div>
              <input className="input h-8 w-full text-xs" type="number" step="0.1" min="0" placeholder="Unknown" value={ctx.distance} onChange={(e) => setCtx({ ...ctx, distance: e.target.value })} />
            </label>
          </div>
          <label className="text-xs"><div className="label-caps mb-1">Mode</div>
            <select className="input h-8 w-full text-xs" value={mode} onChange={(e) => setMode(e.target.value as "immediate" | "contextual")}>
              <option value="immediate">Immediate — next action</option><option value="contextual">Contextual — also show the effect of opponent / position</option>
            </select>
          </label>
          <button className="btn-primary w-full" disabled={!hist.length || busy || !fighter} onClick={run}>{busy ? <Spinner className="size-4" /> : <Sparkles className="size-4" />}Predict next move</button>
          <p className="text-[11px] text-ink-500">Timing is unknown for actions entered here; use the Video timeline for predictions with full timing, pose and distance context.</p>
        </div>
      </Panel>
      {!res ? <Empty>Build a sequence and click “Predict next move”.</Empty> : (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Top predicted next moves" right={<LevelBadge level={res.reliability.level} />}>
              <div className="px-5 py-4">
                <div className="mb-3 text-xs text-ink-400">After <span className="text-ink-100">{res.sequence.map((s) => s.label).join(" → ")}</span> · model {res.model.version}</div>
                <DistBars items={res.top3} />
                <details className="mt-3 text-xs text-ink-400"><summary className="cursor-pointer">Full distribution ({res.distribution.length} classes, sums to {pct(res.probability_sum)})</summary><div className="mt-2"><DistBars items={res.distribution} highlight={0} /></div></details>
                {res.context_shift && (
                  <div className="mt-4 rounded-md bg-ink-800 px-3 py-2 text-xs text-ink-300">
                    <div className="label-caps mb-1">Effect of opponent / position context</div>
                    Without context: {res.without_context?.map((d) => `${d.label} ${pct(d.probability)}`).join(" · ")}
                    <div className="mt-1">Largest changes: {res.context_shift.map((c) => `${c.label} ${c.change > 0 ? "+" : ""}${pct(c.change, 1)}`).join(" · ")}</div>
                  </div>
                )}
              </div>
            </Panel>
            <Panel title="Prediction reliability"><ReliabilityBlock p={res} /></Panel>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Supporting evidence">
              <div className="space-y-3 px-5 py-4 text-sm">
                {res.evidence.supporting_combinations.length ? res.evidence.supporting_combinations.map((c, i) => (
                  <div key={i} className="flex flex-wrap items-center justify-between gap-2"><span className="text-ink-200">{c.sequence.join(" → ")}</span><span className="text-xs text-ink-400 tabular">{c.count}/{c.of} · {c.fights} fight(s) · {c.scope}</span></div>
                )) : <p className="text-ink-500">No identical sequences in the training data.</p>}
                {res.evidence.context_notes.map((n) => <p key={n} className="text-xs text-ink-300">{n}</p>)}
              </div>
            </Panel>
            <Panel title="Fighter intelligence">
              <div className="space-y-2 px-5 py-4 text-sm">
                <div className="text-xs text-ink-400">{res.evidence.fighter_tendencies.fighter_training_examples} training examples for this fighter · <Link className="text-ink-200 hover:underline" to={`/patterns/${fighter}`}>full pattern analysis</Link></div>
                <div><span className="text-ink-400">Most frequent actions: </span>{res.evidence.fighter_tendencies.most_frequent_actions.map((a) => `${a.action} ${pct(a.share)}`).join(" · ") || "—"}</div>
                <div><span className="text-ink-400">Usual follow-ups after the current action: </span>{res.evidence.fighter_tendencies.usual_follow_ups_after_last_action.map((a) => `${a.action} (${a.count}/${a.of})`).join(" · ") || "—"}</div>
              </div>
            </Panel>
          </div>
          {seqs && (
            <Panel title="Possible next sequences (2–3 actions)">
              <div className="divide-y divide-ink-700/60">
                {seqs.sequences.map((s, i) => <div key={i} className="flex items-center justify-between px-5 py-2 text-sm"><span className="text-ink-200">{s.labels.join(" → ")}</span><span className="text-xs tabular text-ink-400">{pct(s.probability, 1)}</span></div>)}
              </div>
              <p className="px-5 pb-3 text-[11px] text-ink-500">{seqs.note}</p>
            </Panel>
          )}
          {res.prediction_id && (
            <Panel title="What actually happened?">
              <div className="flex flex-wrap items-center gap-3 px-5 py-4 text-sm">
                {evaluated?.evaluation ? (
                  <span className={evaluated.evaluation.correct_top1 ? "text-emerald-300" : evaluated.evaluation.correct_top3 ? "text-sky-300" : "text-blood-300"}>
                    Recorded: {evaluated.evaluation.actual_label} — {evaluated.evaluation.correct_top1 ? "top-1 correct" : evaluated.evaluation.correct_top3 ? "in the top 3" : "not in the top 3"}. The stored prediction is unchanged.
                  </span>
                ) : (
                  <>
                    <select className="input h-8 w-60 text-xs" value={actual} onChange={(e) => setActual(e.target.value)}><option value="">Choose the actual next action…</option>{cats.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}</select>
                    <button className="btn-secondary h-8 text-xs" disabled={!actual} onClick={evaluate}><Check className="size-3.5" />Record outcome</button>
                  </>
                )}
              </div>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================ video timeline
function VideoTab({ fighter }: { fighter: string }) {
  const toast = useToast();
  const videos = useData(() => api.listVideos({ status: "completed", page_size: 100 } as never), []);
  const [vid, setVid] = useState("");
  const detail = useData(() => (vid ? api.getVideo(vid) : Promise.resolve(undefined)), [vid]);
  const tl = useData(() => (vid ? predictionsApi.timeline(vid) : Promise.resolve(undefined)), [vid], (d) => (d?.job && (d.job.status === "queued" || d.job.status === "processing") ? 1000 : 0));
  const [only, setOnly] = useState(true);
  const [now, setNow] = useState(0);
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (!vid && videos.data?.items.length) setVid(videos.data.items[0].id);
  }, [videos.data, vid]);
  const preds = useMemo(() => (tl.data?.predictions ?? []).filter((p) => !only || p.fighter_key === fighter), [tl.data, only, fighter]);
  // time at which each prediction's actual next action starts (from the next stored prediction of the same fighter)
  const actualAt = useMemo(() => {
    const m: Record<string, number | null> = {};
    const all = tl.data?.predictions ?? [];
    for (const p of all) {
      const nxt = all.find((q) => q.fighter_key === p.fighter_key && (q.at_time ?? 0) > (p.at_time ?? 0));
      const last = nxt?.sequence[nxt.sequence.length - 1];
      m[p.id] = p.evaluation?.status === "evaluated" && last?.start != null ? last.start : null;
    }
    return m;
  }, [tl.data]);
  const current = [...preds].reverse().find((p) => (p.at_time ?? 0) <= now + 0.001);
  const run = async () => {
    try {
      await predictionsApi.videoPredict(vid);
      tl.reload(true);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    }
  };
  const job = tl.data?.job;
  const evald = preds.filter((p) => p.evaluation?.status === "evaluated");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select className="input h-9 w-80" value={vid} onChange={(e) => setVid(e.target.value)}>{videos.data?.items.map((v) => <option key={v.id} value={v.id}>{v.title}</option>)}</select>
        <button className="btn-primary" onClick={run} disabled={!vid || job?.status === "processing" || job?.status === "queued"}><Play className="size-4" />Predict through this video</button>
        <label className="flex items-center gap-2 text-xs text-ink-300"><input type="checkbox" checked={only} onChange={(e) => setOnly(e.target.checked)} />Selected fighter only</label>
      </div>
      {job && (job.status === "queued" || job.status === "processing") && <ProgressBar value={job.progress} tone="sky" />}
      {job?.status === "failed" && <Warn items={[job.error_message ?? "Failed"]} />}
      {preds.some((p) => p.in_training_data) && <Warn items={["The active model was trained on this fight: these predictions are in-sample and optimistic. They are reported separately from held-out performance."]} />}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="panel overflow-hidden">
          {detail.data && <video ref={ref} src={apiUrl(detail.data.stream_url)} controls className="aspect-video w-full bg-black" onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)} />}
        </div>
        <Panel title={`Prediction at ${now.toFixed(2)} s`}>
          {current ? (
            <div className="space-y-3 px-5 py-4 text-sm">
              <div className="text-xs text-ink-400">Recent sequence</div>
              <div className="flex flex-wrap gap-1">{current.sequence.map((s, i) => <span key={i} className={cn("rounded px-1.5 py-0.5 text-xs ring-1", i === current.sequence.length - 1 ? "bg-ink-700 text-white ring-ink-500" : "bg-ink-800 text-ink-300 ring-ink-700")}>{s.label}</span>)}</div>
              <div className="text-xs text-ink-400">Predicted next (made at {current.at_time?.toFixed(2)} s, before the outcome)</div>
              <DistBars items={current.top3} />
              <LevelBadge level={current.reliability.level} />
              <div className="border-t border-ink-700 pt-2 text-xs">
                Actual next: {actualAt[current.id] != null && now < (actualAt[current.id] ?? 0)
                  ? <span className="text-ink-500">revealed at {actualAt[current.id]?.toFixed(2)} s</span>
                  : current.evaluation?.status === "evaluated"
                    ? <span className={current.evaluation.correct_top1 ? "text-emerald-300" : current.evaluation.correct_top3 ? "text-sky-300" : "text-blood-300"}>{current.evaluation.actual_label} {current.evaluation.correct_top1 ? "✓ top-1" : current.evaluation.correct_top3 ? "✓ top-3" : "✗"}</span>
                    : <span className="text-ink-500">no follow-up action in this sequence</span>}
              </div>
            </div>
          ) : <Empty>{preds.length ? "Play the video — predictions appear as actions complete." : "No predictions for this video yet."}</Empty>}
        </Panel>
      </div>
      {preds.length > 0 && (
        <Panel title={`Timeline · ${evald.length} evaluated · top-1 ${evald.filter((p) => p.evaluation?.correct_top1).length} · top-3 ${evald.filter((p) => p.evaluation?.correct_top3).length}`} right={<button className="btn-ghost h-7 text-xs" onClick={() => setNow(1e9)}>Reveal all outcomes</button>}>
          <div className="flex items-center gap-4 px-5 pt-3 text-[11px] text-ink-400">
            <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm bg-ink-600" />Actual action</span>
            <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm border border-dashed border-blood-400" />Predicted next</span>
            <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm bg-emerald-500/70" />Actual next — correct</span>
            <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm bg-blood-500/70" />Actual next — wrong</span>
          </div>
          <div className="max-h-[28rem] divide-y divide-ink-700/60 overflow-y-auto">
            {preds.map((p) => (
              <button key={p.id} onClick={() => { if (ref.current && p.at_time != null) ref.current.currentTime = p.at_time; }} className={cn("flex w-full flex-wrap items-center gap-2 px-5 py-2 text-left text-xs hover:bg-ink-800", current?.id === p.id && "bg-ink-800")}>
                <span className="w-14 tabular text-ink-500">{p.at_time?.toFixed(2)} s</span>
                <span className="rounded bg-ink-600 px-1.5 py-0.5 text-ink-100">{p.sequence[p.sequence.length - 1]?.label}</span>
                <span className="text-ink-500">→</span>
                <span className="rounded border border-dashed border-blood-400 px-1.5 py-0.5 text-blood-200">{p.predicted_label} {pct(p.predicted_probability)}</span>
                {actualAt[p.id] != null && now < (actualAt[p.id] ?? 0) ? <span className="text-ink-500">actual revealed at {actualAt[p.id]?.toFixed(2)} s</span>
                  : p.evaluation?.status === "evaluated" ? (
                  <span className={cn("rounded px-1.5 py-0.5 text-white", p.evaluation.correct_top1 ? "bg-emerald-500/70" : p.evaluation.correct_top3 ? "bg-sky-500/60" : "bg-blood-500/70")}>actual: {p.evaluation.actual_label}</span>
                ) : <span className="text-ink-500">no follow-up</span>}
              </button>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

// ============================================================================ evaluation
function EvaluationTab({ fighter }: { fighter: string; cats: unknown[] }) {
  const perf = useData(() => (fighter ? predictionsApi.performance(fighter) : Promise.resolve(undefined)), [fighter]);
  const hist = useData(() => (fighter ? predictionsApi.history(fighter) : Promise.resolve(undefined)), [fighter]);
  if (!perf.data || !hist.data) return <LoadingBlock />;
  const p = perf.data;
  const card = (title: string, a: typeof p.held_out) => (
    <div className="panel px-5 py-4">
      <div className="label-caps">{title}</div>
      <div className="mt-2 flex gap-6">
        <div><div className="font-display text-3xl text-white tabular">{pct(a.top1_rate)}</div><div className="text-[11px] text-ink-500">top-1 · {a.top1}/{a.n} {ciTxt(a.top1_ci95)}</div></div>
        <div><div className="font-display text-3xl text-white tabular">{pct(a.top3_rate)}</div><div className="text-[11px] text-ink-500">top-3 · {a.top3}/{a.n} {ciTxt(a.top3_ci95)}</div></div>
      </div>
    </div>
  );
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">{card("Held-out predictions", p.held_out)}{card("In-sample (fight was in training)", p.in_training_data)}</div>
      <p className="text-xs text-ink-500">{p.note}</p>
      <Panel title="Held-out performance over time">{p.over_time.length ? <div className="px-3 py-3"><OverTime data={p.over_time} /></div> : <Empty>No held-out evaluated predictions yet.</Empty>}</Panel>
      <Panel title={`Prediction history (${hist.data.length})`}>
        {hist.data.length ? (
          <div className="max-h-[28rem] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-ink-850 text-ink-500"><tr><th className="px-5 py-2">When</th><th>Sequence</th><th>Top 3 (as predicted)</th><th>Actual</th><th>Result</th><th>Model</th></tr></thead>
              <tbody className="text-ink-200">
                {hist.data.map((h) => (
                  <tr key={h.id} className="border-t border-ink-700/50">
                    <td className="px-5 py-1.5 whitespace-nowrap text-ink-400">{h.mode === "video" ? `video ${h.at_time?.toFixed(1)} s` : new Date(h.created_at).toLocaleString()}</td>
                    <td>{h.sequence.map((s) => s.label).join(" → ")}</td>
                    <td>{h.top3.map((t) => `${t.label} ${pct(t.probability)}`).join(", ")}</td>
                    <td>{h.evaluation?.actual_label ?? "—"}</td>
                    <td className={h.evaluation?.correct_top1 ? "text-emerald-300" : h.evaluation?.correct_top3 ? "text-sky-300" : h.evaluation?.status === "evaluated" ? "text-blood-300" : "text-ink-500"}>
                      {h.evaluation?.status === "evaluated" ? (h.evaluation.correct_top1 ? "Top-1 ✓" : h.evaluation.correct_top3 ? "Top-3 ✓" : "✗") : "not evaluated"}{h.in_training_data && <span className="ml-1 text-ink-500">(in-sample)</span>}
                    </td>
                    <td className="font-mono text-ink-400">{h.model_version}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty>No predictions for this fighter yet.</Empty>}
      </Panel>
    </div>
  );
}

// ============================================================================ models
function ModelsTab({ running, onChanged, external }: { running?: { progress: number; current_stage?: string | null }; onChanged: () => void; external: { dataset: string; actions: number; fights: number }[] }) {
  const toast = useToast();
  const evalr = useData(() => predictionsApi.evaluation().catch(() => undefined), [running ? 1 : 0]);
  const versions = useData(predictionsApi.versions, [running ? 1 : 0]);
  const jobs = useData(predictionsApi.jobs, [running ? 1 : 0]);
  const [showSettings, setShowSettings] = useState(false);
  const train = async () => {
    try {
      await predictionsApi.train();
      toast("info", "Training queued — models are evaluated on unseen fights");
      onChanged();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    }
  };
  const last = jobs.data?.[0];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <button className="btn-primary" onClick={train} disabled={!!running}>{running ? <Spinner className="size-4" /> : <RefreshCw className="size-4" />}Train & evaluate models</button>
        <button className="btn-secondary" onClick={() => setShowSettings(!showSettings)}>Settings & data import</button>
      </div>
      {running && <div className="panel px-5 py-3"><div className="flex justify-between text-xs text-sky-200"><span>{running.current_stage ?? "Queued"}</span><span className="tabular">{Math.round(running.progress * 100)}%</span></div><ProgressBar value={running.progress} tone="sky" className="mt-1" /></div>}
      {last?.status === "failed" && <Warn items={[`Last training failed: ${last.error_message}`]} />}
      {showSettings && <SettingsImport onChanged={onChanged} external={external} />}
      {evalr.data?.report ? (evalr.data.report.status === "ok" ? <ReportView report={evalr.data.report} /> : <Warn items={[evalr.data.report.reason ?? "Not enough data."]} />) : <Empty>No completed training run yet.</Empty>}
      <Panel title="Model registry">
        {versions.data?.length ? (
          <table className="w-full text-left text-xs">
            <thead className="text-ink-500"><tr className="border-b border-ink-700"><th className="px-5 py-2">Model</th><th>Trained</th><th>Dataset</th><th>Top-1 (CV)</th><th>Log loss</th><th>ECE</th><th>T</th><th /></tr></thead>
            <tbody className="text-ink-200">
              {versions.data.map((m) => (
                <tr key={m.id} className="border-b border-ink-700/50">
                  <td className="px-5 py-1.5 font-mono">{m.name}{m.selected && <span className="ml-2 font-sans text-[10px] uppercase text-blood-300">selected</span>}{m.experimental && <span className="ml-2 font-sans text-[10px] uppercase text-amber-300">experimental</span>}</td>
                  <td>{new Date(m.created_at).toLocaleString()}</td><td className="font-mono">{m.dataset_version}</td>
                  <td className="tabular">{pct(m.summary.top1_accuracy, 1)}</td><td className="tabular">{m.summary.log_loss?.toFixed(3) ?? "—"}</td><td className="tabular">{m.summary.ece?.toFixed(3) ?? "—"}</td><td className="tabular">{m.temperature.toFixed(2)}</td>
                  <td className="px-5 text-right">{m.is_active ? <span className="text-emerald-300">active</span> : <button className="btn-ghost h-7 text-xs" onClick={async () => { await predictionsApi.activate(m.id); onChanged(); versions.reload(); }}>Activate</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <Empty>No models yet.</Empty>}
      </Panel>
    </div>
  );
}

function SettingsImport({ onChanged, external }: { onChanged: () => void; external: { dataset: string; actions: number; fights: number }[] }) {
  const toast = useToast();
  const { data } = useData(predictionsApi.settings, []);
  const [s, setS] = useState<PredictionSettings>();
  const [files, setFiles] = useState<FileList | null>(null);
  const [dataset, setDataset] = useState("boxingvi");
  const [fps, setFps] = useState("30");
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (data) setS(data); }, [data]);
  if (!s) return <LoadingBlock />;
  const save = async () => {
    try { setS(await predictionsApi.saveSettings(s)); toast("success", "Prediction settings saved"); } catch (e) { toast("error", e instanceof Error ? e.message : String(e)); }
  };
  const doImport = async () => {
    if (!files) return;
    setBusy(true);
    let n = 0;
    try {
      for (const f of Array.from(files)) {
        const stem = f.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]/g, "_");
        const fd = new FormData();
        fd.append("file", f); fd.append("dataset", dataset); fd.append("fighter_key", `${dataset}-${stem}`); fd.append("fight_key", `${dataset}-${stem}`);
        fd.append("units", "frames"); fd.append("fps", fps); fd.append("label_map", "boxingvi_orthodox");
        n += (await predictionsApi.importDataset(fd)).imported;
      }
      toast("success", `Imported ${n} actions from ${files.length} file(s)`);
      onChanged();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  };
  const field = (k: keyof PredictionSettings, l: string) => (
    <label className="text-xs"><div className="label-caps mb-1">{l}</div><input type="number" className="input h-8 w-full" value={s[k] as number} onChange={(e) => setS({ ...s, [k]: Number(e.target.value) })} /></label>
  );
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Training settings" right={<button className="btn-primary h-8 px-3 text-xs" onClick={save}>Save</button>}>
        <div className="grid grid-cols-2 gap-3 px-5 py-4">
          {field("history_length", "History length (2–10)")}{field("min_class_count", "Min. examples per class")}{field("folds", "Cross-validation folds")}
          {field("min_examples_deep", "Min. examples for GRU/LSTM")}{field("min_examples_transformer", "Min. examples for Transformer")}{field("min_support", "Min. supporting observations")}
          <label className="text-xs"><div className="label-caps mb-1">Device</div><select className="input h-8 w-full" value={s.device} onChange={(e) => setS({ ...s, device: e.target.value as PredictionSettings["device"] })}><option value="auto">Auto (CUDA if available)</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></label>
          <label className="flex items-center gap-2 pt-5 text-xs text-ink-200"><input type="checkbox" checked={s.include_external} onChange={(e) => setS({ ...s, include_external: e.target.checked })} />Include imported sequences</label>
        </div>
      </Panel>
      <Panel title="Import real action sequences (e.g. BoxingVI)">
        <div className="space-y-3 px-5 py-4 text-sm">
          <p className="text-xs text-ink-400">Annotation sheets with <code>start_frame, end_frame, punch_class</code> rows (one file per video/athlete). Each file becomes one fighter and one fight. BoxingVI labels are mapped for an orthodox stance (jab → left jab, cross → right cross, lead hook → left hook, …).</p>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs"><div className="label-caps mb-1">Dataset name</div><input className="input h-8 w-full" value={dataset} onChange={(e) => setDataset(e.target.value)} /></label>
            <label className="text-xs"><div className="label-caps mb-1">Frames per second</div><input className="input h-8 w-full" value={fps} onChange={(e) => setFps(e.target.value)} /></label>
          </div>
          <input type="file" multiple accept=".xlsx,.xls,.csv" onChange={(e) => setFiles(e.target.files)} className="text-xs text-ink-300" />
          <button className="btn-secondary" disabled={!files || busy} onClick={doImport}>{busy ? <Spinner className="size-4" /> : <Upload className="size-4" />}Import</button>
          {external.length > 0 && <div className="text-xs text-ink-400">Imported: {external.map((x) => `${x.dataset} — ${x.actions} actions, ${x.fights} files`).join(" · ")}</div>}
        </div>
      </Panel>
    </div>
  );
}
