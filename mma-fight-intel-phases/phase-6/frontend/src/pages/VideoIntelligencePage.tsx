import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { CheckCircle2, CircleDashed, CircleSlash, Loader2, Play, RotateCw, XCircle } from "lucide-react";
import { ApiError } from "../api/client";
import { actionsApi } from "../api/actions";
import { platformApi, type Stage } from "../api/platform";
import { visionApi, type Tracking } from "../api/vision";
import ActionHistory from "../components/actions/ActionHistory";
import ActionTimeline from "../components/actions/ActionTimeline";
import { useTaxonomy } from "../components/actions/common";
import { PredictionCard } from "../components/platform/PredictionCard";
import { pct } from "../components/platform/FightDashboard";
import { EmptyState, ErrorBlock, LoadingBlock, PageHeader, ProgressBar, Spinner } from "../components/ui";
import OverlayPlayer, { type OverlayToggles, type PlayerHandle } from "../components/vision/OverlayPlayer";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const TOGGLES: OverlayToggles = { boxes: true, ids: true, skeleton: true, landmarks: false, fightersOnly: false };

function StageChip({ s }: { s: Stage }) {
  const icon = s.status === "completed" || s.status === "reused" ? <CheckCircle2 className="size-4 text-emerald-400" />
    : s.status === "running" ? <Loader2 className="size-4 animate-spin text-sky-300" />
      : s.status === "failed" ? <XCircle className="size-4 text-blood-400" />
        : s.status === "skipped" ? <CircleSlash className="size-4 text-amber-300" /> : <CircleDashed className="size-4 text-ink-500" />;
  return (
    <div className="flex min-w-0 flex-1 items-start gap-2 rounded-md border border-ink-700 bg-ink-800/60 px-3 py-2" title={s.message}>
      {icon}
      <div className="min-w-0">
        <div className="truncate text-xs text-ink-100">{s.label}</div>
        <div className="truncate text-[11px] text-ink-500">{s.status === "running" && s.job_progress != null ? pct(s.job_progress) : s.status}{s.message && s.status !== "running" ? ` — ${s.message}` : ""}</div>
      </div>
    </div>
  );
}

export default function VideoIntelligencePage() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const toast = useToast();
  const categories = useTaxonomy();
  const playerRef = useRef<PlayerHandle>(null);
  const [frame, setFrame] = useState(0);
  const [source, setSource] = useState<"original" | "annotated">("original");
  const [tracking, setTracking] = useState<Tracking>();
  const [fighter, setFighter] = useState("");
  const [predAt, setPredAt] = useState<number | null>(null);
  const runs = useData(() => platformApi.pipelineRuns(id), [id], (d) => (d?.[0] && (d[0].status === "processing" || d[0].status === "queued") ? 1500 : 0));
  const runKey = runs.data?.[0] ? `${runs.data[0].id}-${runs.data[0].status}` : "";
  const ca = useData(() => platformApi.complete(id), [id, runKey]);
  const tl = useData(() => actionsApi.timeline(id), [id, runKey]);
  const vstatus = useData(() => visionApi.status(id), [id, runKey]);
  const pred = useData(() => (fighter ? platformApi.predictions(fighter, { video_id: id, at: predAt ?? undefined }) : Promise.resolve(undefined)), [fighter, id, predAt, runKey]);

  useEffect(() => {
    setTracking(undefined);
    if (vstatus.data?.current) visionApi.tracking(id).then(setTracking).catch(() => undefined);
  }, [id, vstatus.data?.current?.id]);
  useEffect(() => {
    if (!fighter && ca.data?.fighters.length) setFighter(ca.data.fighters[0]);
  }, [ca.data, fighter]);

  const fps = ca.data?.fps ?? ca.data?.video.fps ?? 30;
  const time = frame / fps;
  const seek = useCallback((t: number) => playerRef.current?.seekToFrame(Math.round(t * fps)), [fps]);
  const t0 = params.get("t");
  useEffect(() => {
    if (t0 && tracking) window.setTimeout(() => seek(Number(t0)), 400);
  }, [t0, tracking, seek]);

  const run = async (force = false) => {
    try {
      const r = await platformApi.runPipeline(id, force);
      toast("info", r.created ? "Full analysis started" : "Analysis is already running");
      void runs.reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Request failed");
    }
  };
  const names = useMemo(() => {
    const m: Record<string, string> = {};
    ca.data?.tracks.forEach((t) => { if (t.fighter_id && t.fighter) m[t.fighter_id] = t.fighter; });
    ca.data?.actions.forEach((a) => { if (a.fighter_id && a.fighter) m[a.fighter_id] = a.fighter; });
    return m;
  }, [ca.data]);

  if (ca.error) return <ErrorBlock message={ca.error} onRetry={() => ca.reload()} />;
  if (!ca.data) return <LoadingBlock />;
  const d = ca.data;
  const cur = runs.data?.[0];
  const running = cur && (cur.status === "processing" || cur.status === "queued");
  const combos = d.combinations.filter((c) => !fighter || c.fighter_key === fighter);
  const duration = d.video.duration_seconds ?? 1;
  const storedNow = [...d.predictions].filter((p) => (!fighter || p.fighter_key === fighter) && p.at_time <= time + 0.01).pop();

  return (
    <>
      <PageHeader title={d.video.title} subtitle="Video intelligence — footage, tracking & pose, recognised actions, combinations and next-move predictions in one place"
        actions={
          <div className="flex gap-2">
            <Link to={`/videos/${id}`} className="btn-secondary">Video details</Link>
            <button className="btn-primary" disabled={!!running} onClick={() => run(false)}>{running ? <Spinner className="size-4" /> : <Play className="size-4" />}Run full analysis</button>
            <button className="btn-ghost" disabled={!!running} onClick={() => run(true)} title="Re-run every stage even if results exist"><RotateCw className="size-4" />Force re-run</button>
          </div>
        } />
      <div className="mb-4 space-y-2">
        <div className="flex flex-wrap gap-2">
          {(cur?.stages ?? [
            { name: "processing", label: "Video processing", status: d.stages.processing === "completed" ? "completed" : d.stages.processing },
            { name: "vision", label: "Detection, tracking & pose", status: d.stages.vision ?? "pending" },
            { name: "actions", label: "Action recognition", status: d.stages.actions ?? "pending" },
            { name: "patterns", label: "Pattern analysis", status: "pending" },
            { name: "prediction", label: "Next-move prediction", status: d.stages.predictions ?? "pending" },
          ]).map((s) => <StageChip key={s.name} s={s as Stage} />)}
        </div>
        {running && <ProgressBar value={cur.progress} tone="sky" />}
        {cur?.status === "failed" && <div className="text-xs text-blood-300">{cur.error_message}</div>}
      </div>

      {!d.stages.vision ? (
        <div className="panel"><EmptyState title="Not analysed yet" body="Run the full analysis: detection, tracking and pose, then actions, patterns and predictions." action={<button className="btn-primary" onClick={() => run(false)}><Play className="size-4" />Run full analysis</button>} /></div>
      ) : (
        <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_440px]">
          <div className="min-w-0 space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex rounded-md border border-ink-600 bg-ink-900 p-0.5 text-xs">
                {(["original", "annotated"] as const).map((s) => (
                  <button key={s} disabled={s === "annotated" && !d.annotated_video_url} onClick={() => setSource(s)} className={cn("rounded px-3 py-1.5 disabled:opacity-40", source === s ? "bg-ink-700 text-white" : "text-ink-400 hover:text-ink-100")}>
                    {s === "original" ? "Original video" : "Annotated video"}
                  </button>
                ))}
              </div>
              <label className="ml-auto flex items-center gap-2 text-xs"><span className="label-caps">Fighter</span>
                <select className="input h-8 w-56 py-0 text-xs" value={fighter} onChange={(e) => { setFighter(e.target.value); setPredAt(null); }}>
                  <option value="">All</option>
                  {d.fighters.map((f) => <option key={f} value={f}>{names[f] ?? f}</option>)}
                </select>
              </label>
            </div>
            {tracking && vstatus.data?.current ? (
              <OverlayPlayer ref={playerRef} videoId={id} originalUrl={d.video.stream_url} annotatedUrl={d.annotated_video_url} source={source} fps={fps}
                stride={tracking.stride} totalFrames={vstatus.data.current.stats?.frames_total ?? d.video.frame_count ?? 0} tracks={tracking.tracks}
                toggles={TOGGLES} kpThreshold={0.35} onFrame={setFrame} />
            ) : <LoadingBlock />}
            <p className="text-[11px] text-ink-500">Overlay: detection boxes, track IDs and pose skeletons from the tracking analysis. Fighter tracks: {d.tracks.filter((t) => t.fighter).map((t) => `#${t.track_id} ${t.fighter} (${pct(t.coverage)} coverage, ${t.interruptions} interruptions)`).join(" · ") || "none assigned"} · {d.tracks.filter((t) => !t.fighter).length} other tracks — <Link className="underline" to={`/tracking/${id}`}>assign in Computer Vision Analysis</Link></p>

            <section className="panel">
              <div className="panel-header"><h3 className="label-caps">Detected action timeline — click to jump</h3><span className="text-xs text-ink-500">{d.actions.length} actions · model {d.stages.action_model ?? "—"}</span></div>
              <div className="p-3"><ActionTimeline lanes={tl.data?.lanes ?? []} duration={tl.data?.duration ?? duration} time={time} categories={categories} onSeek={seek} /></div>
            </section>

            <section className="panel">
              <div className="panel-header"><h3 className="label-caps">Fighting combination timeline</h3><span className="text-xs text-ink-500">recurring combinations of each fighter, where they occur in this video</span></div>
              {combos.length ? (
                <div className="p-4">
                  <div className="relative h-auto space-y-1">
                    {combos.map((c, i) => (
                      <button key={i} onClick={() => seek(c.start)} className="relative block h-6 w-full rounded bg-ink-800 text-left hover:bg-ink-750" title={`${c.labels.join(" → ")} · seen ${c.occurrences_total}× in ${c.fights} fight(s)`}>
                        <span className="absolute inset-y-0 rounded bg-emerald-500/40 ring-1 ring-emerald-400/60" style={{ left: `${(c.start / duration) * 100}%`, width: `${Math.max(0.6, ((c.end - c.start) / duration) * 100)}%` }} />
                        <span className="relative z-10 px-2 text-[11px] leading-6 text-ink-100">{c.start.toFixed(2)} s · {c.labels.join(" → ")} <span className="text-ink-400">(×{c.occurrences_total}, {c.fighter})</span></span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : <p className="px-5 py-6 text-sm text-ink-500">No recurring combination occurs in this video.</p>}
            </section>

            <section className="panel">
              <div className="panel-header"><h3 className="label-caps">Review & correct recognised actions</h3><span className="text-xs text-ink-500">corrections are saved as verified training data</span></div>
              {tl.data?.lanes.length ? <ActionHistory lanes={tl.data.lanes} categories={categories} onSeek={seek} onChanged={() => { void tl.reload(true); void ca.reload(true); }} />
                : <p className="px-5 py-6 text-sm text-ink-500">No recognised actions in this video{d.stages.actions ? "" : " — action recognition has not run (it needs a trained action model)"}.</p>}
            </section>
          </div>

          <aside className="space-y-4">
            <section className="panel border-l-2 border-l-blood-500">
              <div className="panel-header">
                <h3 className="label-caps">Next-move prediction</h3>
                <button className="btn-secondary h-8 px-2 text-xs" disabled={!fighter} onClick={() => setPredAt(Math.round(time * 100) / 100)}>Predict at {time.toFixed(2)} s</button>
              </div>
              {!fighter ? <p className="px-5 py-6 text-sm text-ink-500">Select a fighter.</p> : pred.error ? <ErrorBlock message={pred.error} /> : !pred.data ? <LoadingBlock /> : (
                <>
                  <p className="px-5 pt-3 text-[11px] text-ink-500">{predAt == null ? "Using the fighter's latest actions in this video." : `Using only actions completed by ${predAt.toFixed(2)} s.`}</p>
                  <PredictionCard data={pred.data} onSeek={seek} compact />
                </>
              )}
            </section>
            <section className="panel">
              <div className="panel-header"><h3 className="label-caps">Stored predictions at playhead</h3><span className="text-xs text-ink-500">{d.predictions.length} stored</span></div>
              {storedNow ? (
                <div className="space-y-2 px-5 py-4 text-sm">
                  <div className="text-xs text-ink-400">Made at {storedNow.at_time.toFixed(2)} s, after {storedNow.sequence.map((s) => s.label ?? s.category).slice(-3).join(" → ")}</div>
                  {storedNow.top3.map((t, i) => <div key={t.action} className="flex justify-between rounded border border-dashed border-blood-400/60 px-2 py-1"><span>{i + 1}. {t.label}</span><span className="tabular text-ink-300">{pct(t.probability)}</span></div>)}
                  <div className="text-xs">Actual next: {storedNow.evaluation?.status === "evaluated"
                    ? <span className={storedNow.evaluation.top1 ? "text-emerald-300" : storedNow.evaluation.top3 ? "text-sky-300" : "text-blood-300"}>{storedNow.evaluation.actual_label} {storedNow.evaluation.top1 ? "✓ top-1" : storedNow.evaluation.top3 ? "✓ top-3" : "✗"}</span>
                    : <span className="text-ink-500">no follow-up in this sequence</span>}{storedNow.in_training_data && <span className="ml-1 text-ink-500">(in-sample)</span>}</div>
                </div>
              ) : <p className="px-5 py-6 text-sm text-ink-500">{d.predictions.length ? "No stored prediction before the playhead." : "Run the full analysis with an active prediction model to store predictions through the video."}</p>}
            </section>
          </aside>
        </div>
      )}
    </>
  );
}
