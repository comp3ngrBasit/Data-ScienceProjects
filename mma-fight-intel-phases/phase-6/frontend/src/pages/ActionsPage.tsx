import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Brain, CircleAlert, Film, RotateCw, Swords, Trash2, TriangleAlert } from "lucide-react";
import { ApiError, api } from "../api/client";
import { actionsApi, GROUP_COLORS } from "../api/actions";
import { visionApi, type Tracking } from "../api/vision";
import ActionCharts from "../components/actions/ActionCharts";
import ActionHistory from "../components/actions/ActionHistory";
import ActionTimeline from "../components/actions/ActionTimeline";
import AnnotationPanel from "../components/actions/AnnotationPanel";
import SequenceView from "../components/actions/SequenceView";
import { Marker, pct, useTaxonomy } from "../components/actions/common";
import { ConfirmDialog, EmptyState, KV, LoadingBlock, PageHeader, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import OverlayPlayer, { type OverlayToggles, type PlayerHandle } from "../components/vision/OverlayPlayer";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const TABS = ["Detected actions", "Sequences", "Charts", "Annotate"] as const;
type Tab = (typeof TABS)[number];
const TOGGLES: OverlayToggles = { boxes: true, ids: true, skeleton: true, landmarks: false, fightersOnly: false };

export default function ActionsPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const categories = useTaxonomy();
  const playerRef = useRef<PlayerHandle>(null);
  const [frame, setFrame] = useState(0);
  const [tab, setTab] = useState<Tab>("Detected actions");
  const [source, setSource] = useState<"original" | "annotated">("original");
  const [tracking, setTracking] = useState<Tracking>();
  const [busy, setBusy] = useState(false);
  const [confirmDel, setConfirmDel] = useState(false);
  const [annVersion, setAnnVersion] = useState(0);

  const { data: videos } = useData(() => api.listVideos({ status: "completed", limit: 500 }), []);
  const { data: fighters } = useData(() => api.listFighters(), []);
  const { data: video } = useData(() => (id ? api.getVideo(id) : Promise.resolve(undefined)), [id]);
  const { data: vstatus } = useData(() => (id ? visionApi.status(id) : Promise.resolve(undefined)), [id]);
  const { data: st, reload } = useData(() => (id ? actionsApi.status(id) : Promise.resolve(undefined)), [id], (d) => (d?.active_job ? 1500 : undefined));
  const jobId = st?.current?.id ?? null;
  const { data: tl, reload: reloadTl } = useData(() => (id ? actionsApi.timeline(id) : Promise.resolve(undefined)), [id, jobId, annVersion]);

  useEffect(() => {
    setTracking(undefined);
    if (id && vstatus?.current) visionApi.tracking(id).then(setTracking).catch(() => undefined);
  }, [id, vstatus?.current?.id]);
  useEffect(() => {
    if (!id && videos?.items.length) navigate(`/actions/${videos.items[0].id}`, { replace: true });
  }, [id, videos, navigate]);

  const fps = vstatus?.current?.stats?.fps ?? video?.fps ?? 30;
  const time = frame / fps;
  const onFrame = useCallback((f: number) => setFrame(f), []);
  const seek = useCallback((t: number) => playerRef.current?.seekToFrame(Math.round(t * fps)), [fps]);
  const refresh = () => { void reloadTl(true); void reload(true); };

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    try { await fn(); toast("info", ok); await reload(true); } catch (e) { toast("error", e instanceof ApiError ? e.message : "Request failed"); } finally { setBusy(false); }
  };

  const lanes = tl?.lanes ?? [];
  const current = lanes.flatMap((l) => l.actions.filter((a) => a.start_time <= time + 0.05 && a.end_time >= time - 0.05 && a.review_status !== "rejected").map((a) => ({ ...a, lane: l.name })));
  const fam = Object.fromEntries(categories.map((c) => [c.key, c.family]));
  const job = st?.current;
  const combos = job?.stats?.repeated_combinations ?? {};

  const selector = (
    <div className="flex gap-2">
      <select className="input w-auto max-w-[280px]" value={id ?? ""} onChange={(e) => navigate(`/actions/${e.target.value}`)} aria-label="Video">
        {videos?.items.map((v) => <option key={v.id} value={v.id}>{v.title}</option>)}
      </select>
      <Link to="/actions/models" className="btn-secondary"><Brain className="size-4" />Models & training</Link>
    </div>
  );

  if (videos && !videos.items.length)
    return <><PageHeader title="MMA Action Recognition" /><div className="panel"><EmptyState icon={<Film className="size-5" />} title="No processed videos yet" /></div></>;

  return (
    <>
      <PageHeader title="MMA Action Recognition" subtitle="Temporal recognition of strikes and movements per fighter, with manual annotation for training." actions={selector} />
      {!id || !st || !video ? <LoadingBlock /> : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="min-w-0 space-y-4">
            {vstatus?.current && tracking ? (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex rounded-md border border-ink-600 bg-ink-900 p-0.5 text-xs">
                    {(["original", "annotated"] as const).map((s) => (
                      <button key={s} disabled={s === "annotated" && !vstatus.annotated_video_url} onClick={() => setSource(s)}
                        className={cn("rounded px-3 py-1.5 disabled:opacity-40", source === s ? "bg-ink-700 text-white" : "text-ink-400 hover:text-ink-100")}>
                        {s === "original" ? "Original video" : "Annotated video"}
                      </button>
                    ))}
                  </div>
                  <div className="ml-auto flex min-h-8 flex-wrap items-center gap-1.5" aria-live="polite">
                    {current.length ? current.map((a) => (
                      <span key={a.id} className="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs text-white" style={{ borderColor: GROUP_COLORS[a.group], background: `${GROUP_COLORS[a.group]}26` }}>
                        <Marker family={fam[a.category] ?? "unknown"} group={a.group} size={10} />{a.lane}: {a.label} <span className="text-ink-300 tabular">{pct(a.confidence)}</span>
                      </span>
                    )) : <span className="text-xs text-ink-500">no action at playhead</span>}
                  </div>
                </div>
                <OverlayPlayer ref={playerRef} videoId={video.id} originalUrl={video.stream_url} annotatedUrl={vstatus.annotated_video_url}
                  source={source} fps={fps} stride={tracking.stride} totalFrames={vstatus.current.stats?.frames_total ?? video.frame_count ?? 0}
                  tracks={tracking.tracks} toggles={TOGGLES} kpThreshold={0.35} onFrame={onFrame} />
              </>
            ) : (
              <div className="panel"><EmptyState title="Fighter tracking & pose required" body="Action recognition works on Phase 2 pose sequences. Analyse this video first." action={<Link to={`/tracking/${video.id}`} className="btn-primary">Open Fighter Tracking</Link>} /></div>
            )}

            <ActionTimeline lanes={lanes} duration={tl?.duration ?? video.duration_seconds ?? 1} time={time} categories={categories} onSeek={seek} />

            <section className="panel">
              <div className="flex gap-1 border-b border-ink-700/70 px-3 pt-2">
                {TABS.map((t) => (
                  <button key={t} onClick={() => setTab(t)} className={cn("relative px-3 py-2 text-sm", tab === t ? "text-white" : "text-ink-400 hover:text-ink-100")}>
                    {t}{tab === t && <span className="absolute inset-x-2 -bottom-px h-0.5 bg-blood-500" />}
                  </button>
                ))}
              </div>
              {tab === "Detected actions" && (jobId ? <ActionHistory lanes={lanes} categories={categories} onSeek={seek} onChanged={refresh} /> : <p className="px-5 py-8 text-center text-sm text-ink-500">Run action analysis to see detected actions.</p>)}
              {tab === "Sequences" && <SequenceView videoId={video.id} jobId={jobId} categories={categories} onSeek={seek} />}
              {tab === "Charts" && <ActionCharts lanes={lanes} />}
              {tab === "Annotate" && (
                <AnnotationPanel videoId={video.id} time={time} fighters={fighters ?? []} tracks={tracking?.tracks ?? []} categories={categories}
                  onSeek={seek} onChanged={() => setAnnVersion((v) => v + 1)} version={annVersion} />
              )}
            </section>
          </div>

          <aside className="space-y-4">
            <section className="panel">
              <div className="panel-header">
                <h2 className="label-caps flex items-center gap-2"><Swords className="size-3.5" />Recognition</h2>
                <div className="flex gap-1">
                  {!st.active_job && <button className="btn-primary px-3 py-1.5 text-xs" disabled={!st.can_analyze || busy} title={st.reason ?? ""} onClick={() => act(() => actionsApi.analyze(video.id), "Action analysis queued")}>
                    {busy ? <Spinner className="size-3 text-white" /> : <RotateCw className="size-3.5" />}{job ? "Re-analyse" : "Analyse actions"}</button>}
                  {(job || st.history.length > 0) && <button className="btn-ghost p-1.5 hover:text-blood-400" title="Delete results" onClick={() => setConfirmDel(true)}><Trash2 className="size-4" /></button>}
                </div>
              </div>
              <div className="space-y-3 px-5 py-4 text-sm">
                {st.reason && !st.active_job && <p className="flex gap-2 text-xs text-amber-200"><CircleAlert className="size-3.5 shrink-0" />{st.reason}{st.reason.includes("model") && <Link to="/actions/models" className="underline">Train</Link>}</p>}
                {st.active_job && (
                  <div>
                    <div className="flex justify-between text-xs text-sky-200"><span className="flex items-center gap-2"><Spinner className="size-3 text-sky-300" />{st.active_job.status === "queued" ? "Queued" : st.active_job.current_stage ?? "Working"}</span><span className="tabular">{Math.round(st.active_job.progress * 100)}%</span></div>
                    <ProgressBar value={st.active_job.progress} tone="sky" className="mt-2" />
                  </div>
                )}
                {st.history[0] && st.history[0].status === "failed" && !st.active_job && <p className="text-xs text-blood-300">Last run failed: {st.history[0].error_message}</p>}
                {job?.stats && (
                  <dl>
                    <KV k="Model" v={<span className="font-mono text-[11px]">{job.model_version}</span>} />
                    <KV k="Actions found" v={job.stats.actions} />
                    <KV k="Pose sampling" v={`${job.stats.pose_rate_fps} fps`} />
                    <KV k="Window" v={`${job.stats.window_seconds}s · ${job.stats.windows_classified}/${job.stats.windows_total} windows classified`} />
                    <KV k="Skipped (idle)" v={job.stats.windows_idle ?? 0} />
                    <KV k="Unclassified movements" v={job.stats.windows_unknown ?? 0} />
                    <KV k="Duplicates removed" v={job.stats.duplicates_removed ?? 0} />
                    <KV k="Model knows" v={<span className="text-xs">{(job.stats.model_classes ?? []).map((c) => categories.find((x) => x.key === c)?.label ?? c).join(", ")}</span>} />
                  </dl>
                )}
                {job?.warnings?.map((w) => <p key={w} className="flex gap-2 text-xs text-amber-200"><TriangleAlert className="size-3.5 shrink-0" />{w}</p>)}
                {st.vision && st.vision.pose_fps < 15 && <p className="flex gap-2 text-xs text-amber-200"><TriangleAlert className="size-3.5 shrink-0" />Pose sampled at {st.vision.pose_fps} fps — re-run Fighter Tracking at 25 fps or every frame for strikes.</p>}
                <p className="text-[11px] leading-relaxed text-ink-500">
                  Percentages are the model's own output probability. “Validated” is the precision that class achieved on held-out videos — trust it only when n is large. Classes the model was never trained on cannot be detected.
                </p>
              </div>
            </section>

            {Object.keys(combos).length > 0 && (
              <section className="panel">
                <div className="panel-header"><h2 className="label-caps">Repeated combinations</h2></div>
                <div className="space-y-2 px-5 py-3 text-xs">
                  {Object.entries(combos).map(([k, list]) => list.length ? (
                    <div key={k}>
                      <div className="mb-1 text-ink-400">{lanes.find((l) => l.key === k)?.name ?? k.replace("track:", "Track ")}</div>
                      {list.slice(0, 5).map((c) => <div key={c.keys.join()} className="flex justify-between text-ink-200"><span>{c.combination.join(" → ")}</span><span className="text-ink-400 tabular">×{c.count}</span></div>)}
                    </div>
                  ) : null)}
                </div>
              </section>
            )}

            {st.history.length > 0 && (
              <section className="panel">
                <div className="panel-header"><h2 className="label-caps">Analysis history</h2></div>
                <ul className="divide-y divide-ink-700/50 text-xs">
                  {st.history.map((j) => (
                    <li key={j.id} className="flex items-center gap-2 px-4 py-2">
                      <StatusBadge status={j.status} /><span className="flex-1 truncate font-mono text-[10px] text-ink-500">{j.model_version}</span>
                      <span className="text-ink-500">{j.finished_at ? new Date(j.finished_at).toLocaleString() : ""}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </aside>
        </div>
      )}
      <ConfirmDialog open={confirmDel} title="Delete action results" busy={busy} onClose={() => setConfirmDel(false)}
        onConfirm={async () => { await act(() => actionsApi.deleteResults(id!), "Action results deleted"); setConfirmDel(false); void reloadTl(true); }}
        body="Delete recognised actions and sequences for this video? Manual annotations and corrections are kept." />
    </>
  );
}
