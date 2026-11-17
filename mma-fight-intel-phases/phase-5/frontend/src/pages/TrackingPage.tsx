import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Download, Film, ScanEye } from "lucide-react";
import { ApiError, api } from "../api/client";
import { visionApi, type Tracking } from "../api/vision";
import { ConfirmDialog, EmptyState, ErrorBlock, LoadingBlock, PageHeader } from "../components/ui";
import OverlayPlayer, { type OverlayToggles, type PlayerHandle } from "../components/vision/OverlayPlayer";
import { AnalysisCard, FrameInspector, JobHistoryList, MovementChart, TrackPanel, TrackTimeline } from "../components/vision/panels";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const TOGGLES: { key: keyof OverlayToggles; label: string }[] = [
  { key: "boxes", label: "Boxes" },
  { key: "ids", label: "Track IDs" },
  { key: "skeleton", label: "Skeleton" },
  { key: "landmarks", label: "Landmarks" },
  { key: "fightersOnly", label: "Fighters only" },
];

function loadToggles(): OverlayToggles {
  const d = { boxes: true, ids: true, skeleton: true, landmarks: true, fightersOnly: false };
  try {
    return { ...d, ...JSON.parse(localStorage.getItem("overlay-toggles") ?? "{}") };
  } catch {
    return d;
  }
}

export default function TrackingPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const playerRef = useRef<PlayerHandle>(null);
  const [frame, setFrame] = useState(0);
  const [source, setSource] = useState<"original" | "annotated">("original");
  const [toggles, setToggles] = useState<OverlayToggles>(loadToggles);
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [tracking, setTracking] = useState<Tracking>();

  const { data: videos } = useData(() => api.listVideos({ status: "completed", limit: 500, sort: "uploaded_at", order: "desc" }), []);
  const { data: fighters } = useData(() => api.listFighters(), []);
  const { data: video } = useData(() => (id ? api.getVideo(id) : Promise.resolve(undefined)), [id]);
  const { data: status, error, loading, reload } = useData(
    () => (id ? visionApi.status(id) : Promise.resolve(undefined)),
    [id],
    (d) => (d?.active_job ? 1500 : undefined),
  );
  const currentJobId = status?.current?.id;

  useEffect(() => {
    setTracking(undefined);
    if (!id || !currentJobId) return;
    visionApi.tracking(id).then(setTracking).catch(() => setTracking(undefined));
  }, [id, currentJobId]);

  useEffect(() => {
    try {
      localStorage.setItem("overlay-toggles", JSON.stringify(toggles));
    } catch {
      /* ignore */
    }
  }, [toggles]);

  // Pick the first video automatically
  useEffect(() => {
    if (!id && videos?.items.length) navigate(`/tracking/${videos.items[0].id}`, { replace: true });
  }, [id, videos, navigate]);

  const onFrame = useCallback((f: number) => setFrame(f), []);
  const seek = useCallback((f: number) => playerRef.current?.seekToFrame(f), []);

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    try {
      await fn();
      toast("info", ok);
      await reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  const selector = (
    <select className="input w-auto max-w-[320px]" value={id ?? ""} onChange={(e) => navigate(`/tracking/${e.target.value}`)} aria-label="Video">
      {!id && <option value="">Select a video…</option>}
      {videos?.items.map((v) => <option key={v.id} value={v.id}>{v.title}</option>)}
    </select>
  );

  if (videos && videos.items.length === 0)
    return (
      <>
        <PageHeader title="Fighter Tracking" />
        <div className="panel"><EmptyState icon={<Film className="size-5" />} title="No processed videos yet" body="Upload footage and let Phase 1 processing finish, then analyse it here." action={<Link to="/upload" className="btn-primary">Upload footage</Link>} /></div>
      </>
    );

  const st = status?.current?.stats;
  const fps = st?.fps ?? video?.fps ?? 30;
  const totalFrames = st?.frames_total ?? video?.frame_count ?? 0;

  return (
    <>
      <PageHeader
        title="Fighter Tracking"
        subtitle="Detection, tracking and pose estimation — inspect frame by frame and link tracks to fighter profiles."
        actions={selector}
      />
      {!id || (loading && !status) ? (
        <LoadingBlock />
      ) : error && !status ? (
        <ErrorBlock message={error} onRetry={() => reload()} />
      ) : status && video ? (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="min-w-0 space-y-4">
            {status.current && tracking ? (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex rounded-md border border-ink-600 bg-ink-900 p-0.5 text-xs">
                    {(["original", "annotated"] as const).map((s) => (
                      <button key={s} disabled={s === "annotated" && !status.annotated_video_url} onClick={() => setSource(s)}
                        className={cn("rounded px-3 py-1.5 disabled:opacity-40", source === s ? "bg-ink-700 text-white" : "text-ink-400 hover:text-ink-100")}>
                        {s === "original" ? "Original + live overlay" : "Annotated video"}
                      </button>
                    ))}
                  </div>
                  {source === "original" && (
                    <div className="flex flex-wrap gap-1">
                      {TOGGLES.map((t) => (
                        <label key={t.key} className={cn("flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors", toggles[t.key] ? "border-ink-400 bg-ink-700 text-white" : "border-ink-600 text-ink-400")}>
                          <input type="checkbox" className="sr-only" checked={toggles[t.key]} onChange={(e) => setToggles({ ...toggles, [t.key]: e.target.checked })} />
                          {t.label}
                        </label>
                      ))}
                    </div>
                  )}
                  <div className="ml-auto flex gap-1">
                    {status.annotated_video_url && (
                      <a className="btn-ghost px-2 py-1 text-xs" href={`${status.annotated_video_url}&download=true`}><Download className="size-3.5" />Annotated MP4</a>
                    )}
                    <details className="relative">
                      <summary className="btn-ghost cursor-pointer list-none px-2 py-1 text-xs"><Download className="size-3.5" />Data</summary>
                      <div className="absolute right-0 z-20 mt-1 w-48 rounded-md border border-ink-600 bg-ink-800 p-1 shadow-xl">
                        {["detections", "pose", "movement", "distances", "tracks", "npz"].map((k) => (
                          <a key={k} href={visionApi.exportUrl(video.id, k)} className="block rounded px-3 py-1.5 text-xs text-ink-200 hover:bg-ink-700">
                            {k === "npz" ? "NumPy bundle (.npz)" : `${k[0].toUpperCase()}${k.slice(1)} (.csv)`}
                          </a>
                        ))}
                      </div>
                    </details>
                  </div>
                </div>
                <OverlayPlayer
                  ref={playerRef}
                  videoId={video.id}
                  originalUrl={video.stream_url}
                  annotatedUrl={status.annotated_video_url}
                  source={source}
                  fps={fps}
                  stride={tracking.stride}
                  totalFrames={totalFrames}
                  tracks={tracking.tracks}
                  toggles={toggles}
                  kpThreshold={Number(status.current.settings_snapshot?.keypoint_confidence ?? 0.35)}
                  onFrame={onFrame}
                />
                <TrackTimeline tracking={tracking} totalFrames={totalFrames} frame={frame} onSeek={seek} />
                <MovementChart videoId={video.id} tracking={tracking} frame={frame} onSeek={seek} jobId={tracking.job_id} />
              </>
            ) : (
              <div className="panel">
                <EmptyState
                  icon={<ScanEye className="size-5" />}
                  title={status.active_job ? "Analysis in progress" : "This video hasn't been analysed yet"}
                  body={status.active_job ? "Results appear here automatically when the analysis completes." : "Run computer-vision analysis to detect and track fighters and extract their pose."}
                  action={!status.active_job && status.can_analyze ? (
                    <button className="btn-primary" disabled={busy} onClick={() => act(() => visionApi.analyze(video.id), "Analysis queued")}><ScanEye className="size-4" />Analyse video</button>
                  ) : undefined}
                />
              </div>
            )}
          </div>

          <aside className="space-y-4">
            <AnalysisCard
              status={status}
              busy={busy}
              onAnalyze={() => act(() => visionApi.analyze(video.id), "Analysis queued")}
              onDelete={() => setConfirmDelete(true)}
              onRetry={(jid) => act(() => visionApi.retry(jid), "Retry queued")}
            />
            {status.current && tracking && (
              <>
                <TrackPanel videoId={video.id} tracking={tracking} fighters={fighters ?? []} onSaved={setTracking} onSeek={seek} />
                <FrameInspector videoId={video.id} frame={frame} stride={tracking.stride} jobId={tracking.job_id} />
              </>
            )}
            <JobHistoryList status={status} onRetry={(jid) => act(() => visionApi.retry(jid), "Retry queued")} />
          </aside>
        </div>
      ) : null}
      <ConfirmDialog
        open={confirmDelete}
        title="Delete analysis"
        busy={busy}
        onClose={() => setConfirmDelete(false)}
        onConfirm={async () => {
          await act(() => visionApi.deleteAnalysis(id!), "Analysis deleted");
          setConfirmDelete(false);
        }}
        body="Delete all detections, tracks, pose landmarks, movement data and the annotated video for this video? The video itself is kept."
      />
    </>
  );
}
