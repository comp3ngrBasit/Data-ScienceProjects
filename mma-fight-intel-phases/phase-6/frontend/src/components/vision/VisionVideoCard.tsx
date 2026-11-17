import { Link } from "react-router-dom";
import { ScanEye } from "lucide-react";
import { ApiError } from "../../api/client";
import { STAGE_LABELS, visionApi } from "../../api/vision";
import { useToast } from "../../context/ToastContext";
import { useData } from "../../hooks/useData";
import { ProgressBar, Spinner } from "../ui";

/** Compact Phase 2 status card for the video details page. */
export default function VisionVideoCard({ videoId, videoStatus }: { videoId: string; videoStatus: string }) {
  const toast = useToast();
  const { data: st, reload } = useData(() => visionApi.status(videoId), [videoId, videoStatus], (d) => (d?.active_job ? 1500 : undefined));
  if (!st) return null;
  const a = st.active_job;
  const s = st.current?.stats;
  const analyze = async () => {
    try {
      await visionApi.analyze(videoId);
      toast("info", "Computer-vision analysis queued");
      void reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Could not start analysis");
    }
  };
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps flex items-center gap-2"><ScanEye className="size-3.5" /> Computer vision</h2>
        <Link to={`/tracking/${videoId}`} className="text-xs text-ink-400 hover:text-ink-100">Open tracking →</Link>
      </div>
      <div className="space-y-3 px-5 py-4 text-sm">
        {a ? (
          <div>
            <div className="flex justify-between text-xs text-sky-200"><span className="flex items-center gap-2"><Spinner className="size-3 text-sky-300" />{a.status === "queued" ? "Queued" : STAGE_LABELS[a.current_stage ?? ""] ?? "Analysing"}</span><span className="tabular">{Math.round(a.progress * 100)}%</span></div>
            <ProgressBar value={a.progress} tone="sky" className="mt-2" />
          </div>
        ) : s ? (
          <div className="grid grid-cols-3 gap-2 text-center">
            {[["Tracks", s.tracks_non_fragment], ["Frames", s.frames_analysed], ["Det. conf", s.avg_detection_confidence != null ? `${Math.round(s.avg_detection_confidence * 100)}%` : "—"]].map(([k, v]) => (
              <div key={k as string} className="rounded-md border border-ink-700 bg-ink-900 py-2">
                <div className="font-display text-xl font-semibold text-white tabular">{v}</div>
                <div className="text-[10px] uppercase tracking-wider text-ink-500">{k}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-ink-400">{st.can_analyze ? "Detect and track fighters and extract body landmarks." : st.reason}</p>
        )}
        {!a && (
          <button className="btn-secondary w-full" disabled={!st.can_analyze} onClick={analyze}>
            <ScanEye className="size-4" />{st.current ? "Re-analyse" : "Analyse fighters & pose"}
          </button>
        )}
      </div>
    </section>
  );
}
