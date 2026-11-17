import { Link } from "react-router-dom";
import { RotateCw, ScanEye } from "lucide-react";
import { ApiError } from "../../api/client";
import { STAGE_LABELS, visionApi } from "../../api/vision";
import { useToast } from "../../context/ToastContext";
import { useData } from "../../hooks/useData";
import { timeAgo } from "../../lib/format";
import { EmptyState, ProgressBar, StatusBadge } from "../ui";

export default function VisionJobsTable() {
  const toast = useToast();
  const { data, reload } = useData(() => visionApi.jobs(), [], (d) => (d && (d.counts.queued || d.counts.processing) ? 1500 : 8000));
  const retry = async (id: string) => {
    try {
      await visionApi.retry(id);
      toast("info", "Retry queued");
      void reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Retry failed");
    }
  };
  return (
    <section className="mt-8">
      <h2 className="mb-3 flex items-center gap-2 font-display text-2xl font-bold uppercase tracking-wide text-white"><ScanEye className="size-5 text-blood-400" />Computer vision jobs</h2>
      {!data ? null : data.items.length === 0 ? (
        <div className="panel"><EmptyState icon={<ScanEye className="size-5" />} title="No vision analyses yet" body="Start one from Fighter Tracking or a video's details page." /></div>
      ) : (
        <div className="panel overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead>
              <tr className="border-b border-ink-700 text-left">
                {["Video", "Status", "Progress", "Attempt", "Created", "Duration", ""].map((h) => <th key={h} className="label-caps px-4 py-3 font-semibold">{h}</th>)}
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-700/50">
              {data.items.map((j) => (
                <tr key={j.id} className="hover:bg-ink-800/50">
                  <td className="max-w-[300px] px-4 py-3">
                    <Link to={`/tracking/${j.video_id}`} className="block truncate text-ink-100 hover:underline">{j.video_title}</Link>
                    {j.error_message && <div className="truncate text-xs text-blood-300" title={j.error_message}>{j.error_message}</div>}
                    {j.stats?.superseded_by && <div className="text-[11px] text-ink-500">superseded by a newer analysis</div>}
                  </td>
                  <td className="px-4"><StatusBadge status={j.status} /></td>
                  <td className="w-48 px-4">
                    <div className="flex items-center gap-2">
                      <ProgressBar value={j.progress} tone={j.status === "completed" ? "green" : j.status === "failed" ? "red" : "sky"} />
                      <span className="w-9 text-right text-xs text-ink-400 tabular">{Math.round(j.progress * 100)}%</span>
                    </div>
                    {j.status === "processing" && <div className="mt-1 text-[11px] text-ink-500">{STAGE_LABELS[j.current_stage ?? ""] ?? j.current_stage}</div>}
                  </td>
                  <td className="px-4 text-ink-300 tabular">#{j.attempt}</td>
                  <td className="px-4 text-ink-400">{timeAgo(j.created_at)}</td>
                  <td className="px-4 text-ink-300 tabular">{j.duration_seconds != null ? `${j.duration_seconds.toFixed(0)}s` : "—"}</td>
                  <td className="px-4 text-right">
                    {(j.status === "failed" || j.status === "cancelled") && (
                      <button className="btn-secondary px-2.5 py-1 text-xs" onClick={() => retry(j.id)}><RotateCw className="size-3.5" />Retry</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
