import { Fragment, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Activity, ChevronDown, CircleAlert, RotateCw, TriangleAlert } from "lucide-react";
import { ApiError, api } from "../api/client";
import type { JobStatus } from "../api/types";
import { EmptyState, ErrorBlock, LoadingBlock, PageHeader, ProgressBar, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn, formatDate, timeAgo } from "../lib/format";

const FILTERS: (JobStatus | "")[] = ["", "queued", "processing", "completed", "failed", "cancelled"];

export default function JobsPage() {
  const [params, setParams] = useSearchParams();
  const status = (params.get("status") as JobStatus) || "";
  const toast = useToast();
  const [open, setOpen] = useState<string | null>(null);
  const [retrying, setRetrying] = useState<string | null>(null);
  const { data, error, loading, reload } = useData(() => api.listJobs({ status, limit: 200 }), [status], (d) =>
    d && (d.counts.queued > 0 || d.counts.processing > 0) ? 1500 : 8000,
  );

  const retry = async (jobId: string) => {
    setRetrying(jobId);
    try {
      await api.retryJob(jobId);
      toast("info", "Retry queued");
      void reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Retry failed");
    } finally {
      setRetrying(null);
    }
  };

  const counts = data?.counts;
  const failedRetryable = data?.items.filter((j) => j.status === "failed") ?? [];

  return (
    <>
      <PageHeader
        title="Processing Jobs"
        subtitle="Background metadata extraction, thumbnail generation and frame sampling."
        actions={
          status === "failed" && failedRetryable.length > 1 ? (
            <button
              className="btn-secondary"
              onClick={async () => {
                // retry the latest failed job per video
                const seen = new Set<string>();
                for (const j of failedRetryable) {
                  if (seen.has(j.video_id)) continue;
                  seen.add(j.video_id);
                  await api.retryJob(j.id).catch(() => undefined);
                }
                toast("info", `Queued ${seen.size} retr${seen.size === 1 ? "y" : "ies"}`);
                void reload(true);
              }}
            >
              <RotateCw className="size-4" />Retry all failed
            </button>
          ) : undefined
        }
      />

      <div className="mb-4 flex flex-wrap gap-1.5">
        {FILTERS.map((f) => {
          const n = f ? counts?.[f] : counts ? Object.values(counts).reduce((a, b) => a + b, 0) : undefined;
          return (
            <button
              key={f || "all"}
              onClick={() => { const p = new URLSearchParams(params); if (f) p.set("status", f); else p.delete("status"); setParams(p, { replace: true }); }}
              className={cn("rounded-full border px-3 py-1 text-xs transition-colors", status === f ? "border-blood-500 bg-blood-900/30 text-white" : "border-ink-600 text-ink-300 hover:border-ink-500 hover:text-white")}
            >
              {f ? f[0].toUpperCase() + f.slice(1) : "All"}
              {n != null && <span className="ml-1.5 text-ink-400 tabular">{n}</span>}
            </button>
          );
        })}
      </div>

      {loading && !data ? (
        <LoadingBlock />
      ) : error && !data ? (
        <ErrorBlock message={error} onRetry={() => reload()} />
      ) : data && data.items.length === 0 ? (
        <div className="panel"><EmptyState icon={<Activity className="size-5" />} title={status ? `No ${status} jobs` : "No processing jobs yet"} body="Jobs are created when videos are uploaded (with auto-processing on) or when you start processing manually." /></div>
      ) : data ? (
        <div className="panel overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead>
              <tr className="border-b border-ink-700 text-left">
                {["Video", "Status", "Progress", "Attempt", "Created", "Duration", ""].map((h) => <th key={h} className="label-caps px-4 py-3 font-semibold">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.items.map((j) => (
                <Fragment key={j.id}>
                  <tr className={cn("border-b border-ink-700/50 hover:bg-ink-800/50", open === j.id && "bg-ink-800/50")}>
                    <td className="max-w-[300px] px-4 py-3">
                      <Link to={`/videos/${j.video_id}`} className="block truncate text-ink-100 hover:text-white hover:underline">{j.video_title ?? j.video_id}</Link>
                      {j.error_message && <div className="truncate text-xs text-blood-300" title={j.error_message}>{j.error_message}</div>}
                    </td>
                    <td className="px-4"><StatusBadge status={j.status} /></td>
                    <td className="w-48 px-4">
                      <div className="flex items-center gap-2">
                        <ProgressBar value={j.progress} tone={j.status === "completed" ? "green" : j.status === "failed" ? "red" : "sky"} />
                        <span className="w-9 text-right text-xs text-ink-400 tabular">{Math.round(j.progress * 100)}%</span>
                      </div>
                      {j.status === "processing" && j.current_stage && <div className="mt-1 text-[11px] text-ink-500">{j.stage_log?.find((s) => s.status === "running")?.label ?? j.current_stage}</div>}
                    </td>
                    <td className="px-4 text-ink-300 tabular">#{j.attempt}</td>
                    <td className="px-4 text-ink-400" title={formatDate(j.created_at)}>{timeAgo(j.created_at)}</td>
                    <td className="px-4 text-ink-300 tabular">{j.duration_seconds != null ? `${j.duration_seconds.toFixed(1)}s` : "—"}</td>
                    <td className="px-4">
                      <div className="flex justify-end gap-1">
                        {(j.status === "failed" || j.status === "cancelled") && (
                          <button className="btn-secondary px-2.5 py-1 text-xs" disabled={retrying === j.id} onClick={() => retry(j.id)}><RotateCw className="size-3.5" />Retry</button>
                        )}
                        <button className="btn-ghost p-1.5" onClick={() => setOpen(open === j.id ? null : j.id)} aria-label="Details">
                          <ChevronDown className={cn("size-4 transition-transform", open === j.id && "rotate-180")} />
                        </button>
                      </div>
                    </td>
                  </tr>
                  {open === j.id && (
                    <tr className="border-b border-ink-700/50 bg-ink-900/60">
                      <td colSpan={7} className="px-4 py-3 text-xs">
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="space-y-1.5">
                            <div className="label-caps">Stages</div>
                            {j.stage_log?.length ? j.stage_log.map((s) => (
                              <div key={s.stage} className="flex justify-between gap-3">
                                <span className="text-ink-300">{s.label}</span>
                                <span className={cn(s.status === "completed" ? "text-emerald-300" : s.status === "failed" ? "text-blood-300" : "text-ink-400")}>
                                  {s.status}{s.seconds != null && <span className="ml-2 text-ink-500 tabular">{s.seconds}s</span>}
                                </span>
                              </div>
                            )) : <span className="text-ink-500">Not started</span>}
                          </div>
                          <div className="space-y-1.5">
                            <div className="label-caps">Details</div>
                            <div className="text-ink-400">Job ID <span className="font-mono text-ink-300">{j.id}</span></div>
                            <div className="text-ink-400">Started {formatDate(j.started_at)} · Finished {formatDate(j.finished_at)}</div>
                            {j.error_message && <p className="flex gap-2 text-blood-300"><CircleAlert className="size-3.5 shrink-0" />{j.error_message}</p>}
                            {j.warnings?.map((w) => <p key={w} className="flex gap-2 text-amber-200"><TriangleAlert className="size-3.5 shrink-0" />{w}</p>)}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </>
  );
}
