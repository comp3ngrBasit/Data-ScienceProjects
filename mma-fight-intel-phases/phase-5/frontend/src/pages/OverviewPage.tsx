import { Link } from "react-router-dom";
import { Activity, ChevronRight, Clock, Film, HardDrive, Layers, Upload, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { Overview, VideoStatus } from "../api/types";
import { EmptyState, ErrorBlock, LoadingBlock, PageHeader, ProgressBar, StatCard, StatusBadge, Thumb } from "../components/ui";
import { useData } from "../hooks/useData";
import VisionOverviewSection from "../components/vision/VisionOverviewSection";
import ActionOverviewSection from "../components/actions/ActionOverviewSection";
import PatternOverviewSection from "../components/patterns/PatternOverviewSection";
import PredictionOverviewSection from "../components/predictions/PredictionOverviewSection";
import { formatBytes, formatDuration, formatLongDuration, formatResolution, timeAgo } from "../lib/format";

const STATUS_ORDER: VideoStatus[] = ["completed", "processing", "queued", "uploaded", "failed"];
const STATUS_BAR: Record<VideoStatus, string> = {
  completed: "bg-emerald-400",
  processing: "bg-sky-400",
  queued: "bg-amber-400",
  uploaded: "bg-ink-300",
  failed: "bg-blood-500",
};

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  const d = new Date(`${label}T00:00:00Z`);
  return (
    <div className="rounded-md border border-ink-600 bg-ink-800 px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-400">{d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" })}</div>
      <div className="mt-0.5 font-medium text-white tabular">
        {payload[0].value} upload{payload[0].value === 1 ? "" : "s"}
      </div>
    </div>
  );
}

function StatusBreakdown({ o }: { o: Overview }) {
  const total = o.total_videos || 1;
  return (
    <div className="space-y-3">
      <div className="flex h-2 w-full gap-[2px] overflow-hidden rounded-full bg-ink-700">
        {STATUS_ORDER.filter((s) => o.status_counts[s]).map((s) => (
          <div key={s} className={STATUS_BAR[s]} style={{ width: `${(o.status_counts[s] / total) * 100}%` }} title={`${s}: ${o.status_counts[s]}`} />
        ))}
      </div>
      <ul className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
        {STATUS_ORDER.map((s) => (
          <li key={s} className="flex items-center justify-between gap-2 text-sm">
            <StatusBadge status={s} />
            <span className="text-ink-200 tabular">{o.status_counts[s]}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function OverviewPage() {
  const { data: o, error, loading, reload } = useData(api.overview, [], (d) => (d && d.active_jobs > 0 ? 2500 : 15000));

  const header = (
    <PageHeader
      title="Overview"
      subtitle="Fight footage library and processing status"
      actions={
        <Link to="/upload" className="btn-primary px-5 py-2.5 text-[15px] shadow-lg shadow-blood-900/50">
          <Upload className="size-4" /> Upload Fight Footage
        </Link>
      }
    />
  );

  if (loading && !o) return <>{header}<LoadingBlock /></>;
  if (error && !o) return <>{header}<ErrorBlock message={error} onRetry={() => reload()} /></>;
  if (!o) return null;

  const storageTotal = Object.values(o.storage_bytes).reduce((a, b) => a + b, 0);
  const hasUploadsInRange = o.uploads_by_day.some((d) => d.count > 0);

  return (
    <>
      {header}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Total videos" value={o.total_videos} icon={<Film className="size-4" />} hint={`${formatLongDuration(o.total_duration_seconds)} of footage`} accent />
        <StatCard label="Fighters" value={o.total_fighters} icon={<Users className="size-4" />} hint={<Link className="hover:text-ink-200" to="/fighters">Manage profiles →</Link>} />
        <StatCard label="Processed" value={o.videos_processed} icon={<Layers className="size-4" />} hint={`${o.total_sampled_frames.toLocaleString()} frames sampled`} />
        <StatCard
          label="Awaiting processing"
          value={o.videos_awaiting}
          icon={<Clock className="size-4" />}
          hint={o.videos_failed ? <Link to="/jobs?status=failed" className="text-blood-300 hover:text-blood-400">{o.videos_failed} failed — review</Link> : o.active_jobs ? `${o.active_jobs} job(s) active` : "Queue is clear"}
        />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <section className="panel lg:col-span-2">
          <div className="panel-header">
            <h2 className="label-caps">Uploads · last 14 days</h2>
            <span className="text-xs text-ink-400 tabular">{o.uploads_by_day.reduce((a, d) => a + d.count, 0)} total</span>
          </div>
          <div className="h-56 px-3 pb-3 pt-4">
            {hasUploadsInRange ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={o.uploads_by_day} margin={{ top: 4, right: 8, left: -24, bottom: 0 }} barCategoryGap="30%">
                  <CartesianGrid vertical={false} stroke="#26262d" />
                  <XAxis
                    dataKey="date"
                    tickLine={false}
                    axisLine={{ stroke: "#33333c" }}
                    tick={{ fill: "#75757f", fontSize: 11 }}
                    tickFormatter={(d: string) => new Date(`${d}T00:00:00Z`).toLocaleDateString(undefined, { day: "numeric", month: "short", timeZone: "UTC" })}
                    interval="preserveStartEnd"
                    minTickGap={16}
                  />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: "#75757f", fontSize: 11 }} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar dataKey="count" fill="#e11d2e" radius={[4, 4, 0, 0]} maxBarSize={28} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="grid h-full place-items-center text-sm text-ink-500">No uploads in the last 14 days</div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2 className="label-caps">Library status</h2>
            <Link to="/videos" className="text-xs text-ink-400 hover:text-ink-100">View library</Link>
          </div>
          <div className="space-y-5 px-5 py-4">
            <StatusBreakdown o={o} />
            <div className="border-t border-ink-700/60 pt-4">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2 text-ink-300"><HardDrive className="size-4 text-ink-500" /> Storage used</span>
                <span className="text-ink-100 tabular">{formatBytes(storageTotal)}</span>
              </div>
              <dl className="mt-2 space-y-1 text-xs text-ink-400">
                {Object.entries(o.storage_bytes).map(([k, v]) => (
                  <div key={k} className="flex justify-between"><dt className="capitalize">{k}</dt><dd className="tabular">{formatBytes(v)}</dd></div>
                ))}
              </dl>
            </div>
            {o.resolution_breakdown.length > 0 && (
              <div className="border-t border-ink-700/60 pt-4">
                <div className="label-caps mb-2">Resolution</div>
                <div className="flex flex-wrap gap-1.5">
                  {o.resolution_breakdown.map((r) => (
                    <span key={r.label} className="rounded border border-ink-600 bg-ink-800 px-2 py-0.5 text-xs text-ink-200">
                      {r.label} <span className="text-ink-400 tabular">· {r.count}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <section className="panel">
          <div className="panel-header">
            <h2 className="label-caps">Recent uploads</h2>
            <Link to="/videos" className="flex items-center gap-1 text-xs text-ink-400 hover:text-ink-100">All videos <ChevronRight className="size-3" /></Link>
          </div>
          {o.recent_videos.length === 0 ? (
            <EmptyState title="No fight footage yet" body="Upload MP4, MOV or AVI files to start building the library." action={<Link to="/upload" className="btn-primary"><Upload className="size-4" />Upload footage</Link>} />
          ) : (
            <ul className="divide-y divide-ink-700/60">
              {o.recent_videos.map((v) => (
                <li key={v.id}>
                  <Link to={`/videos/${v.id}`} className="flex items-center gap-3 px-5 py-3 hover:bg-ink-800/60">
                    <Thumb url={v.thumbnail_url} alt={v.title} status={v.status} className="h-11 w-20 shrink-0 rounded" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-ink-100">{v.title}</div>
                      <div className="mt-0.5 text-xs text-ink-400 tabular">
                        {formatDuration(v.duration_seconds)} · {formatResolution(v.width, v.height)} · {formatBytes(v.file_size)}
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <StatusBadge status={v.status} />
                      <span className="text-[11px] text-ink-500">{timeAgo(v.uploaded_at)}</span>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2 className="label-caps">Recent processing activity</h2>
            <Link to="/jobs" className="flex items-center gap-1 text-xs text-ink-400 hover:text-ink-100">All jobs <ChevronRight className="size-3" /></Link>
          </div>
          {o.recent_jobs.length === 0 ? (
            <EmptyState icon={<Activity className="size-5" />} title="No processing jobs yet" body="Jobs appear here when videos are queued for processing." />
          ) : (
            <ul className="divide-y divide-ink-700/60">
              {o.recent_jobs.map((j) => (
                <li key={j.id}>
                  <Link to={`/videos/${j.video_id}`} className="block px-5 py-3 hover:bg-ink-800/60">
                    <div className="flex items-center justify-between gap-3">
                      <span className="truncate text-sm text-ink-100">{j.video_title ?? "Deleted video"}</span>
                      <StatusBadge status={j.status} />
                    </div>
                    {j.status === "processing" ? (
                      <div className="mt-2 flex items-center gap-3">
                        <ProgressBar value={j.progress} tone="sky" />
                        <span className="w-10 text-right text-xs text-ink-400 tabular">{Math.round(j.progress * 100)}%</span>
                      </div>
                    ) : (
                      <div className="mt-1 truncate text-xs text-ink-500">
                        Attempt {j.attempt} · {timeAgo(j.finished_at ?? j.started_at ?? j.created_at)}
                        {j.duration_seconds != null && ` · ${j.duration_seconds.toFixed(1)}s`}
                        {j.error_message && <span className="text-blood-300"> · {j.error_message}</span>}
                      </div>
                    )}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
      <VisionOverviewSection />
      <ActionOverviewSection />
      <PatternOverviewSection />
      <PredictionOverviewSection />
    </>
  );
}
