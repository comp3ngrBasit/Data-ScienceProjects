import { Link } from "react-router-dom";
import { ChevronRight, Crosshair, Gauge, Layers, ScanEye, Users } from "lucide-react";
import { apiUrl } from "../../api/client";
import { STAGE_LABELS, visionApi } from "../../api/vision";
import { useData } from "../../hooks/useData";
import { formatDate } from "../../lib/format";
import { EmptyState, ProgressBar, StatCard, StatusBadge } from "../ui";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

export default function VisionOverviewSection() {
  const { data: o } = useData(visionApi.overview, [], (d) => (d && d.active_jobs.length ? 2000 : 15000));
  if (!o) return null;
  const done = o.job_counts.completed ?? 0;
  const failed = o.job_counts.failed ?? 0;
  return (
    <section className="mt-8">
      <div className="mb-3 flex items-end justify-between">
        <div>
          <h2 className="font-display text-2xl font-bold uppercase tracking-wide text-white">Computer Vision Analysis</h2>
          <p className="text-sm text-ink-400">Fighter detection, tracking and pose estimation (Phase 2)</p>
        </div>
        <Link to="/tracking" className="btn-secondary"><ScanEye className="size-4" />Fighter tracking</Link>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Detected fighters"
          value={o.fighter_tracks_assigned || o.fighter_tracks_suggested}
          icon={<Users className="size-4" />}
          hint={o.fighter_tracks_assigned ? `${o.distinct_fighters_identified} profile${o.distinct_fighters_identified === 1 ? "" : "s"} linked` : `suggested tracks · none confirmed yet`}
          accent
        />
        <StatCard label="Frames analysed" value={o.total_frames_analysed.toLocaleString()} icon={<Layers className="size-4" />} hint={`${o.videos_analysed} of ${o.videos_eligible} videos analysed`} />
        <StatCard label="Detection confidence" value={pct(o.avg_detection_confidence)} icon={<Crosshair className="size-4" />} hint={`pose (core joints) ${pct(o.avg_pose_confidence)}`} />
        <StatCard
          label="Tracking consistency"
          value={pct(o.tracking.avg_fighter_track_coverage)}
          icon={<Gauge className="size-4" />}
          hint={`fighter-track coverage · ${o.tracking.fighter_track_interruptions} interruption${o.tracking.fighter_track_interruptions === 1 ? "" : "s"}`}
        />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <section className="panel">
          <div className="panel-header"><h3 className="label-caps">Pose extraction status</h3></div>
          <div className="space-y-3 px-5 py-4">
            <div className="flex flex-wrap gap-2 text-sm">
              {(["completed", "processing", "queued", "failed", "cancelled"] as const).map((s) => (
                <span key={s} className="flex items-center gap-1.5"><StatusBadge status={s} /><span className="text-ink-200 tabular">{o.job_counts[s] ?? 0}</span></span>
              ))}
            </div>
            {o.active_jobs.length > 0 ? (
              <ul className="space-y-3 border-t border-ink-700/60 pt-3">
                {o.active_jobs.map((j) => (
                  <li key={j.id}>
                    <Link to={`/tracking/${j.video_id}`} className="block text-sm text-ink-100 hover:underline">{j.video_title}</Link>
                    <div className="mt-1 flex items-center gap-2">
                      <ProgressBar value={j.progress} tone="sky" />
                      <span className="w-10 text-right text-xs text-ink-400 tabular">{Math.round(j.progress * 100)}%</span>
                    </div>
                    <div className="text-[11px] text-ink-500">{j.status === "queued" ? "Queued" : STAGE_LABELS[j.current_stage ?? ""] ?? j.current_stage}</div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="border-t border-ink-700/60 pt-3 text-xs text-ink-500">
                No analysis running. {done ? `${done} completed` : "Nothing analysed yet"}{failed ? ` · ${failed} failed` : ""}.
              </p>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3 className="label-caps">Recently analysed</h3>
            <Link to="/tracking" className="flex items-center gap-1 text-xs text-ink-400 hover:text-ink-100">Open <ChevronRight className="size-3" /></Link>
          </div>
          {o.recent.length === 0 ? (
            <EmptyState icon={<ScanEye className="size-5" />} title="No analyses yet" body="Open a processed video in Fighter Tracking and click Analyse." />
          ) : (
            <ul className="divide-y divide-ink-700/60">
              {o.recent.map((j) => (
                <li key={j.id}>
                  <Link to={`/tracking/${j.video_id}`} className="block px-5 py-2.5 hover:bg-ink-800/60">
                    <div className="truncate text-sm text-ink-100">{j.video_title}</div>
                    <div className="text-[11px] text-ink-500 tabular">
                      {j.stats?.frames_analysed} frames · {j.stats?.tracks_non_fragment} tracks · conf {pct(j.stats?.avg_detection_confidence)} · {formatDate(j.finished_at)}
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="panel overflow-hidden">
          <div className="panel-header"><h3 className="label-caps">Annotated video preview</h3></div>
          {o.latest_annotated ? (
            <>
              <video src={`${apiUrl(o.latest_annotated.url)}#t=0.5`} className="aspect-video w-full bg-black" controls muted preload="metadata" />
              <Link to={`/tracking/${o.latest_annotated.video_id}`} className="block truncate px-5 py-2.5 text-xs text-ink-300 hover:text-white">{o.latest_annotated.title} →</Link>
            </>
          ) : (
            <EmptyState title="No annotated video yet" body="Annotated videos with boxes, IDs and skeletons appear after analysis." />
          )}
        </section>
      </div>
    </section>
  );
}
