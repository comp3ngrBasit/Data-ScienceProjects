import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Check, ChevronDown, CircleAlert, Download, Pencil, Play, RotateCw, Target, Trash2, TriangleAlert, Users } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, api, apiUrl } from "../api/client";
import type { Fighter, Frame, Job, VideoDetail } from "../api/types";
import { EmptyState, ErrorBlock, Field, KV, LoadingBlock, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn, formatBytes, formatDate, formatDuration, formatFps, formatResolution, formatTimestamp } from "../lib/format";
import { useVideoActions } from "./VideoLibraryPage";

function ActiveJobCard({ job }: { job: Job }) {
  return (
    <div className="panel border-sky-500/30 px-5 py-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm">
          <Spinner className="text-sky-300" />
          <span className="text-ink-100">{job.status === "queued" ? "Waiting in queue…" : `Processing · ${job.stage_log?.find((s) => s.status === "running")?.label ?? job.current_stage ?? "starting"}`}</span>
        </div>
        <span className="text-sm text-ink-300 tabular">{Math.round(job.progress * 100)}%</span>
      </div>
      <ProgressBar value={job.progress} tone="sky" className="mt-3" />
      {job.stage_log && job.stage_log.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {job.stage_log.map((s) => (
            <span key={s.stage} className={cn("rounded border px-2 py-0.5 text-[11px]", s.status === "completed" ? "border-emerald-500/30 text-emerald-300" : s.status === "running" ? "border-sky-500/40 text-sky-200" : "border-blood-600/40 text-blood-300")}>
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function FighterAssignment({ video, fighters, onSaved }: { video: VideoDetail; fighters: Fighter[]; onSaved: (v: VideoDetail) => void }) {
  const toast = useToast();
  const [assigned, setAssigned] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [targetNotes, setTargetNotes] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setAssigned(video.fighters.map((f) => f.id));
    setTarget(video.target_fighter?.id ?? "");
    setTargetNotes(video.target_fighter_notes ?? "");
    setNotes(video.notes ?? "");
  }, [video.id, video.updated_at]); // eslint-disable-line react-hooks/exhaustive-deps

  const dirty =
    JSON.stringify([...assigned].sort()) !== JSON.stringify(video.fighters.map((f) => f.id).sort()) ||
    target !== (video.target_fighter?.id ?? "") ||
    targetNotes !== (video.target_fighter_notes ?? "") ||
    notes !== (video.notes ?? "");

  const toggle = (id: string) => {
    setAssigned((a) => {
      if (a.includes(id)) {
        if (target === id) setTarget("");
        return a.filter((x) => x !== id);
      }
      return a.length >= 10 ? a : [...a, id];
    });
  };

  const save = async () => {
    setBusy(true);
    try {
      const v = await api.updateVideo(video.id, {
        fighter_ids: assigned,
        target_fighter_id: target || null,
        target_fighter_notes: targetNotes,
        notes,
      });
      onSaved(v);
      toast("success", "Fighter assignment saved");
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps flex items-center gap-2"><Users className="size-3.5" /> Fighters in this video</h2>
        <span className="text-[11px] text-ink-500">Manual assignment</span>
      </div>
      <div className="space-y-4 px-5 py-4">
        {fighters.length === 0 ? (
          <p className="text-sm text-ink-400">
            No fighter profiles yet. <Link to="/fighters" className="text-ink-200 underline underline-offset-2">Create one</Link> to assign it to this video.
          </p>
        ) : (
          <>
            <div>
              <div className="mb-1.5 text-xs font-medium text-ink-300">Appearing in footage</div>
              <div className="flex flex-wrap gap-1.5">
                {fighters.map((f) => {
                  const on = assigned.includes(f.id);
                  return (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => toggle(f.id)}
                      className={cn("flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors", on ? "border-ink-400 bg-ink-700 text-white" : "border-ink-600 text-ink-400 hover:border-ink-500 hover:text-ink-200")}
                    >
                      {on && <Check className="size-3" />}
                      {f.name}
                    </button>
                  );
                })}
              </div>
            </div>
            <Field label="Fighter to analyze" hint="The athlete future detection & prediction phases will focus on.">
              <select
                className="input"
                value={target}
                onChange={(e) => {
                  const id = e.target.value;
                  setTarget(id);
                  if (id && !assigned.includes(id)) setAssigned((a) => [...a, id]);
                }}
              >
                <option value="">— Not selected —</option>
                {fighters.map((f) => (
                  <option key={f.id} value={f.id}>{f.name}{f.stance !== "Unknown" ? ` (${f.stance})` : ""}</option>
                ))}
              </select>
            </Field>
            <Field label="How to identify them in this footage" hint="e.g. “red corner, black shorts” — helps later detection phases.">
              <input className="input" value={targetNotes} maxLength={500} onChange={(e) => setTargetNotes(e.target.value)} placeholder="Corner, shorts colour, tattoos…" />
            </Field>
          </>
        )}
        <Field label="Video notes">
          <textarea className="input min-h-[70px] resize-y" value={notes} maxLength={5000} onChange={(e) => setNotes(e.target.value)} placeholder="Event, round, opponent, source…" />
        </Field>
        <div className="flex justify-end">
          <button className="btn-primary" disabled={!dirty || busy} onClick={save}>{busy && <Spinner className="text-white" />}Save assignment</button>
        </div>
      </div>
    </section>
  );
}

function FrameStrip({ videoId, onSeek, total }: { videoId: string; onSeek: (t: number) => void; total: number }) {
  const { data, loading } = useData(() => api.listFrames(videoId, 1000), [videoId, total]);
  const [metric, setMetric] = useState<"sharpness" | "brightness" | "contrast">("sharpness");
  const [showAll, setShowAll] = useState(false);
  if (loading && !data) return <LoadingBlock label="Loading frames…" />;
  if (!data || data.items.length === 0) return <p className="px-5 py-6 text-sm text-ink-500">No sampled frames yet.</p>;
  const frames = data.items;
  const visible = showAll ? frames : frames.slice(0, 24);
  const hasImages = frames.some((f) => f.image_url);
  return (
    <div className="space-y-4 px-5 py-4">
      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs text-ink-400">Per-frame {metric} across the video</span>
          <select className="input w-auto py-1 text-xs" value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)} aria-label="Metric">
            <option value="sharpness">Sharpness (Laplacian variance)</option>
            <option value="brightness">Brightness (mean luma)</option>
            <option value="contrast">Contrast (luma std-dev)</option>
          </select>
        </div>
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={frames} margin={{ top: 4, right: 8, left: -12, bottom: 0 }} onClick={(s) => {
              const idx = s?.activeTooltipIndex;
              if (idx != null && frames[Number(idx)]) onSeek(frames[Number(idx)].timestamp_seconds);
            }}>
              <CartesianGrid vertical={false} stroke="#26262d" />
              <XAxis dataKey="timestamp_seconds" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(t: number) => formatDuration(t)} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "#33333c" }} />
              <YAxis tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} width={48} />
              <Tooltip
                cursor={{ stroke: "#4b4b56" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const f = payload[0].payload as Frame;
                  return (
                    <div className="rounded-md border border-ink-600 bg-ink-800 px-3 py-2 text-xs shadow-lg">
                      <div className="text-ink-400">{formatTimestamp(f.timestamp_seconds)} · frame {f.frame_index}</div>
                      <div className="mt-0.5 font-medium capitalize text-white tabular">{metric}: {(f[metric] ?? 0).toFixed(1)}</div>
                      <div className="text-[10px] text-ink-500">click to seek</div>
                    </div>
                  );
                }}
              />
              <Line type="monotone" dataKey={metric} stroke="#e11d2e" strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: "#131317", strokeWidth: 2 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      {hasImages ? (
        <>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
            {visible.map((f) => (
              <button key={f.id} className="group relative overflow-hidden rounded border border-ink-700 bg-ink-950 hover:border-blood-500" onClick={() => onSeek(f.timestamp_seconds)} title={`Frame ${f.frame_index} · sharpness ${f.sharpness}`}>
                {f.image_url && <img src={apiUrl(f.image_url)} alt={`Frame ${f.frame_index}`} loading="lazy" className="aspect-video w-full object-cover" />}
                <span className="absolute bottom-0 left-0 right-0 bg-black/70 px-1 py-0.5 text-[10px] text-ink-200 tabular">{formatTimestamp(f.timestamp_seconds)}</span>
              </button>
            ))}
          </div>
          {frames.length > 24 && (
            <button className="btn-ghost w-full text-xs" onClick={() => setShowAll((s) => !s)}>
              <ChevronDown className={cn("size-3.5 transition-transform", showAll && "rotate-180")} />
              {showAll ? "Show fewer" : `Show all ${frames.length} frames`}
            </button>
          )}
        </>
      ) : (
        <p className="text-xs text-ink-500">Frame images were not saved for this run (Settings → “Save frame images” was off); only timestamps and stats were stored.</p>
      )}
    </div>
  );
}

function JobHistory({ jobs }: { jobs: Job[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (jobs.length === 0) return <p className="px-5 py-6 text-sm text-ink-500">No processing runs yet.</p>;
  return (
    <ul className="divide-y divide-ink-700/60">
      {jobs.map((j) => (
        <li key={j.id}>
          <button className="flex w-full items-center gap-3 px-5 py-3 text-left hover:bg-ink-800/50" onClick={() => setOpen(open === j.id ? null : j.id)}>
            <span className="w-16 text-xs text-ink-400">Attempt {j.attempt}</span>
            <StatusBadge status={j.status} />
            <span className="flex-1 truncate text-xs text-ink-400">{formatDate(j.started_at ?? j.created_at)}{j.duration_seconds != null && ` · ${j.duration_seconds.toFixed(1)}s`}</span>
            <ChevronDown className={cn("size-4 text-ink-500 transition-transform", open === j.id && "rotate-180")} />
          </button>
          {open === j.id && (
            <div className="space-y-2 bg-ink-900/60 px-5 pb-4 pt-1 text-xs">
              {j.error_message && <p className="flex gap-2 text-blood-300"><CircleAlert className="size-3.5 shrink-0" />{j.error_message}</p>}
              {j.warnings?.map((w) => <p key={w} className="flex gap-2 text-amber-200"><TriangleAlert className="size-3.5 shrink-0" />{w}</p>)}
              {j.stage_log && (
                <table className="w-full">
                  <tbody>
                    {j.stage_log.map((s) => (
                      <tr key={s.stage} className="border-b border-ink-700/40 last:border-0">
                        <td className="py-1.5 text-ink-300">{s.label}</td>
                        <td className={cn("py-1.5", s.status === "completed" ? "text-emerald-300" : s.status === "failed" ? "text-blood-300" : "text-ink-400")}>{s.status}</td>
                        <td className="py-1.5 text-right text-ink-400 tabular">{s.seconds != null ? `${s.seconds}s` : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {j.settings_snapshot && (
                <p className="text-ink-500">
                  Settings: {String(j.settings_snapshot.sampling_mode)} sampling
                  {j.settings_snapshot.sampling_mode === "interval" ? ` every ${j.settings_snapshot.sample_interval_seconds}s` : ` × ${j.settings_snapshot.uniform_frame_count}`}, max {String(j.settings_snapshot.max_sampled_frames)} frames
                </p>
              )}
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

function QualityPanel({ v }: { v: VideoDetail }) {
  const q = v.quality;
  if (!q || q.score == null) return <p className="px-5 py-6 text-sm text-ink-500">Quality is assessed after processing.</p>;
  const tone = q.rating === "Good" ? "text-emerald-300" : q.rating === "Fair" ? "text-amber-200" : "text-blood-300";
  return (
    <div className="px-5 py-4">
      <div className="flex items-end justify-between">
        <div>
          <div className={cn("font-display text-3xl font-semibold uppercase", tone)}>{q.rating}</div>
          <div className="text-xs text-ink-500">Heuristic score {q.score}/100 · {q.frames_analyzed} frames</div>
        </div>
        <span className="rounded border border-ink-600 px-2 py-0.5 text-xs text-ink-300">{q.resolution_class}</span>
      </div>
      <ProgressBar value={q.score / 100} tone={q.rating === "Good" ? "green" : q.rating === "Fair" ? "sky" : "red"} className="mt-3" />
      <dl className="mt-3">
        <KV k="Mean sharpness" v={q.mean_sharpness?.toFixed(1)} />
        <KV k="Blurry frames" v={q.blurry_frame_pct != null ? `${q.blurry_frame_pct}%` : "—"} />
        <KV k="Mean brightness" v={q.mean_brightness != null ? `${q.mean_brightness.toFixed(0)} / 255` : "—"} />
        <KV k="Dark / overexposed" v={`${q.dark_frame_pct ?? 0}% / ${q.overexposed_frame_pct ?? 0}%`} />
        <KV k="Mean contrast" v={q.mean_contrast?.toFixed(1)} />
      </dl>
      {(q.issues.length > 0 || (q.warnings?.length ?? 0) > 0) && (
        <ul className="mt-3 space-y-1.5 text-xs">
          {q.issues.map((i) => <li key={i} className="flex gap-2 text-amber-200"><TriangleAlert className="size-3.5 shrink-0" />{i}</li>)}
          {q.warnings?.map((w) => <li key={w} className="flex gap-2 text-ink-400"><TriangleAlert className="size-3.5 shrink-0" />{w}</li>)}
        </ul>
      )}
      <p className="mt-3 text-[11px] leading-relaxed text-ink-500">Signal-level heuristics (resolution, fps, bitrate, exposure, Laplacian sharpness) — not a learned quality model.</p>
    </div>
  );
}

export default function VideoDetailPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playError, setPlayError] = useState(false);
  const { data: v, error, loading, reload, setData } = useData(() => api.getVideo(id), [id], (d) =>
    d && (d.status === "queued" || d.status === "processing") ? 1500 : undefined,
  );
  const { data: fighters } = useData(() => api.listFighters(), []);
  const actions = useVideoActions(() => reload(true));

  useEffect(() => setPlayError(false), [v?.stream_url]);

  if (loading && !v) return <LoadingBlock />;
  if (error && !v)
    return (
      <>
        <Link to="/videos" className="btn-ghost mb-4 -ml-3"><ArrowLeft className="size-4" />Library</Link>
        <ErrorBlock message={error.includes("not found") ? "This video no longer exists." : error} onRetry={() => reload()} />
      </>
    );
  if (!v) return null;

  const activeJob = v.jobs.find((j) => j.status === "queued" || j.status === "processing");
  const seek = (t: number) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = t;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  const reprocess = async () => {
    try {
      await api.processVideo(v.id);
      toast("info", v.status === "failed" ? "Retry queued" : "Processing queued");
      void reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Could not queue processing");
    }
  };

  return (
    <>
      <Link to="/videos" className="btn-ghost mb-3 -ml-3"><ArrowLeft className="size-4" />Library</Link>
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="truncate font-display text-3xl font-bold uppercase tracking-wide text-white" title={v.title}>{v.title}</h1>
            <button className="btn-ghost shrink-0 p-1.5" onClick={() => actions.setRenaming(v)} title="Rename"><Pencil className="size-4" /></button>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-400">
            <StatusBadge status={v.status} />
            <span className="truncate">{v.original_filename}</span>
            <span>Uploaded {formatDate(v.uploaded_at)}</span>
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          {!activeJob && (
            <button className="btn-secondary" onClick={reprocess}>
              <RotateCw className="size-4" />
              {v.status === "failed" ? "Retry processing" : v.status === "uploaded" ? "Process now" : "Reprocess"}
            </button>
          )}
          <a className="btn-secondary" href={apiUrl(`/api/videos/${v.id}/download`)}><Download className="size-4" />Original</a>
          <button className="btn-danger" onClick={() => actions.setDeleting(v)}><Trash2 className="size-4" />Delete</button>
        </div>
      </div>

      {v.status === "failed" && v.error_message && !activeJob && (
        <div className="mb-4 flex items-start gap-3 rounded-lg border border-blood-700/60 bg-blood-900/30 px-4 py-3 text-sm">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-blood-400" />
          <div className="flex-1">
            <div className="font-medium text-blood-300">Processing failed</div>
            <div className="text-ink-300">{v.error_message}</div>
          </div>
          <button className="btn-secondary shrink-0 py-1.5" onClick={reprocess}><RotateCw className="size-3.5" />Retry</button>
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <div className="panel overflow-hidden">
            <div className="relative bg-black">
              {playError ? (
                <div className="flex aspect-video flex-col items-center justify-center gap-2 px-6 text-center text-sm text-ink-400">
                  <Play className="size-8 text-ink-600" />
                  Your browser can't play this format ({v.file_ext.slice(1).toUpperCase()}{v.video_codec ? ` / ${v.video_codec}` : ""}).
                  {v.status !== "completed" ? " A browser preview will be generated during processing." : " Enable “Generate preview proxy” in Settings and reprocess, or download the original."}
                </div>
              ) : (
                <video
                  ref={videoRef}
                  key={v.stream_url}
                  src={apiUrl(v.stream_url)}
                  controls
                  preload="metadata"
                  poster={v.thumbnail_url ? apiUrl(v.thumbnail_url) : undefined}
                  className="aspect-video w-full bg-black"
                  onError={() => setPlayError(true)}
                />
              )}
            </div>
            {v.has_preview_proxy && (
              <div className="border-t border-ink-700/60 px-4 py-2 text-[11px] text-ink-500">Playing a browser-compatible H.264 preview. The original {v.file_ext.slice(1).toUpperCase()} is preserved for analysis.</div>
            )}
          </div>

          {activeJob && <ActiveJobCard job={activeJob} />}

          <section className="panel">
            <div className="panel-header">
              <h2 className="label-caps">Extracted frames</h2>
              <span className="text-xs text-ink-400 tabular">
                {v.sampled_frame_count} sampled
                {v.sampling_config && ` · ${v.sampling_config.mode === "interval" ? `every ${v.sampling_config.interval_seconds}s` : `${v.sampling_config.uniform_frame_count} uniform`} · ≤${v.sampling_config.frame_max_width}px`}
              </span>
            </div>
            {v.status === "completed" ? (
              <FrameStrip videoId={v.id} onSeek={seek} total={v.sampled_frame_count} />
            ) : (
              <EmptyState title={activeJob ? "Frames are being extracted" : "No frames extracted"} body={activeJob ? "This updates automatically when processing completes." : "Process the video to sample frames."} />
            )}
          </section>

          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Processing history</h2><span className="text-xs text-ink-400">{v.jobs.length} run{v.jobs.length === 1 ? "" : "s"}</span></div>
            <JobHistory jobs={v.jobs} />
          </section>
        </div>

        <div className="space-y-4">
          {v.target_fighter && (
            <div className="panel flex items-center gap-3 border-blood-700/50 px-5 py-4">
              <div className="grid size-10 place-items-center rounded-full bg-blood-900/50 text-blood-400"><Target className="size-5" /></div>
              <div className="min-w-0">
                <div className="label-caps">Analyzing</div>
                <Link to={`/fighters/${v.target_fighter.id}`} className="block truncate font-medium text-white hover:underline">{v.target_fighter.name}</Link>
                {v.target_fighter_notes && <div className="truncate text-xs text-ink-400">{v.target_fighter_notes}</div>}
              </div>
            </div>
          )}
          <FighterAssignment video={v} fighters={fighters ?? []} onSaved={(nv) => setData(nv)} />

          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Video metadata</h2><span className="text-[11px] text-ink-500">{v.metadata_source ?? "pending"}</span></div>
            <dl className="px-5 py-2">
              <KV k="Duration" v={v.duration_seconds != null ? `${formatDuration(v.duration_seconds)} (${v.duration_seconds.toFixed(2)}s)` : "—"} />
              <KV k="Resolution" v={formatResolution(v.width, v.height)} />
              <KV k="Frame rate" v={formatFps(v.fps)} />
              <KV k="Total frames" v={v.frame_count != null ? `${v.frame_count.toLocaleString()}${v.frame_count_source && v.frame_count_source !== "container" ? ` (${v.frame_count_source})` : ""}` : "—"} />
              <KV k="Video codec" v={v.video_codec ? `${v.video_codec}${v.pixel_format ? ` · ${v.pixel_format}` : ""}` : "—"} />
              <KV k="Container" v={v.container_format} />
              <KV k="Bitrate" v={v.bitrate_kbps ? `${v.bitrate_kbps.toLocaleString()} kb/s` : "—"} />
              <KV k="Audio" v={v.has_audio == null ? "Unknown" : v.has_audio ? `${v.audio_codec ?? "yes"}${v.audio_sample_rate ? ` · ${v.audio_sample_rate / 1000} kHz` : ""}${v.audio_channels ? ` · ${v.audio_channels}ch` : ""}` : "No audio track"} />
              {!!v.rotation && <KV k="Rotation" v={`${v.rotation}°`} />}
              <KV k="File size" v={formatBytes(v.file_size)} />
              <KV k="Processed" v={formatDate(v.processed_at)} />
              <KV k="SHA-256" v={`${v.sha256.slice(0, 16)}…`} mono />
            </dl>
          </section>

          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Basic quality</h2></div>
            <QualityPanel v={v} />
          </section>
        </div>
      </div>
      {actions.dialogs(() => navigate("/videos"))}
    </>
  );
}
