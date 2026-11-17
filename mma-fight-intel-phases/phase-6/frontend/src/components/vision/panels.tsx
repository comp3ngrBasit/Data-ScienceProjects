import { useEffect, useMemo, useState } from "react";
import { CircleAlert, RotateCw, ScanEye, Trash2, TriangleAlert } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Fighter } from "../../api/types";
import { STAGE_LABELS, trackColor, visionApi, type FrameDetail, type MovementRow, type Role, type Track, type Tracking, type VisionStatus } from "../../api/vision";
import { useToast } from "../../context/ToastContext";
import { cn, formatDate, formatTimestamp } from "../../lib/format";
import { KV, ProgressBar, Spinner, StatusBadge } from "../ui";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const num = (v: number | null | undefined, d = 2) => (v == null ? "—" : v.toFixed(d));

/* ------------------------------------------------------------------ Analysis status */
export function AnalysisCard({
  status,
  onAnalyze,
  onDelete,
  onRetry,
  busy,
}: {
  status: VisionStatus;
  onAnalyze: () => void;
  onDelete: () => void;
  onRetry: (jobId: string) => void;
  busy: boolean;
}) {
  const active = status.active_job;
  const cur = status.current;
  const lastFailed = status.history.find((j) => j.status === "failed" || j.status === "cancelled");
  const showFailure = lastFailed && (!cur || new Date(lastFailed.created_at) > new Date(cur.created_at)) && !active;
  const st = cur?.stats;
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps flex items-center gap-2"><ScanEye className="size-3.5" /> Analysis</h2>
        <div className="flex gap-1.5">
          {!active && (
            <button className="btn-primary px-3 py-1.5 text-xs" disabled={!status.can_analyze || busy} onClick={onAnalyze} title={status.reason ?? ""}>
              {busy ? <Spinner className="size-3 text-white" /> : <ScanEye className="size-3.5" />}
              {cur ? "Re-analyse" : "Analyse video"}
            </button>
          )}
          {(cur || active || status.history.length > 0) && (
            <button className="btn-ghost p-1.5 hover:text-blood-400" onClick={onDelete} title="Delete analysis"><Trash2 className="size-4" /></button>
          )}
        </div>
      </div>
      <div className="space-y-3 px-5 py-4 text-sm">
        {!status.can_analyze && !active && status.reason && <p className="text-xs text-amber-200">{status.reason}</p>}
        {active && (
          <div>
            <div className="flex items-center justify-between text-xs">
              <span className="flex items-center gap-2 text-sky-200"><Spinner className="size-3 text-sky-300" />
                {active.status === "queued" ? "Queued" : STAGE_LABELS[active.current_stage ?? ""] ?? active.current_stage ?? "Working"}</span>
              <span className="text-ink-300 tabular">{Math.round(active.progress * 100)}%</span>
            </div>
            <ProgressBar value={active.progress} tone="sky" className="mt-2" />
            <p className="mt-2 text-[11px] text-ink-500">Runs in the background — you can leave this page.</p>
          </div>
        )}
        {showFailure && lastFailed && (
          <div className="flex items-start gap-2 rounded-md border border-blood-700/60 bg-blood-900/30 px-3 py-2 text-xs">
            <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-blood-400" />
            <div className="flex-1 text-ink-200">
              <div className="font-medium text-blood-300">Attempt {lastFailed.attempt} {lastFailed.status}</div>
              {lastFailed.error_message}
            </div>
            <button className="btn-secondary px-2 py-1 text-xs" onClick={() => onRetry(lastFailed.id)}><RotateCw className="size-3" />Retry</button>
          </div>
        )}
        {cur && st ? (
          <>
            <dl>
              <KV k="Frames analysed" v={`${st.frames_analysed.toLocaleString()} of ${st.frames_total.toLocaleString()} (${st.stride === 1 ? "every frame" : `1 in ${st.stride}`} · ${st.analysis_fps} fps)`} />
              <KV k="People detected (max / frame)" v={st.max_people_in_frame} />
              <KV k="Tracks" v={`${st.tracks_non_fragment} main · ${st.fragment_tracks} short fragments`} />
              <KV k="Detection confidence" v={pct(st.avg_detection_confidence)} />
              <KV k="Pose confidence (core joints)" v={pct(st.avg_pose_confidence)} />
              <KV k="Camera cuts" v={st.scene_cuts?.length ?? 0} />
              <KV k="Model · device" v={`${st.model} · ${st.device}`} />
              <KV k="Speed" v={`${st.inference_ms_per_frame} ms/frame · ${st.processing_seconds}s total${st.realtime_factor ? ` · ${st.realtime_factor}× real time` : ""}`} />
              <KV k="Completed" v={formatDate(cur.finished_at)} />
            </dl>
            {cur.warnings?.map((w) => <p key={w} className="flex gap-2 text-xs text-amber-200"><TriangleAlert className="size-3.5 shrink-0" />{w}</p>)}
          </>
        ) : !active && !showFailure ? (
          <p className="text-xs text-ink-400">Not analysed yet. Analysis detects every person, tracks them with consistent IDs, extracts 17 body landmarks per person and computes movement features.</p>
        ) : null}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Timeline */
export function TrackTimeline({ tracking, totalFrames, frame, onSeek }: { tracking: Tracking; totalFrames: number; frame: number; onSeek: (f: number) => void }) {
  const lanes = tracking.tracks.filter((t) => t.status !== "fragment").slice(0, 10);
  const hidden = tracking.tracks.length - lanes.length;
  const total = Math.max(totalFrames, 1);
  const click = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(Math.round(((e.clientX - r.left) / r.width) * total));
  };
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps">Tracking timeline</h2>
        <span className="text-[11px] text-ink-500">
          {tracking.analysed_frames} analysed frames · every {tracking.stride} frame{tracking.stride > 1 ? "s" : ""}
          {tracking.scene_cuts.length > 0 && ` · ${tracking.scene_cuts.length} camera cut${tracking.scene_cuts.length > 1 ? "s" : ""}`}
          {hidden > 0 && ` · ${hidden} short fragments hidden`}
        </span>
      </div>
      <div className="px-4 py-3">
        <div className="relative cursor-pointer select-none" onClick={click}>
          {lanes.map((t) => (
            <div key={t.track_id} className="flex items-center gap-2 py-[3px]">
              <span className="w-28 shrink-0 truncate text-[11px] text-ink-300">
                <span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: t.color }} />
                ID {t.track_id}{t.assigned_fighter ? ` · ${t.assigned_fighter.name}` : t.role !== "unreviewed" ? ` · ${t.role}` : ""}
              </span>
              <div className="relative h-3 flex-1 rounded-sm bg-ink-800">
                {t.segments.map(([a, b], i) => (
                  <div key={i} className="absolute inset-y-0 rounded-sm" title={`ID ${t.track_id}: frames ${a}–${b}`}
                    style={{ left: `${(a / total) * 100}%`, width: `${Math.max(((b - a + tracking.stride) / total) * 100, 0.4)}%`, background: t.color, opacity: 0.85 }} />
                ))}
              </div>
            </div>
          ))}
          <div className="pointer-events-none absolute inset-y-0 left-[7.5rem] right-0">
            {tracking.scene_cuts.map((c) => (
              <div key={c} className="absolute inset-y-0 w-px border-l border-dashed border-ink-400/60" style={{ left: `${(c / total) * 100}%` }} title={`Camera cut at frame ${c}`} />
            ))}
            <div className="absolute inset-y-[-4px] w-0.5 bg-white shadow-[0_0_6px_rgba(255,255,255,0.6)]" style={{ left: `${(frame / total) * 100}%` }} />
          </div>
        </div>
        <div className="mt-2 flex justify-between pl-[7.5rem] text-[10px] text-ink-500 tabular">
          <span>0:00</span>
          <span>{formatTimestamp(total / (tracking.fps || 30))}</span>
        </div>
        {lanes.length === 0 && <p className="py-4 text-center text-xs text-ink-500">No people were tracked in this video.</p>}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Track assignment */
export function TrackPanel({
  videoId,
  tracking,
  fighters,
  onSaved,
  onSeek,
}: {
  videoId: string;
  tracking: Tracking;
  fighters: Fighter[];
  onSaved: (t: Tracking) => void;
  onSeek: (f: number) => void;
}) {
  const toast = useToast();
  const [draft, setDraft] = useState<Record<number, { role: Role; fighter_id: string | null }>>({});
  const [busy, setBusy] = useState(false);
  const [showFragments, setShowFragments] = useState(false);
  useEffect(() => {
    setDraft(Object.fromEntries(tracking.tracks.map((t) => [t.track_id, { role: t.role, fighter_id: t.assigned_fighter?.id ?? null }])));
  }, [tracking]);

  const changed = tracking.tracks.filter((t) => {
    const d = draft[t.track_id];
    return d && (d.role !== t.role || d.fighter_id !== (t.assigned_fighter?.id ?? null));
  });
  const save = async () => {
    setBusy(true);
    try {
      const res = await visionApi.assign(videoId, changed.map((t) => ({ track_id: t.track_id, fighter_id: draft[t.track_id].fighter_id, role: draft[t.track_id].role })));
      onSaved(res);
      toast("success", `Saved ${changed.length} track assignment${changed.length > 1 ? "s" : ""}`);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  };
  const list = tracking.tracks.filter((t) => showFragments || t.status !== "fragment");
  const c = tracking.consistency;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps">Tracks → fighters</h2>
        <button className="btn-primary px-3 py-1.5 text-xs" disabled={!changed.length || busy} onClick={save}>
          {busy && <Spinner className="size-3 text-white" />}Save{changed.length ? ` (${changed.length})` : ""}
        </button>
      </div>
      <div className="grid grid-cols-3 gap-px border-b border-ink-700/70 bg-ink-700/50 text-center">
        {[
          ["Fighter coverage", pct(c.fighter_coverage)],
          ["Interruptions", c.fighter_interruptions],
          ["Tracks / max people", `${c.tracks_non_fragment} / ${c.max_people_in_frame ?? "—"}`],
        ].map(([k, v]) => (
          <div key={k as string} className="bg-ink-850 px-2 py-2.5">
            <div className="font-display text-xl font-semibold text-white tabular">{v}</div>
            <div className="text-[10px] uppercase tracking-wider text-ink-500">{k}</div>
          </div>
        ))}
      </div>
      <p className="border-b border-ink-700/70 px-4 py-2 text-[11px] leading-relaxed text-ink-500">
        Not everyone detected is a fighter. Mark each track's role and link fighters to their profiles. {c.note}
      </p>
      <ul className="max-h-[480px] divide-y divide-ink-700/50 overflow-y-auto">
        {list.map((t) => {
          const d = draft[t.track_id] ?? { role: t.role, fighter_id: null };
          return (
            <li key={t.track_id} className="px-4 py-3">
              <div className="flex items-center gap-2">
                <button className="flex items-center gap-2 text-left" onClick={() => onSeek(t.first_frame)} title="Jump to first appearance">
                  <span className="grid size-7 place-items-center rounded font-display text-sm font-bold text-ink-950" style={{ background: t.color }}>{t.track_id}</span>
                </button>
                <div className="min-w-0 flex-1 text-[11px] text-ink-400 tabular">
                  <div className="flex items-center gap-1.5">
                    {t.suggested_role === "fighter" && <span className="rounded bg-blood-900/50 px-1.5 py-px text-[10px] font-medium uppercase tracking-wide text-blood-300" title="Heuristic suggestion: present most of the time and large on screen">likely fighter</span>}
                    <span className={cn(t.status === "fragment" && "text-ink-500", t.status === "interrupted" && "text-amber-200/80")}>{t.status}</span>
                  </div>
                  {formatTimestamp(t.first_timestamp)}–{formatTimestamp(t.last_timestamp)} · {t.frames_detected} frames · {pct(t.video_coverage)} of video · conf {num(t.avg_confidence)}
                  {t.interruptions > 0 && ` · ${t.interruptions} gap${t.interruptions > 1 ? "s" : ""}`}
                </div>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <select
                  className="input py-1.5 text-xs"
                  value={d.role}
                  aria-label={`Role for track ${t.track_id}`}
                  onChange={(e) => setDraft({ ...draft, [t.track_id]: { role: e.target.value as Role, fighter_id: e.target.value === "fighter" ? d.fighter_id : null } })}
                >
                  <option value="unreviewed">Unreviewed</option>
                  <option value="fighter">Fighter</option>
                  <option value="referee">Referee</option>
                  <option value="other">Other / not a fighter</option>
                </select>
                <select
                  className="input py-1.5 text-xs"
                  value={d.fighter_id ?? ""}
                  aria-label={`Fighter profile for track ${t.track_id}`}
                  onChange={(e) => setDraft({ ...draft, [t.track_id]: { role: e.target.value ? "fighter" : d.role === "fighter" ? "unreviewed" : d.role, fighter_id: e.target.value || null } })}
                >
                  <option value="">— No profile —</option>
                  {fighters.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
                </select>
              </div>
            </li>
          );
        })}
      </ul>
      {tracking.consistency.fragment_tracks > 0 && (
        <button className="w-full border-t border-ink-700/70 py-2 text-xs text-ink-400 hover:text-ink-100" onClick={() => setShowFragments((s) => !s)}>
          {showFragments ? "Hide" : "Show"} {tracking.consistency.fragment_tracks} short fragment track{tracking.consistency.fragment_tracks > 1 ? "s" : ""}
        </button>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ Frame inspector */
const STATE_STYLE: Record<string, string> = {
  moving: "text-emerald-300",
  stationary: "text-sky-300",
  low_confidence: "text-amber-200",
  missing: "text-ink-500",
};

export function FrameInspector({ videoId, frame, stride, jobId }: { videoId: string; frame: number; stride: number; jobId: string }) {
  const analysed = frame - (frame % Math.max(stride, 1));
  const [data, setData] = useState<FrameDetail>();
  const [openKp, setOpenKp] = useState<number | null>(null);
  useEffect(() => {
    let alive = true;
    const t = window.setTimeout(() => {
      visionApi.frame(videoId, analysed).then((d) => alive && setData(d)).catch(() => undefined);
    }, 120);
    return () => {
      alive = false;
      window.clearTimeout(t);
    };
  }, [videoId, analysed, jobId]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps">Frame {analysed} details</h2>
        <span className="text-[11px] text-ink-500 tabular">{data?.timestamp != null ? formatTimestamp(data.timestamp) : ""}</span>
      </div>
      {!data ? (
        <div className="px-5 py-6 text-xs text-ink-500"><Spinner className="mr-2 inline size-3" />Loading…</div>
      ) : data.people.length === 0 ? (
        <p className="px-5 py-6 text-xs text-ink-500">No tracked people in this analysed frame.</p>
      ) : (
        <div className="divide-y divide-ink-700/50">
          {data.people.map((p) => {
            const m = p.movement;
            return (
              <div key={p.track_id} className="px-4 py-3 text-xs">
                <div className="flex items-center gap-2">
                  <span className="size-2.5 rounded-full" style={{ background: trackColor(p.track_id) }} />
                  <span className="font-medium text-ink-100">ID {p.track_id}{p.fighter_name ? ` · ${p.fighter_name}` : ""}</span>
                  {p.role && p.role !== "unreviewed" && <span className="text-ink-500">({p.role})</span>}
                  <span className="ml-auto text-ink-400 tabular">conf {p.confidence.toFixed(2)}</span>
                  {m && <span className={cn("font-medium", STATE_STYLE[m.state])}>{m.state.replace("_", " ")}</span>}
                </div>
                {m && m.state !== "missing" && (
                  <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-ink-400 tabular sm:grid-cols-3">
                    <span>Wrist L/R <b className="font-medium text-ink-200">{num(m.left_wrist_speed, 1)} / {num(m.right_wrist_speed, 1)}</b></span>
                    <span>Elbow L/R <b className="font-medium text-ink-200">{num(m.left_elbow_speed, 1)} / {num(m.right_elbow_speed, 1)}</b></span>
                    <span>Shoulder L/R <b className="font-medium text-ink-200">{num(m.left_shoulder_speed, 1)} / {num(m.right_shoulder_speed, 1)}</b></span>
                    <span>Hip L/R <b className="font-medium text-ink-200">{num(m.left_hip_speed, 1)} / {num(m.right_hip_speed, 1)}</b></span>
                    <span>Knee L/R <b className="font-medium text-ink-200">{num(m.left_knee_speed, 1)} / {num(m.right_knee_speed, 1)}</b></span>
                    <span>Body <b className="font-medium text-ink-200">{num(m.center_speed, 1)}</b>{m.direction_deg != null && ` → ${Math.round(m.direction_deg)}°`}</span>
                    <span>Elbow ∠ L/R <b className="font-medium text-ink-200">{num(m.left_elbow_angle, 0)}° / {num(m.right_elbow_angle, 0)}°</b></span>
                    <span>Knee ∠ L/R <b className="font-medium text-ink-200">{num(m.left_knee_angle, 0)}° / {num(m.right_knee_angle, 0)}°</b></span>
                    <span>Lean <b className="font-medium text-ink-200">{num(m.torso_lean_deg, 0)}°</b> · stance <b className="font-medium text-ink-200">{num(m.stance_width, 1)}</b></span>
                    <span>Guard <b className="font-medium text-ink-200">{num(m.guard_height, 2)}</b></span>
                    <span>Posture Δ <b className="font-medium text-ink-200">{num(m.posture_change, 3)}</b></span>
                    <span>Landmarks <b className="font-medium text-ink-200">{m.valid_landmarks}/13</b></span>
                  </div>
                )}
                <button className="mt-2 text-[11px] text-ink-500 underline decoration-ink-600 underline-offset-2 hover:text-ink-200" onClick={() => setOpenKp(openKp === p.track_id ? null : p.track_id)}>
                  {openKp === p.track_id ? "Hide" : "Show"} landmarks · bbox [{p.bbox.join(", ")}]
                </button>
                {openKp === p.track_id && (
                  <table className="mt-2 w-full font-mono text-[11px]">
                    <thead><tr className="text-left text-ink-500"><th className="font-normal">landmark</th><th className="font-normal">x</th><th className="font-normal">y</th><th className="font-normal">conf</th></tr></thead>
                    <tbody>
                      {p.keypoints.map((k) => (
                        <tr key={k.landmark} className={k.confidence < 0.35 ? "text-ink-600" : "text-ink-300"}>
                          <td>{k.landmark}</td><td>{k.x.toFixed(3)}</td><td>{k.y.toFixed(3)}</td><td>{k.confidence.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            );
          })}
          {data.distances.length > 0 && (
            <div className="px-4 py-3 text-xs text-ink-400">
              {data.distances.map((d) => (
                <div key={`${d.track_a}-${d.track_b}`} className="tabular">
                  Distance ID {d.track_a} ↔ ID {d.track_b}: <b className="font-medium text-ink-200">{num(d.distance_torso, 1)} torso lengths</b>
                  {" "}({num(d.distance_frame * 100, 0)}% of frame height)
                  {d.closing_speed != null && <> · {d.closing_speed > 0 ? "closing" : "separating"} {num(Math.abs(d.closing_speed), 1)}/s</>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <p className="border-t border-ink-700/70 px-4 py-2 text-[10px] text-ink-500">Speeds in torso lengths per second (resolution-independent). States: moving · stationary · low confidence (landmarks unreliable) · missing (not detected).</p>
    </section>
  );
}

/* ------------------------------------------------------------------ Movement charts */
const METRICS: { key: string; label: string; series: [keyof MovementRow, string][]; unit: string }[] = [
  { key: "wrist", label: "Wrist speed", series: [["left_wrist_speed", "Left"], ["right_wrist_speed", "Right"]], unit: "torso/s" },
  { key: "elbow", label: "Elbow speed", series: [["left_elbow_speed", "Left"], ["right_elbow_speed", "Right"]], unit: "torso/s" },
  { key: "shoulder", label: "Shoulder speed", series: [["left_shoulder_speed", "Left"], ["right_shoulder_speed", "Right"]], unit: "torso/s" },
  { key: "hip", label: "Hip speed", series: [["left_hip_speed", "Left"], ["right_hip_speed", "Right"]], unit: "torso/s" },
  { key: "knee", label: "Knee speed", series: [["left_knee_speed", "Left"], ["right_knee_speed", "Right"]], unit: "torso/s" },
  { key: "body", label: "Body (centre) speed", series: [["center_speed", "Body"]], unit: "torso/s" },
  { key: "elbow_angle", label: "Elbow angle", series: [["left_elbow_angle", "Left"], ["right_elbow_angle", "Right"]], unit: "°" },
  { key: "knee_angle", label: "Knee angle", series: [["left_knee_angle", "Left"], ["right_knee_angle", "Right"]], unit: "°" },
  { key: "guard", label: "Guard height", series: [["guard_height", "Hands vs shoulders"]], unit: "torso" },
  { key: "lean", label: "Torso lean", series: [["torso_lean_deg", "Lean"]], unit: "°" },
  { key: "posture", label: "Posture change", series: [["posture_change", "Δ pose"]], unit: "" },
];
// Two-series charts: left = neutral, right = accent (identity also carried by legend + tooltip labels)
const SERIES_COLORS = ["#a3a3ad", "#e11d2e"];

export function MovementChart({ videoId, tracking, frame, onSeek, jobId }: { videoId: string; tracking: Tracking; frame: number; onSeek: (f: number) => void; jobId: string }) {
  const candidates = useMemo(() => {
    const fighters = tracking.tracks.filter((t) => t.role === "fighter");
    const rest = tracking.tracks.filter((t) => t.role !== "fighter" && t.status !== "fragment");
    return [...fighters, ...rest];
  }, [tracking]);
  const [trackId, setTrackId] = useState<number | null>(null);
  const [metric, setMetric] = useState("wrist");
  const [rows, setRows] = useState<MovementRow[]>([]);
  const [loading, setLoading] = useState(false);
  const tid = trackId ?? candidates[0]?.track_id ?? null;

  useEffect(() => {
    if (tid == null) return;
    let alive = true;
    setLoading(true);
    visionApi.movement(videoId, tid).then((d) => alive && setRows(d.items)).catch(() => alive && setRows([])).finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [videoId, tid, jobId]);

  const m = METRICS.find((x) => x.key === metric)!;
  const t: Track | undefined = tracking.tracks.find((x) => x.track_id === tid);
  const states = useMemo(() => {
    const c: Record<string, number> = {};
    rows.forEach((r) => (c[r.state] = (c[r.state] ?? 0) + 1));
    return c;
  }, [rows]);

  if (!candidates.length) return null;
  return (
    <section className="panel">
      <div className="panel-header flex-wrap">
        <h2 className="label-caps">Movement · ID {tid}{t?.assigned_fighter ? ` · ${t.assigned_fighter.name}` : ""}</h2>
        <div className="flex gap-2">
          <select className="input w-auto py-1 text-xs" value={tid ?? ""} onChange={(e) => setTrackId(Number(e.target.value))} aria-label="Track">
            {candidates.map((c) => <option key={c.track_id} value={c.track_id}>ID {c.track_id}{c.assigned_fighter ? ` · ${c.assigned_fighter.name}` : c.role !== "unreviewed" ? ` · ${c.role}` : ""}</option>)}
          </select>
          <select className="input w-auto py-1 text-xs" value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="Metric">
            {METRICS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
          </select>
        </div>
      </div>
      <div className="h-56 px-2 pb-2 pt-3">
        {loading && !rows.length ? (
          <div className="grid h-full place-items-center text-xs text-ink-500"><Spinner /></div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 4, right: 12, left: -8, bottom: 0 }}
              onClick={(s) => { const i = s?.activeTooltipIndex; if (i != null && rows[Number(i)]) onSeek(rows[Number(i)].frame_number); }}>
              <CartesianGrid vertical={false} stroke="#26262d" />
              <XAxis dataKey="frame_number" type="number" domain={["dataMin", "dataMax"]} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false}
                axisLine={{ stroke: "#33333c" }} tickFormatter={(f: number) => formatTimestamp(f / (tracking.fps || 30)).replace(/\.\d+$/, "")} />
              <YAxis tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
              <Tooltip
                cursor={{ stroke: "#4b4b56" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const r = payload[0].payload as MovementRow;
                  return (
                    <div className="rounded-md border border-ink-600 bg-ink-800 px-3 py-2 text-xs shadow-lg">
                      <div className="text-ink-400">{formatTimestamp(r.timestamp)} · frame {r.frame_number} · {r.state.replace("_", " ")}</div>
                      {m.series.map(([k, label]) => (
                        <div key={k} className="tabular text-white">{label}: {r[k] == null ? "— (missing)" : `${(r[k] as number).toFixed(2)} ${m.unit}`}</div>
                      ))}
                    </div>
                  );
                }}
              />
              {m.series.length > 1 && <Legend wrapperStyle={{ fontSize: 11, color: "#a3a3ad" }} iconType="plainline" />}
              <ReferenceLine x={frame - (frame % Math.max(tracking.stride, 1))} stroke="#ffffff" strokeOpacity={0.5} />
              {m.series.map(([k, label], i) => (
                <Line key={k} dataKey={k as string} name={label} stroke={m.series.length > 1 ? SERIES_COLORS[i] : "#e11d2e"} strokeWidth={2} dot={false}
                  connectNulls={false} isAnimationActive={false} activeDot={{ r: 4, stroke: "#131317", strokeWidth: 2 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="flex flex-wrap gap-3 border-t border-ink-700/70 px-4 py-2 text-[11px] text-ink-400">
        {(["moving", "stationary", "low_confidence", "missing"] as const).map((s) => (
          <span key={s}><span className={STATE_STYLE[s]}>●</span> {s.replace("_", " ")} <b className="font-medium text-ink-200 tabular">{states[s] ?? 0}</b></span>
        ))}
        <span className="ml-auto text-ink-500">Gaps in the line = landmark missing or low confidence (not zero movement)</span>
      </div>
    </section>
  );
}

export function JobHistoryList({ status, onRetry }: { status: VisionStatus; onRetry: (id: string) => void }) {
  if (!status.history.length) return null;
  return (
    <section className="panel">
      <div className="panel-header"><h2 className="label-caps">Analysis history</h2></div>
      <ul className="divide-y divide-ink-700/50 text-xs">
        {status.history.map((j) => (
          <li key={j.id} className="flex items-center gap-3 px-4 py-2.5">
            <span className="w-16 text-ink-400">Attempt {j.attempt}</span>
            <StatusBadge status={j.status} />
            <span className="flex-1 truncate text-ink-500">
              {formatDate(j.created_at)}{j.duration_seconds != null && ` · ${j.duration_seconds.toFixed(0)}s`}
              {j.stats?.superseded_by && " · superseded"}
              {j.id === status.current?.id && <span className="text-emerald-300"> · current</span>}
              {j.error_message && <span className="text-blood-300"> · {j.error_message}</span>}
            </span>
            {(j.status === "failed" || j.status === "cancelled") && !status.active_job && (
              <button className="btn-ghost px-2 py-1 text-xs" onClick={() => onRetry(j.id)}><RotateCw className="size-3" />Retry</button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
