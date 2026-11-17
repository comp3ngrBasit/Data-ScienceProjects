import { useMemo } from "react";
import { GROUP_COLORS, GROUP_LABELS, type ActionGroup, type Category, type Lane } from "../../api/actions";
import { cn } from "../../lib/format";
import { Marker, fmtT, pct } from "./common";

/** Per-fighter timeline: detected actions (top row) and manual annotations (bottom row). Click to seek. */
export default function ActionTimeline({
  lanes,
  duration,
  time,
  categories,
  onSeek,
  showAnnotations = true,
}: {
  lanes: Lane[];
  duration: number;
  time: number;
  categories: Category[];
  onSeek: (t: number) => void;
  showAnnotations?: boolean;
}) {
  const fam = useMemo(() => Object.fromEntries(categories.map((c) => [c.key, c.family])), [categories]);
  const D = Math.max(duration, 0.1);
  const pos = (t: number) => `${Math.min(100, Math.max(0, (t / D) * 100))}%`;
  const click = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * D);
  };

  return (
    <section className="panel">
      <div className="panel-header flex-wrap">
        <h2 className="label-caps">Detected action timeline</h2>
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-400">
          {(Object.keys(GROUP_LABELS) as ActionGroup[]).map((g) => (
            <span key={g} className="flex items-center gap-1"><span className="size-2 rounded-full" style={{ background: GROUP_COLORS[g] }} />{GROUP_LABELS[g]}</span>
          ))}
          <span className="flex items-center gap-1"><Marker family="punch" group="offensive" size={10} />punch</span>
          <span className="flex items-center gap-1"><Marker family="kick" group="offensive" size={10} />kick</span>
          <span className="flex items-center gap-1"><Marker family="knee_elbow" group="offensive" size={10} />knee/elbow</span>
          <span className="flex items-center gap-1"><Marker family="grappling" group="offensive" size={10} />grappling</span>
          <span className="flex items-center gap-1"><Marker family="defense" group="defensive" size={10} />defence</span>
        </div>
      </div>
      <div className="px-4 py-3">
        {lanes.length === 0 && <p className="py-6 text-center text-xs text-ink-500">No actions or annotations yet.</p>}
        {lanes.map((lane) => (
          <div key={lane.key} className="mb-3 last:mb-0">
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="font-medium text-ink-100">{lane.name}{lane.track_ids.length > 0 && <span className="ml-2 text-ink-500">IDs {lane.track_ids.join(", ")}</span>}</span>
              <span className="text-ink-500 tabular">{lane.actions.length} detected · {lane.annotations.length} annotated</span>
            </div>
            <div className="relative cursor-pointer select-none rounded bg-ink-900" onClick={click}>
              <div className="relative h-7 border-b border-ink-800">
                <span className="absolute left-1 top-1 text-[9px] uppercase tracking-wider text-ink-600">model</span>
                {lane.actions.map((a) => (
                  <button
                    key={a.id}
                    title={`${fmtT(a.start_time)} ${a.label} — confidence ${pct(a.confidence)}${a.low_confidence ? " (low)" : ""}${a.review_status !== "unreviewed" ? ` · ${a.review_status}` : ""}`}
                    onClick={(e) => { e.stopPropagation(); onSeek(a.start_time); }}
                    className={cn("absolute top-1 flex h-5 items-center rounded-sm border px-0.5 hover:z-10 hover:brightness-125",
                      a.low_confidence ? "border-dashed" : "border-solid", a.review_status === "rejected" && "opacity-30")}
                    style={{ left: pos(a.start_time), width: `max(14px, ${((a.end_time - a.start_time) / D) * 100}%)`,
                      borderColor: GROUP_COLORS[a.group], background: `${GROUP_COLORS[a.group]}${a.low_confidence ? "22" : "40"}` }}
                  >
                    <Marker family={fam[a.category] ?? "unknown"} group={a.group} size={11} />
                  </button>
                ))}
              </div>
              {showAnnotations && (
                <div className="relative h-6">
                  <span className="absolute left-1 top-1 text-[9px] uppercase tracking-wider text-ink-600">labels</span>
                  {lane.annotations.map((a) => (
                    <button
                      key={a.id}
                      title={`Annotation: ${a.label} ${fmtT(a.start_time)}–${fmtT(a.end_time)} (${a.status})`}
                      onClick={(e) => { e.stopPropagation(); onSeek(a.start_time); }}
                      className="absolute top-1 h-4 rounded-sm border border-white/50 hover:brightness-125"
                      style={{ left: pos(a.start_time), width: `max(6px, ${((a.end_time - a.start_time) / D) * 100}%)`, background: `${GROUP_COLORS[a.group]}aa` }}
                    />
                  ))}
                </div>
              )}
              <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/80" style={{ left: pos(time) }} />
            </div>
          </div>
        ))}
        <div className="mt-1 flex justify-between text-[10px] text-ink-500 tabular"><span>00:00</span><span>{fmtT(D)}</span></div>
      </div>
    </section>
  );
}
