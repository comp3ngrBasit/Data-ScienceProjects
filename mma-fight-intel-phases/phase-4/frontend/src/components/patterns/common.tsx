import { useState, type ReactNode } from "react";
import { CircleAlert, Database, TriangleAlert } from "lucide-react";
import { RELIABILITY, type CI, type Markov, type Meta, type Reliability } from "../../api/patterns";
import { cn } from "../../lib/format";

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 2) => (v == null ? "—" : v.toFixed(d));
export const ci = (c: CI) => (c ? `${pct(c[0])}–${pct(c[1])}` : "—");

export function ReliabilityBadge({ r, className }: { r: Reliability; className?: string }) {
  const m = RELIABILITY[r] ?? RELIABILITY.insufficient;
  return <span title={m.hint} className={cn("inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-inset", m.cls, className)}>{m.label}</span>;
}

export function Warnings({ items }: { items: string[] }) {
  if (!items?.length) return null;
  return (
    <div className="rounded-md border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-200">
      {items.map((w) => <div key={w} className="flex gap-2"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-400" />{w}</div>)}
    </div>
  );
}

export function MetaLine({ meta }: { meta: Meta }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-500">
      <span className="flex items-center gap-1"><Database className="size-3" />{meta.mode === "snapshot" ? `Stored analysis · ${new Date(meta.snapshot_at!).toLocaleString()}` : "Computed live from current data"}</span>
      {meta.stale && <span className="text-amber-400">Stored analysis is out of date — re-run analysis to refresh the saved tables</span>}
      <span className="font-mono">dataset {meta.dataset_version}</span>
    </div>
  );
}

export function Panel({ title, right, children, className }: { title: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("panel", className)}>
      <div className="panel-header flex items-center justify-between gap-3"><h3 className="label-caps">{title}</h3>{right}</div>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="flex items-center justify-center gap-2 px-5 py-8 text-center text-sm text-ink-500"><CircleAlert className="size-4" />{children}</p>;
}

export function Seq({ labels, className }: { labels: string[]; className?: string }) {
  return (
    <span className={cn("inline-flex flex-wrap items-center gap-1", className)}>
      {labels.map((l, i) => (
        <span key={i} className="inline-flex items-center gap-1">
          {i > 0 && <span className="text-ink-500">→</span>}
          <span className="rounded bg-ink-750 px-1.5 py-0.5 text-xs text-ink-100 ring-1 ring-ink-600">{l}</span>
        </span>
      ))}
    </span>
  );
}

/** Share bar with a 95% CI whisker. Single hue; text stays in ink tokens. */
export function ShareBar({ value, interval, max = 1 }: { value: number; interval?: CI; max?: number }) {
  const w = (v: number) => `${Math.max(0, Math.min(1, v / max)) * 100}%`;
  return (
    <div className="relative h-2.5 w-full rounded-full bg-ink-750">
      <div className="absolute inset-y-0 left-0 rounded-full bg-blood-500" style={{ width: w(value) }} />
      {interval && <div className="absolute top-1/2 h-px -translate-y-1/2 bg-ink-200/70" style={{ left: w(interval[0]), width: `calc(${w(interval[1])} - ${w(interval[0])})` }} />}
    </div>
  );
}

/** Interactive transition heatmap: rows = previous action(s), columns = next action. */
export function Heatmap({ m, mode }: { m: Markov; mode: "raw" | "smoothed" }) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  if (!m.rows.length) return <Empty>No transitions observed with the current settings and filters.</Empty>;
  const hr = hover ? m.rows[hover.r] : null;
  const hc = hr && hover ? hr.cells[hover.c] : null;
  return (
    <div className="p-4">
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-0.5 text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-ink-850 px-2 text-left align-bottom text-[10px] font-medium uppercase tracking-wide text-ink-500">Previous ↓ · Next →</th>
              <th className="px-1 align-bottom text-[10px] font-medium text-ink-500">n</th>
              {m.column_labels.map((l, c) => (
                <th key={c} className={cn("h-28 w-10 align-bottom font-normal", hover?.c === c ? "text-white" : "text-ink-400")}>
                  <div className="mx-auto w-4 whitespace-nowrap [writing-mode:vertical-rl] rotate-180 text-left">{l}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {m.rows.map((row, r) => (
              <tr key={row.context.join(">")}>
                <td className={cn("sticky left-0 z-10 max-w-64 bg-ink-850 px-2 py-1 whitespace-nowrap", hover?.r === r ? "text-white" : "text-ink-200")}>
                  {row.context_labels.join(" → ")} <ReliabilityBadge r={row.reliability} className="ml-1" />
                </td>
                <td className="px-1 text-right text-ink-400 tabular">{row.n}</td>
                {row.cells.map((cell, c) => {
                  const v = mode === "raw" ? cell.raw : cell.smoothed;
                  return (
                    <td key={c} onMouseEnter={() => setHover({ r, c })} onMouseLeave={() => setHover(null)}
                      className={cn("h-9 w-10 cursor-default rounded text-center tabular", hover?.r === r && hover?.c === c && "ring-2 ring-white/70")}
                      style={{ background: cell.count || mode === "smoothed" ? `rgba(225,29,46,${0.08 + 0.92 * v})` : "#18181d", color: v > 0.45 ? "#fff" : "#a3a3ad" }}>
                      {cell.count ? Math.round(v * 100) : mode === "smoothed" ? <span className="text-ink-600">{Math.round(v * 100)}</span> : ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex min-h-12 flex-wrap items-center justify-between gap-3 rounded-md bg-ink-800 px-3 py-2 text-xs">
        {hr && hc ? (
          <div className="text-ink-200">
            <span className="text-white">{hr.context_labels.join(" → ")}</span> → <span className="text-white">{m.column_labels[hover!.c]}</span>:{" "}
            <span className="tabular">{hc.count} of {hr.n}</span> · raw <span className="tabular">{pct(hc.raw, 1)}</span> (95% CI {ci(hc.ci95)}) · smoothed <span className="tabular">{pct(hc.smoothed, 1)}</span>
          </div>
        ) : <span className="text-ink-500">Hover a cell for counts, raw and smoothed probability and the 95% Wilson interval. Values are percentages of the row.</span>}
        <div className="flex items-center gap-2 text-ink-500">
          <span>0%</span><div className="h-2 w-24 rounded" style={{ background: "linear-gradient(90deg, rgba(225,29,46,0.08), rgba(225,29,46,1))" }} /><span>100%</span>
        </div>
      </div>
      <p className="mt-2 text-[11px] text-ink-500">
        {m.observations} transitions · smoothing: {m.smoothing.method}, α = {m.smoothing.alpha} (shown when “smoothed” is selected; unobserved cells get small non-zero values) · entropy rate {m.entropy_rate_bits} bits
        {m.insufficient_contexts ? ` · ${m.insufficient_contexts} contexts hidden (fewer observations than the minimum)` : ""}
      </p>
    </div>
  );
}
