import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { CircleAlert, TriangleAlert } from "lucide-react";
import { ARCH_LABEL, LEVEL, type DistItem, type Level, type Metrics, type Prediction, type Report } from "../../api/predictions";
import { cn } from "../../lib/format";

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
export const ciTxt = (c: [number, number] | null | undefined) => (c ? `${pct(c[0])}–${pct(c[1])}` : "");
const tip = { contentStyle: { background: "#18181d", border: "1px solid #33333c", borderRadius: 6, fontSize: 12 }, labelStyle: { color: "#d6d6dc" }, itemStyle: { color: "#f1f1f4" }, cursor: { fill: "rgba(255,255,255,0.04)" } };

export function LevelBadge({ level }: { level: Level }) {
  const m = LEVEL[level];
  return <span className={cn("inline-flex rounded px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-inset", m.cls)}>{m.label}</span>;
}

export function Panel({ title, right, children, className }: { title: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={cn("panel", className)}>
      <div className="panel-header flex items-center justify-between gap-3"><h3 className="label-caps">{title}</h3>{right}</div>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="flex items-center justify-center gap-2 px-5 py-8 text-center text-sm text-ink-500"><CircleAlert className="size-4 shrink-0" />{children}</p>;
}

export function Warn({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="space-y-1 rounded-md border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-200">
      {items.map((w) => <div key={w} className="flex gap-2"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-400" />{w}</div>)}
    </div>
  );
}

/** Probability bars for the whole distribution (top three highlighted). */
export function DistBars({ items, highlight = 3 }: { items: DistItem[]; highlight?: number }) {
  return (
    <div className="space-y-2">
      {items.map((d, i) => (
        <div key={d.action} className="grid grid-cols-[1.5rem_minmax(0,11rem)_1fr_3.5rem] items-center gap-3 text-sm">
          <span className={cn("text-right font-display text-lg tabular", i < highlight ? "text-white" : "text-ink-500")}>{i + 1}</span>
          <span className={cn("truncate", i < highlight ? "text-ink-100" : "text-ink-400")}>{d.label}</span>
          <div className="h-2.5 rounded-full bg-ink-750"><div className={cn("h-full rounded-full", i < highlight ? "bg-blood-500" : "bg-ink-500")} style={{ width: `${Math.max(1, d.probability * 100)}%` }} /></div>
          <span className="text-right text-xs tabular text-ink-300">{pct(d.probability, 1)}</span>
        </div>
      ))}
    </div>
  );
}

export function ReliabilityBlock({ p }: { p: Prediction }) {
  const r = p.reliability;
  const sup = r.supporting_observations;
  return (
    <div className="space-y-3 px-5 py-4 text-sm">
      <div className="flex flex-wrap items-center gap-2"><LevelBadge level={r.level} />{r.experimental_model && <span className="text-xs text-amber-300">Experimental model</span>}</div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        <dt className="text-ink-400">Predicted probability (calibrated)</dt><dd className="text-right tabular text-white">{pct(r.predicted_probability, 1)}</dd>
        <dt className="text-ink-400">Model confidence before calibration</dt><dd className="text-right tabular text-ink-200">{pct(r.uncalibrated_model_confidence, 1)}</dd>
        <dt className="text-ink-400">Held-out accuracy at this confidence</dt><dd className="text-right tabular text-ink-200">{r.historical_accuracy_at_this_confidence ? `${pct(r.historical_accuracy_at_this_confidence.observed_accuracy)} (n=${r.historical_accuracy_at_this_confidence.n})` : "—"}</dd>
        <dt className="text-ink-400">Uncertainty (normalised entropy)</dt><dd className="text-right tabular text-ink-200">{num(r.uncertainty_normalised_entropy, 2)}</dd>
        <dt className="text-ink-400">Matching history — last action</dt><dd className="text-right tabular text-ink-200">{sup.fighter_order1 ?? 0} fighter · {sup.all_order1 ?? 0} all</dd>
        <dt className="text-ink-400">Matching history — last 2 actions</dt><dd className="text-right tabular text-ink-200">{sup.fighter_order2 ?? "—"} fighter · {sup.all_order2 ?? "—"} all</dd>
        <dt className="text-ink-400">Training set</dt><dd className="text-right tabular text-ink-200">{r.training_examples ?? "—"} examples · {r.training_fights ?? "—"} fights</dd>
        <dt className="text-ink-400">Calibration (CV)</dt><dd className="text-right tabular text-ink-200">T={num(r.calibration.temperature, 2)} · ECE {num(r.calibration.cv_ece)} · Brier {num(r.calibration.cv_brier)}</dd>
        <dt className="text-ink-400">CV top-1 / top-3 accuracy</dt><dd className="text-right tabular text-ink-200">{pct(r.cv_top1_accuracy)} / {pct(r.cv_top3_accuracy)}</dd>
      </dl>
      <p className="text-[11px] text-ink-500">{r.explanation}</p>
      <Warn items={r.warnings} />
    </div>
  );
}

export function MetricsTable({ models, selected }: { models: Record<string, Metrics>; selected?: string }) {
  const rows = Object.entries(models).filter(([, m]) => m && m.n);
  if (!rows.length) return <Empty>No evaluation under this protocol.</Empty>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-[11px] uppercase tracking-wide text-ink-500">
          <tr className="border-b border-ink-700">
            <th className="px-5 py-2">Model</th><th className="px-2">n</th><th className="px-2">Top-1</th><th className="px-2">Top-3</th><th className="px-2">Macro P</th><th className="px-2">Macro R</th><th className="px-2">Macro F1</th><th className="px-2">Log loss</th><th className="px-2">Brier</th><th className="px-2">ECE</th><th className="px-5">vs Markov (top-1)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([a, m]) => (
            <tr key={a} className={cn("border-b border-ink-700/50", a === selected && "bg-blood-500/5")}>
              <td className="px-5 py-2 whitespace-nowrap text-ink-100">{ARCH_LABEL[a] ?? a}{a === selected && <span className="ml-2 text-[10px] uppercase text-blood-300">selected</span>}</td>
              <td className="px-2 tabular text-ink-400">{m.n}</td>
              <td className="px-2 tabular whitespace-nowrap">{pct(m.top1_accuracy, 1)} <span className="text-[10px] text-ink-500">{ciTxt(m.top1_ci95)}</span></td>
              <td className="px-2 tabular">{pct(m.top3_accuracy, 1)}</td>
              <td className="px-2 tabular">{num(m.macro_precision, 2)}</td><td className="px-2 tabular">{num(m.macro_recall, 2)}</td><td className="px-2 tabular">{num(m.macro_f1, 2)}</td>
              <td className="px-2 tabular">{num(m.log_loss)}</td><td className="px-2 tabular">{num(m.brier)}</td><td className="px-2 tabular">{num(m.ece)}</td>
              <td className="px-5 text-xs tabular whitespace-nowrap">{m.vs_markov_top1 ? <span className={m.vs_markov_top1.significant ? (m.vs_markov_top1.difference > 0 ? "text-emerald-300" : "text-blood-300") : "text-ink-400"}>{m.vs_markov_top1.difference > 0 ? "+" : ""}{pct(m.vs_markov_top1.difference, 1)} ({pct(m.vs_markov_top1.ci95[0], 1)} to {pct(m.vs_markov_top1.ci95[1], 1)}){m.vs_markov_top1.significant ? "" : " · n.s."}</span> : a === "markov" ? "baseline" : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const BREAKDOWN_LABEL: Record<string, string> = {
  known_fighters: "Known fighters", unseen_fighters: "Unseen fighters", offensive_targets: "Offensive next actions", defensive_targets: "Defensive next actions",
  with_opponent_context: "With opponent action", without_opponent_context: "Without opponent action",
};

export function BreakdownTable({ m }: { m: Metrics }) {
  const rows = Object.entries(m.breakdown ?? {});
  return (
    <table className="w-full text-left text-xs">
      <thead className="text-ink-500"><tr><th className="py-1">Subset</th><th>n</th><th>Top-1</th><th>Top-3</th><th>Log loss</th></tr></thead>
      <tbody className="text-ink-200">
        {rows.map(([k, v]) => (
          <tr key={k} className="border-t border-ink-700/50"><td className="py-1">{BREAKDOWN_LABEL[k] ?? k.replace("style_", "Style: ").replace(/_/g, " ")}</td>
            <td className="tabular">{v?.n ?? 0}</td><td className="tabular">{v ? pct(v.top1_accuracy, 1) : "—"}</td><td className="tabular">{v ? pct(v.top3_accuracy, 1) : "—"}</td><td className="tabular">{v ? num(v.log_loss) : "—"}</td></tr>
        ))}
      </tbody>
    </table>
  );
}

export function CalibrationChart({ m }: { m: Metrics }) {
  const pts = m.reliability_diagram.map((b) => ({ x: b.mean_confidence, y: b.accuracy, n: b.n }));
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ left: -10, right: 12, top: 8, bottom: 4 }}>
          <CartesianGrid stroke="#26262d" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fill: "#75757f", fontSize: 11 }} name="Mean confidence" />
          <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fill: "#75757f", fontSize: 11 }} name="Observed accuracy" />
          <ZAxis type="number" dataKey="n" range={[40, 300]} name="Predictions" />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#55555f" strokeDasharray="4 4" />
          <Tooltip {...tip} formatter={(v, n) => (n === "Predictions" ? v : `${(Number(v) * 100).toFixed(1)}%`)} />
          <Scatter data={pts} fill="#e11d2e" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Confusion({ m, labels }: { m: Metrics; labels: string[] }) {
  const cm = m.confusion_matrix;
  if (!cm) return null;
  const max = Math.max(1, ...cm.matrix.flat());
  return (
    <div className="overflow-x-auto p-4">
      <table className="border-separate border-spacing-0.5 text-[11px]">
        <thead><tr><th className="px-2 text-left font-normal text-ink-500">Actual ↓ · Predicted →</th>{labels.map((l) => <th key={l} className="h-24 w-10 align-bottom font-normal text-ink-400"><div className="mx-auto w-4 rotate-180 whitespace-nowrap [writing-mode:vertical-rl]">{l}</div></th>)}</tr></thead>
        <tbody>
          {cm.matrix.map((row, i) => (
            <tr key={i}><td className="whitespace-nowrap px-2 text-ink-300">{labels[i]}</td>
              {row.map((v, j) => <td key={j} title={`${labels[i]} → ${labels[j]}: ${v}`} className="h-8 w-10 rounded text-center tabular" style={{ background: v ? `rgba(225,29,46,${0.12 + 0.88 * v / max})` : "#18181d", color: v / max > 0.5 ? "#fff" : "#a3a3ad", outline: i === j ? "1px solid #55555f" : undefined }}>{v || ""}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ReportView({ report }: { report: Report }) {
  const [proto, setProto] = useState("fight_cv");
  const [arch, setArch] = useState(report.selected ?? "markov");
  const p = report.protocols[proto];
  const m = p?.models?.[arch];
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        {[["Training examples", report.data.examples], ["Fights", report.data.fights], ["Fighters", report.data.fighters], ["Classes", report.data.classes.length]].map(([l, v]) => (
          <div key={l as string} className="panel px-4 py-3"><div className="label-caps">{l}</div><div className="mt-1 font-display text-3xl font-semibold text-white tabular">{v}</div></div>
        ))}
      </div>
      {report.experimental && <Warn items={["Experimental: the training data is below the reliability thresholds (examples / fights). Treat every number as indicative only."]} />}
      {Object.keys(report.skipped).length > 0 && <div className="text-xs text-ink-400">Not trained: {Object.entries(report.skipped).map(([a, why]) => `${ARCH_LABEL[a] ?? a} — ${why}`).join(" · ")}</div>}
      <Panel title="Model comparison" right={
        <select className="input h-8 py-0 text-xs" value={proto} onChange={(e) => setProto(e.target.value)}>
          {Object.entries(report.protocols).map(([k, v]) => <option key={k} value={k} disabled={!Object.keys(v.models).length}>{k === "fight_cv" ? "Unseen fights (CV)" : k === "fighter_cv" ? "Unseen fighters (CV)" : "Temporal hold-out"}</option>)}
        </select>}>
        <p className="px-5 pt-3 text-xs text-ink-400">{p?.description}. {report.selection_rule && `Selected: ${ARCH_LABEL[report.selected!]} — ${report.selection_rule}.`}</p>
        <MetricsTable models={p?.models ?? {}} selected={report.selected} />
      </Panel>
      {m && m.n > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Details" right={<select className="input h-8 py-0 text-xs" value={arch} onChange={(e) => setArch(e.target.value)}>{Object.keys(p.models).map((a) => <option key={a} value={a}>{ARCH_LABEL[a] ?? a}</option>)}</select>}>
            <div className="px-5 py-3"><div className="label-caps mb-1">Calibration (held-out)</div><CalibrationChart m={m} /><p className="text-[11px] text-ink-500">Dots on the dashed line = probabilities match observed frequencies. Dot size = number of predictions.</p></div>
            <div className="px-5 pb-4"><div className="label-caps mb-1">Performance by subset</div><BreakdownTable m={m} /></div>
          </Panel>
          <Panel title="Confusion matrix (held-out)"><Confusion m={m} labels={report.data.class_labels} /></Panel>
        </div>
      )}
      <Panel title="Training data">
        <div className="h-56 px-2 py-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.entries(report.data.target_counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ action: k, count: v }))} margin={{ left: -10, right: 8 }}>
              <CartesianGrid vertical={false} stroke="#26262d" />
              <XAxis dataKey="action" tick={{ fill: "#a3a3ad", fontSize: 10 }} interval={0} tickLine={false} />
              <YAxis tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip {...tip} />
              <Bar dataKey="count" name="Next-action examples" fill="#e11d2e" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="px-5 pb-3 text-[11px] text-ink-500">Sources: {Object.entries(report.data.sources).map(([k, v]) => `${v} ${k}`).join(" · ")}{report.data.rare_actions.length ? ` · grouped as “Other (rare)”: ${report.data.rare_actions.join(", ")}` : ""}</p>
      </Panel>
    </div>
  );
}

export function OverTime({ data }: { data: { i: number; top1_rate: number; top3_rate: number }[] }) {
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ left: -10, right: 12, top: 8 }}>
          <CartesianGrid vertical={false} stroke="#26262d" />
          <XAxis dataKey="i" tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} />
          <Tooltip {...tip} formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} labelFormatter={(l) => `After ${l} evaluated predictions`} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line dataKey="top1_rate" name="Cumulative top-1" stroke="#e11d2e" strokeWidth={2} dot={false} />
          <Line dataKey="top3_rate" name="Cumulative top-3" stroke="#38bdf8" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
