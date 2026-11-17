import { TriangleAlert } from "lucide-react";
import { platformApi, type PatternEvalN } from "../api/platform";
import { pct } from "../components/platform/FightDashboard";
import { ReportView } from "../components/predictions/parts";
import { ErrorBlock, LoadingBlock, PageHeader } from "../components/ui";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const ci = (c: [number, number] | null | undefined) => (c ? ` (${pct(c[0])}–${pct(c[1])})` : "");

function Insufficient({ show, children }: { show: boolean; children: React.ReactNode }) {
  if (!show) return null;
  return <div className="flex gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-4 py-2.5 text-sm text-amber-200"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-400" />{children}</div>;
}

function Kpi({ label, value, hint }: { label: string; value: React.ReactNode; hint?: React.ReactNode }) {
  return <div className="panel px-4 py-3"><div className="label-caps">{label}</div><div className="mt-1 font-display text-3xl text-white tabular">{value}</div>{hint && <div className="text-[11px] text-ink-500">{hint}</div>}</div>;
}

function NRow({ n, r }: { n: string; r?: PatternEvalN }) {
  if (!r) return null;
  return (
    <tr className="border-t border-ink-700/50">
      <td className="px-5 py-1.5">{n}-action</td><td className="tabular">{r.detected}</td><td className="tabular">{r.ground_truth}</td><td className="tabular">{r.matched}</td>
      <td className="tabular">{pct(r.precision, 1)}<span className="text-[10px] text-ink-500">{ci(r.precision_ci95)}</span></td>
      <td className="tabular">{pct(r.recall, 1)}<span className="text-[10px] text-ink-500">{ci(r.recall_ci95)}</span></td>
      <td className="tabular">{r.f1?.toFixed(3) ?? "—"}</td><td className="tabular">{r.frequency_mae ?? "—"}</td>
    </tr>
  );
}

export default function ModelEvaluationPage() {
  const { data, error, reload } = useData(platformApi.evaluation, []);
  if (error) return <ErrorBlock message={error} onRetry={() => reload()} />;
  if (!data) return <LoadingBlock />;
  const a = data.action_recognition;
  const pe = data.pattern_recognition;
  const nm = data.next_move;
  const perClass = a ? (Array.isArray(a.metrics.per_class) ? a.metrics.per_class.map((r) => [r.class, r] as const) : Object.entries(a.metrics.per_class ?? {})) : [];
  const cm = a?.metrics.confusion_matrix as { labels: string[]; display_labels?: string[]; matrix: number[][] } | undefined;
  const max = cm ? Math.max(1, ...cm.matrix.flat()) : 1;
  return (
    <>
      <PageHeader title="Model Evaluation" subtitle={data.rules.splits} />
      <p className="mb-4 text-xs text-ink-400">{data.rules.never}</p>

      <h2 className="mb-3 font-display text-2xl font-bold uppercase tracking-wide text-white">Action recognition</h2>
      {!a ? <Insufficient show>No active action-recognition model. Annotate actions and train one (Model Management).</Insufficient> : (
        <div className="space-y-4">
          <Insufficient show={a.insufficient}>Only {a.metrics.n_evaluated} held-out examples — metrics are indicative only (wide confidence intervals).</Insufficient>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Kpi label="Accuracy" value={pct(a.metrics.accuracy, 1)} hint={ci(a.metrics.accuracy_ci95)} />
            <Kpi label="Precision (macro)" value={a.metrics.precision_macro?.toFixed(3)} />
            <Kpi label="Recall (macro)" value={a.metrics.recall_macro?.toFixed(3)} />
            <Kpi label="F1 (macro)" value={a.metrics.f1_macro?.toFixed(3)} />
            <Kpi label="Samples" value={a.metrics.n_evaluated} hint={`evaluated (held-out) · ${a.samples.final_train_samples ?? "—"} training windows · ${a.metrics.evaluation?.strategy ?? ""}`} />
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <section className="panel">
              <div className="panel-header"><h3 className="label-caps">Per-action performance</h3><span className="text-xs text-ink-500">model {a.model.name} ({a.model.architecture})</span></div>
              <table className="w-full text-left text-sm">
                <thead className="text-[11px] uppercase text-ink-500"><tr><th className="px-5 py-2">Action</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                <tbody>{perClass.map(([k, r]) => <tr key={k} className="border-t border-ink-700/50"><td className="px-5 py-1.5">{(r as { label?: string }).label ?? k}</td><td className="tabular">{r.precision?.toFixed(2)}</td><td className="tabular">{r.recall?.toFixed(2)}</td><td className="tabular">{r.f1?.toFixed(2)}</td><td className="tabular">{r.support}</td></tr>)}</tbody>
              </table>
            </section>
            {cm && (
              <section className="panel">
                <div className="panel-header"><h3 className="label-caps">Confusion matrix (held-out)</h3></div>
                <div className="overflow-x-auto p-4">
                  <table className="border-separate border-spacing-0.5 text-[11px]">
                    <thead><tr><th className="px-2 text-left font-normal text-ink-500">Actual ↓ · Predicted →</th>{(cm.display_labels ?? cm.labels).map((l) => <th key={l} className="h-24 w-10 align-bottom font-normal text-ink-400"><div className="mx-auto w-4 rotate-180 whitespace-nowrap [writing-mode:vertical-rl]">{l}</div></th>)}</tr></thead>
                    <tbody>{cm.matrix.map((row, i) => <tr key={i}><td className="whitespace-nowrap px-2 text-ink-300">{(cm.display_labels ?? cm.labels)[i]}</td>{row.map((v, j) => <td key={j} className="h-8 w-10 rounded text-center tabular" style={{ background: v ? `rgba(225,29,46,${0.12 + 0.88 * v / max})` : "#161c28", color: v / max > 0.5 ? "#fff" : "#a3a3ad", outline: i === j ? "1px solid #55555f" : undefined }}>{v || ""}</td>)}</tr>)}</tbody>
                  </table>
                </div>
              </section>
            )}
          </div>
        </div>
      )}

      <h2 className="mb-3 mt-8 font-display text-2xl font-bold uppercase tracking-wide text-white">Fighting pattern recognition</h2>
      <div className="grid gap-4 xl:grid-cols-3">
        <section className="panel xl:col-span-2">
          <div className="panel-header"><h3 className="label-caps">Combination detection & frequency estimation</h3><span className="text-xs text-ink-500">model detections vs verified annotations of the same videos</span></div>
          {pe.combination_detection.status !== "ok" ? <p className="px-5 py-5 text-sm text-amber-200">{pe.combination_detection.reason ?? "Insufficient data: no comparable combinations."} ({pe.combination_detection.videos} video(s) with both)</p> : (
            <table className="w-full text-left text-sm">
              <thead className="text-[11px] uppercase text-ink-500"><tr><th className="px-5 py-2">Length</th><th>Detected</th><th>Ground truth</th><th>Matched</th><th>Precision</th><th>Recall</th><th>F1</th><th title="Mean absolute difference between detected and true occurrence counts per fighter & combination">Freq. MAE</th></tr></thead>
              <tbody><NRow n="2" r={pe.combination_detection.n2} /><NRow n="3" r={pe.combination_detection.n3} /></tbody>
            </table>
          )}
        </section>
        <section className="panel">
          <div className="panel-header"><h3 className="label-caps">Transition reliability & coverage</h3></div>
          <div className="space-y-2 px-5 py-4 text-sm">
            {pe.transition_reliability.status === "ok" ? (
              <>
                <div className="flex justify-between"><span className="text-ink-400">Split-half probabilities compared</span><span className="tabular">{pe.transition_reliability.probabilities_compared}</span></div>
                <div className="flex justify-between"><span className="text-ink-400">Mean |P(half A) − P(half B)|</span><span className="tabular">{pe.transition_reliability.mean_absolute_difference}</span></div>
                <div className="flex justify-between"><span className="text-ink-400">Half-B estimates inside half-A 95% CI</span><span className="tabular">{pct(pe.transition_reliability.share_within_ci95)}</span></div>
              </>
            ) : <p className="text-amber-200">{pe.transition_reliability.reason}</p>}
            {pe.coverage.status === "ok" ? (
              <>
                <div className="flex justify-between border-t border-ink-700/60 pt-2"><span className="text-ink-400">Held-out transitions (leave-one-fight-out)</span><span className="tabular">{pe.coverage.held_out_transitions}</span></div>
                <div className="flex justify-between"><span className="text-ink-400">Previous action seen in other fights</span><span className="tabular">{pct(pe.coverage.previous_action_seen)}</span></div>
                <div className="flex justify-between"><span className="text-ink-400">Exact transition seen before</span><span className="tabular">{pct(pe.coverage.exact_transition_seen)}</span></div>
              </>
            ) : <p className="text-amber-200">{pe.coverage.reason}</p>}
            <p className="text-[11px] text-ink-500">{pe.data.fights} fights · {pe.data.sequences} sequences · {pe.data.actions} actions</p>
          </div>
        </section>
      </div>

      <h2 className="mb-3 mt-8 font-display text-2xl font-bold uppercase tracking-wide text-white">Next-move prediction</h2>
      {!nm ? <Insufficient show>No completed training run yet (Model Management → Train next-move models).</Insufficient> : nm.report.status !== "ok" ? <Insufficient show>{nm.report.reason}</Insufficient> : (
        <div className="space-y-4">
          <Insufficient show={nm.insufficient}>The dataset is below the reliability thresholds — results are experimental.</Insufficient>
          <section className="panel">
            <div className="panel-header"><h3 className="label-caps">Samples per protocol</h3><span className="text-xs text-ink-500">{nm.report.data.examples} examples · {nm.report.data.fights} unique fights · {nm.report.data.fighters} unique fighters · run {nm.finished_at ? new Date(nm.finished_at).toLocaleString() : ""}</span></div>
            <table className="w-full text-left text-sm">
              <thead className="text-[11px] uppercase text-ink-500"><tr><th className="px-5 py-2">Protocol</th><th>Folds</th><th>Training (mean)</th><th>Validation (mean)</th><th>Testing (total)</th></tr></thead>
              <tbody>{Object.entries(nm.samples).map(([k, s]) => <tr key={k} className="border-t border-ink-700/50"><td className="px-5 py-1.5">{k === "fight_cv" ? "Unseen fights (CV)" : k === "fighter_cv" ? "Unseen fighters (CV)" : "Temporal hold-out"}</td><td className="tabular">{s.folds}</td><td className="tabular">{s.mean_train_examples}</td><td className="tabular">{s.mean_validation_examples || "—"}</td><td className="tabular">{s.test_examples}</td></tr>)}</tbody>
            </table>
          </section>
          <section className="panel">
            <div className="panel-header"><h3 className="label-caps">Precision@3 / Recall@3</h3><span className="text-xs text-ink-500">one true next action per example: Recall@3 = top-3 hit rate, Precision@3 = hits ÷ 3</span></div>
            <div className="flex flex-wrap gap-6 px-5 py-3 text-sm">
              {Object.entries(nm.report.protocols.fight_cv?.models ?? {}).map(([k, m]) => {
                const mm = m as { precision_at_3?: number; recall_at_3?: number; top3_accuracy?: number };
                const r3 = mm.recall_at_3 ?? mm.top3_accuracy;  // identical by definition for one true label
                const p3 = mm.precision_at_3 ?? (r3 != null ? r3 / 3 : undefined);
                return <div key={k}><span className={cn("text-ink-300", k === nm.report.selected && "text-white")}>{k}</span> <span className="tabular text-ink-400">P@3 {p3?.toFixed(3) ?? "—"} · R@3 {r3?.toFixed(3) ?? "—"}</span></div>;
              })}
            </div>
          </section>
          <ReportView report={nm.report} />
        </div>
      )}
    </>
  );
}
