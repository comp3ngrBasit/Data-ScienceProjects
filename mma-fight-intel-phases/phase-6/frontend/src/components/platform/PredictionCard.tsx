import { useState } from "react";
import { TriangleAlert } from "lucide-react";
import type { FighterPredictions } from "../../api/platform";
import { cn } from "../../lib/format";
import { DistBars, LevelBadge, ReliabilityBlock } from "../predictions/parts";
import { pct } from "./FightDashboard";

/** Prediction panel: previous actions, context, top 3, reliability, an interactive detected/predicted timeline and evidence. */
export function PredictionCard({ data, onSeek, compact }: { data: FighterPredictions; onSeek?: (t: number) => void; compact?: boolean }) {
  const [sel, setSel] = useState<string | null>(null);
  if (!data.available || !data.prediction) {
    return (
      <div className="flex items-start gap-2 px-5 py-5 text-sm text-amber-200"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-400" />
        <div>{data.reason ?? "Prediction unavailable."}<div className="mt-1 text-xs text-ink-400">Held-out evaluated predictions for this fighter: {data.performance.held_out_evaluated}</div></div>
      </div>
    );
  }
  const p = data.prediction;
  const ins = p.reliability.level === "insufficient" || p.reliability.level === "low";
  const evidenceFor = (label: string) => p.evidence.supporting_combinations.filter((c) => c.sequence[c.sequence.length - 1] === label);
  const fmt = (t: number | null) => (t == null ? "—" : `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(2).padStart(5, "0")}`);
  return (
    <div className="space-y-4 px-5 py-4">
      {ins && (
        <div className="flex gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">
          <TriangleAlert className="size-4 shrink-0 text-amber-400" />
          {p.reliability.level === "insufficient" ? "Insufficient data: too few matching historical observations for a reliable prediction." : "Low reliability — see warnings."} The values below are model outputs, not facts.
        </div>
      )}
      <div className={cn("grid gap-4", !compact && "lg:grid-cols-2")}>
        <div>
          <div className="label-caps mb-2">Interactive timeline</div>
          <ol className="space-y-1">
            {data.timeline?.map((x, i) => (
              <li key={i}>
                <button onClick={() => (x.kind === "detected" ? x.time != null && onSeek?.(x.time) : setSel(sel === x.label ? null : x.label))}
                  className={cn("flex w-full items-center gap-3 rounded px-2 py-1.5 text-left text-sm",
                    x.kind === "detected" ? "bg-ink-750 text-ink-100 hover:bg-ink-700" : "border border-dashed hover:bg-ink-800",
                    x.kind === "predicted" ? "border-blood-400 text-blood-100" : x.kind === "alternative" ? "border-ink-500 text-ink-300" : "",
                    sel === x.label && x.kind !== "detected" && "ring-1 ring-white/50")}
                  title={x.time_basis}>
                  <span className="w-16 font-mono text-xs text-ink-400">{fmt(x.time)}</span>
                  <span className="flex-1">{x.label}</span>
                  <span className="text-[11px] uppercase tracking-wide text-ink-500">{x.kind === "detected" ? "detected" : x.kind === "predicted" ? `predicted ${pct(x.probability)}` : `alternative ${pct(x.probability)}`}</span>
                </button>
              </li>
            ))}
          </ol>
          {!data.timeline?.some((x) => x.kind !== "detected") && <p className="mt-1 text-[11px] text-ink-500">Predicted times need this fighter's typical timing; not available for this model.</p>}
          <p className="mt-2 text-[11px] text-ink-500">Solid = detected in the footage. Dashed = model prediction at an estimated time (median gap after the last action). Click a prediction for its evidence.</p>
          {data.context && (
            <div className="mt-3 text-xs text-ink-400">
              Context: opponent's last action {data.context.opponent_label ? `${data.context.opponent_label} (${data.context.opponent_lag?.toFixed(2)} s before)` : "unknown"} · distance {data.context.distance != null ? `${data.context.distance.toFixed(2)} torso lengths` : "unknown"} · stance {data.context.stance ?? "unknown"}
            </div>
          )}
        </div>
        <div>
          <div className="mb-2 flex items-center justify-between"><span className="label-caps">Top three predicted next actions</span><LevelBadge level={p.reliability.level} /></div>
          <DistBars items={p.top3} />
          <p className="mt-2 text-[11px] text-ink-500">Full distribution sums to {pct(p.probability_sum)} over {p.distribution.length} classes · model {p.model.version}{p.in_training_data ? " · this fight was in the training data (in-sample)" : ""}</p>
          {sel && (
            <div className="mt-3 rounded-md bg-ink-800 px-3 py-2 text-xs">
              <div className="label-caps mb-1">Evidence for “{sel}”</div>
              {evidenceFor(sel).length ? evidenceFor(sel).map((c, i) => <div key={i} className="flex justify-between gap-2"><span className="text-ink-200">{c.sequence.join(" → ")}</span><span className="tabular text-ink-400">{c.count}/{c.of} · {c.fights} fight(s) · {c.scope}</span></div>)
                : <span className="text-ink-500">No identical historical sequence ends with this action; the probability comes from the model's generalisation.</span>}
            </div>
          )}
          {data.next_sequences && data.next_sequences.length > 0 && (
            <div className="mt-3 text-xs"><div className="label-caps mb-1">Possible next two actions</div>
              {data.next_sequences.slice(0, 3).map((s, i) => <div key={i} className="flex justify-between"><span className="text-ink-300">{s.labels.join(" → ")}</span><span className="tabular text-ink-500">{pct(s.probability, 1)}</span></div>)}
            </div>
          )}
        </div>
      </div>
      <details><summary className="cursor-pointer text-xs text-ink-400">Reliability details & supporting observations</summary><ReliabilityBlock p={p} /></details>
      <p className="text-[11px] text-ink-500">{data.disclaimer}</p>
    </div>
  );
}
