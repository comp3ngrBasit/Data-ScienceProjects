import { Link } from "react-router-dom";
import { BrainCircuit, Gauge, ListChecks, Sparkles } from "lucide-react";
import { ARCH_LABEL, predictionsApi } from "../../api/predictions";
import { useData } from "../../hooks/useData";
import { StatCard } from "../ui";

export default function PredictionOverviewSection() {
  const { data: o } = useData(predictionsApi.overview, [], 20000);
  if (!o) return null;
  const m = o.active_model;
  return (
    <section className="mt-8">
      <div className="mb-3 flex items-end justify-between">
        <div>
          <h2 className="font-display text-2xl font-bold uppercase tracking-wide text-white">AI Next-Move Prediction</h2>
          <p className="text-sm text-ink-400">Calibrated next-action predictions evaluated on unseen fights (Phase 5)</p>
        </div>
        <Link to="/predictions" className="btn-secondary"><Sparkles className="size-4" />Open</Link>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Active model" value={m ? m.name : "None"} icon={<BrainCircuit className="size-4" />} hint={m ? `${ARCH_LABEL[m.architecture]}${m.experimental ? " · experimental" : ""}` : "train models first"} accent />
        <StatCard label="Held-out top-1 (CV)" value={m?.summary.top1_accuracy != null ? `${Math.round(m.summary.top1_accuracy * 100)}%` : "—"} icon={<Gauge className="size-4" />} hint={m ? `top-3 ${Math.round((m.summary.top3_accuracy ?? 0) * 100)}% · n=${m.summary.n}` : ""} />
        <StatCard label="Stored predictions" value={o.predictions} icon={<ListChecks className="size-4" />} hint={`${o.evaluated} compared with the actual next action`} />
        <StatCard label="Held-out evaluated" value={o.held_out.n} icon={<Gauge className="size-4" />} hint={o.held_out.n ? `top-1 ${o.held_out.top1} · top-3 ${o.held_out.top3}` : "predictions on fights the model never saw"} />
      </div>
    </section>
  );
}
