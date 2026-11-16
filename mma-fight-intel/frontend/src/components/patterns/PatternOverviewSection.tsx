import { Link } from "react-router-dom";
import { BrainCircuit, GitBranch, Network, Users } from "lucide-react";
import { patternsApi } from "../../api/patterns";
import { useData } from "../../hooks/useData";
import { cn } from "../../lib/format";
import { EmptyState, StatCard } from "../ui";

export default function PatternOverviewSection() {
  const { data: o } = useData(patternsApi.overview, [], 20000);
  if (!o) return null;
  const analysed = o.fighters.filter((f) => f.analysis === "up_to_date").length;
  const multi = o.fighters.filter((f) => f.fights >= 2).length;
  return (
    <section className="mt-8">
      <div className="mb-3 flex items-end justify-between">
        <div>
          <h2 className="font-display text-2xl font-bold uppercase tracking-wide text-white">Fighter Pattern Intelligence</h2>
          <p className="text-sm text-ink-400">Combinations, transition probabilities and tendencies from recognised actions (Phase 4)</p>
        </div>
        <Link to="/patterns" className="btn-secondary"><Network className="size-4" />Open</Link>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Actions available" value={o.totals.actions} icon={<GitBranch className="size-4" />} hint={`${o.totals.videos} video(s) · ${o.totals.fights} fight(s)`} accent />
        <StatCard label="Fighters / tracks" value={o.totals.fighters} icon={<Users className="size-4" />} hint={`${o.fighters.filter((f) => !f.is_track).length} with fighter profiles`} />
        <StatCard label="Analysed" value={`${analysed}/${o.fighters.length}`} icon={<BrainCircuit className="size-4" />} hint="stored analyses up to date" />
        <StatCard label="Cross-fight data" value={multi} icon={<Network className="size-4" />} hint="fighters seen in ≥2 fights" />
      </div>
      {o.fighters.length ? (
        <div className="panel mt-4">
          <div className="panel-header"><h3 className="label-caps">Fighters by available data</h3></div>
          <div className="divide-y divide-ink-700/60">
            {o.fighters.slice(0, 6).map((f) => (
              <Link key={f.fighter_key} to={`/patterns/${f.fighter_key}`} className="flex items-center justify-between px-5 py-2.5 text-sm hover:bg-ink-800">
                <span className={f.is_track ? "text-ink-300" : "text-white"}>{f.name}</span>
                <span className="flex gap-4 text-xs text-ink-400 tabular">
                  <span>{f.actions} actions</span><span>{f.fights} fight(s)</span>
                  <span className={cn(f.analysis === "up_to_date" ? "text-emerald-300" : f.analysis === "stale" ? "text-amber-300" : "text-ink-500")}>{f.analysis.replace(/_/g, " ")}</span>
                </span>
              </Link>
            ))}
          </div>
        </div>
      ) : <div className="mt-4"><EmptyState icon={<Network className="size-5" />} title="No action data yet" body="Patterns appear once Phase 3 has recognised actions or verified annotations." /></div>}
    </section>
  );
}
