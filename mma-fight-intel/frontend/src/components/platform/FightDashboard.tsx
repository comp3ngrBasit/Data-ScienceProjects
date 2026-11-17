import { useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, BrainCircuit, Film, GitBranch, Layers, Sparkles, Swords, Users } from "lucide-react";
import { platformApi } from "../../api/platform";
import { useData } from "../../hooks/useData";
import { cn } from "../../lib/format";
import { Heatmap, Seq } from "../patterns/common";
import { ErrorBlock, LoadingBlock, StatCard } from "../ui";

export const tip = { contentStyle: { background: "#161c28", border: "1px solid #2b3445", borderRadius: 6, fontSize: 12 }, labelStyle: { color: "#d6d6dc" }, itemStyle: { color: "#f1f1f4" }, cursor: { fill: "rgba(255,255,255,0.04)" } };
export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const GROUP_LABEL: Record<string, string> = { offensive: "Offensive", defensive: "Defensive", movement: "Movement", unknown: "Unknown", rare: "Rare" };

function StatusPill({ s }: { s: string }) {
  const m: Record<string, string> = { evaluated: "text-emerald-300", experimental: "text-amber-300", not_trained: "text-ink-400", unevaluated: "text-amber-300" };
  return <span className={cn("text-xs font-medium", m[s] ?? "text-ink-300")}>{s.replace("_", " ")}</span>;
}

export default function FightDashboard() {
  const ov = useData(platformApi.overview, [], (d) => (d?.pipeline_runs.some((r) => r.status === "processing" || r.status === "queued") ? 3000 : 30000));
  const fighters = useData(platformApi.fighters, []);
  const [focus, setFocus] = useState("");
  const an = useData(() => platformApi.analytics(focus || undefined), [focus]);
  if (ov.error) return <ErrorBlock message={ov.error} onRetry={() => ov.reload()} />;
  if (!ov.data) return <LoadingBlock />;
  const t = ov.data.totals;
  const m = ov.data.models;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
        <StatCard label="Fights analysed" value={t.fights_analysed} icon={<Swords className="size-4" />} hint={ov.data.definitions.fights_analysed} accent />
        <StatCard label="Videos processed" value={t.videos_processed} icon={<Film className="size-4" />} hint={`${t.videos_analysed} with tracking & pose · ${t.videos} uploaded`} />
        <StatCard label="Fighters analysed" value={t.fighters_analysed} icon={<Users className="size-4" />} hint={`${t.fighter_profiles} profiles`} />
        <StatCard label="Detected actions" value={t.actions_detected} icon={<Activity className="size-4" />} hint={`${t.verified_annotations} verified annotations · ${t.actions_in_analysis} in analysis`} />
        <StatCard label="Combinations" value={t.combinations_identified} icon={<GitBranch className="size-4" />} hint={ov.data.definitions.combinations_identified} />
        <StatCard label="Predictions" value={t.predictions_generated} icon={<Sparkles className="size-4" />} hint={t.predictions_evaluated_held_out ? `held-out: top-1 ${t.held_out_top1}/${t.predictions_evaluated_held_out}` : "none evaluated on held-out fights"} />
        <div className="panel px-5 py-4">
          <div className="flex items-center justify-between"><span className="label-caps">Model status</span><BrainCircuit className="size-4 text-ink-500" /></div>
          <div className="mt-2 space-y-1.5 text-xs">
            <div className="flex justify-between gap-2"><span className="text-ink-400">Actions</span><StatusPill s={ov.data.status.action_recognition} /></div>
            <div className="text-[11px] text-ink-500">{m.action_recognition ? `${m.action_recognition.algorithm} · F1 ${m.action_recognition.f1_macro?.toFixed(2) ?? "—"} (n=${m.action_recognition.n_evaluated ?? 0})` : "no active model"}</div>
            <div className="flex justify-between gap-2"><span className="text-ink-400">Next move</span><StatusPill s={ov.data.status.next_move} /></div>
            <div className="text-[11px] text-ink-500">{m.next_move ? `${m.next_move.name} · top-1 ${pct(m.next_move.top1)} (n=${m.next_move.n})` : "no active model"}</div>
            <Link to="/evaluation" className="text-[11px] text-ink-300 hover:text-white">Evaluation →</Link>
          </div>
        </div>
      </div>

      <section className="panel">
        <div className="panel-header"><h2 className="label-caps">Fighters</h2><Link to="/intelligence" className="text-xs text-ink-300 hover:text-white">Fighter intelligence →</Link></div>
        {fighters.data?.length ? (
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-ink-850 text-[11px] uppercase tracking-wide text-ink-500">
                <tr className="border-b border-ink-700"><th className="px-5 py-2">Fighter</th><th className="px-2">Videos</th><th className="px-2">Fights</th><th className="px-2">Observed</th><th className="px-2">Actions</th><th className="px-2">Top action</th><th className="px-2">Offensive</th><th className="px-2">Combinations</th><th className="px-2">Predictions</th></tr>
              </thead>
              <tbody>
                {fighters.data.slice(0, 50).map((f) => (
                  <tr key={f.fighter_key} className="border-b border-ink-700/50 hover:bg-ink-800/60">
                    <td className="px-5 py-2"><Link to={`/intelligence/${f.fighter_key}`} className={f.is_profile ? "text-white hover:underline" : "text-ink-300 hover:underline"}>{f.name}</Link></td>
                    <td className="px-2 tabular">{f.videos}</td><td className="px-2 tabular">{f.fights}</td>
                    <td className="px-2 tabular">{f.observed_seconds ? `${Math.round(f.observed_seconds)} s` : "—"}</td>
                    <td className="px-2 tabular">{f.actions}</td><td className="px-2">{f.top_action ?? "—"}</td>
                    <td className="px-2 tabular">{pct(f.offensive_share)}</td><td className="px-2 tabular">{f.combinations}</td><td className="px-2 tabular">{f.predictions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="px-5 py-6 text-sm text-ink-500">No fighters with analysed actions yet. Upload footage and run the full analysis from a video's page.</p>}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2 className="label-caps">Fighting pattern analytics</h2>
          <select className="input h-8 w-64 py-0 text-xs" value={focus} onChange={(e) => setFocus(e.target.value)} aria-label="Fighter filter">
            <option value="">All fighters</option>
            {fighters.data?.filter((f) => f.actions).map((f) => <option key={f.fighter_key} value={f.fighter_key}>{f.name}</option>)}
          </select>
        </div>
        {an.error ? <ErrorBlock message={an.error} /> : !an.data ? <LoadingBlock /> : an.data.actions === 0 ? <p className="px-5 py-6 text-sm text-ink-500">No action sequences yet.</p> : (
          <div className="grid gap-4 p-4 xl:grid-cols-2">
            <div>
              <div className="label-caps mb-2">Most common combinations ({an.data.sequences} sequences, {an.data.actions} actions)</div>
              <div className="max-h-72 divide-y divide-ink-700/60 overflow-auto rounded-md border border-ink-700/60">
                {an.data.top_combinations.length ? an.data.top_combinations.map((c, i) => (
                  <div key={i} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                    <Seq labels={c.labels} />
                    <span className="text-xs text-ink-400 tabular">×{c.occurrences} · {c.distinct_fights} fight(s){!focus && ` · ${c.fighter}`}</span>
                  </div>
                )) : <p className="px-3 py-4 text-sm text-ink-500">No combination reaches the minimum occurrences.</p>}
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="label-caps mb-2">Sequence length distribution</div>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={an.data.sequence_length_distribution} margin={{ left: -16, right: 8 }}>
                      <CartesianGrid vertical={false} stroke="#232b39" />
                      <XAxis dataKey="length" tick={{ fill: "#8b93a3", fontSize: 11 }} tickLine={false} />
                      <YAxis tick={{ fill: "#8b93a3", fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip {...tip} labelFormatter={(l) => `${l} action(s)`} />
                      <Bar dataKey="count" name="Sequences" fill="#e11d2e" radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <div className="label-caps mb-2">Offensive / defensive transitions</div>
                <div className="space-y-1.5 text-xs">
                  {an.data.group_transitions.map((g) => (
                    <div key={g.from + g.to} className="grid grid-cols-[1fr_5rem_3rem] items-center gap-2">
                      <span className="text-ink-200">{GROUP_LABEL[g.from] ?? g.from} → {GROUP_LABEL[g.to] ?? g.to}</span>
                      <div className="h-2 rounded-full bg-ink-750"><div className="h-full rounded-full bg-blood-500" style={{ width: `${g.share * 100}%` }} /></div>
                      <span className="text-right tabular text-ink-400">{g.count}</span>
                    </div>
                  ))}
                </div>
                <div className="label-caps mb-2 mt-4">Action distribution</div>
                <div className="space-y-1 text-xs">
                  {an.data.action_distribution.slice(0, 6).map((d) => <div key={d.action} className="flex justify-between"><span className="text-ink-200">{d.label}</span><span className="tabular text-ink-400">{d.count}</span></div>)}
                </div>
              </div>
            </div>
            <div className="xl:col-span-2">
              <div className="label-caps mb-1">Action transition heatmap (first order, observed)</div>
              {an.data.markov ? <Heatmap m={an.data.markov} mode="raw" /> : null}
            </div>
            {Object.keys(an.data.opponent_patterns).length > 0 && (
              <div className="xl:col-span-2">
                <div className="label-caps mb-2">Opponent-dependent patterns</div>
                <div className="grid gap-2 md:grid-cols-2">
                  {Object.entries(an.data.opponent_patterns).slice(0, 6).map(([k, v]) => (
                    <div key={k} className="rounded-md border border-ink-700/60 px-3 py-2 text-xs">
                      <div className="text-ink-100">{v.fighter}</div>
                      {v.opponents.map((o) => <div key={o.opponent} className="flex justify-between text-ink-400"><span>vs {o.opponent}</span><span className="tabular">{o.actions} actions · counter {pct(o.counter_rate)}{o.sufficient ? "" : " · insufficient"}</span></div>)}
                      <div className="mt-1 text-[11px] text-ink-500">{v.difference_test.valid ? `Difference between opponents: p=${v.difference_test.p_value}` : `Difference not tested — ${v.difference_test.reason}`}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </section>
      {ov.data.pipeline_runs.some((r) => r.status === "processing" || r.status === "queued") && (
        <div className="panel flex items-center gap-3 px-5 py-3 text-xs text-sky-200"><Layers className="size-4" />Full analysis running: {ov.data.pipeline_runs.filter((r) => r.status === "processing").map((r) => r.current_stage).join(", ")}</div>
      )}
    </div>
  );
}
