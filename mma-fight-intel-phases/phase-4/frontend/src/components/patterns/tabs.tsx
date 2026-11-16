import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  DEFINITIONS,
  patternsApi,
  type Combination,
  type Definition,
  type FighterIndexRow,
  type FightRow,
  type PatternFilters,
  type PatternSettings,
} from "../../api/patterns";
import { useToast } from "../../context/ToastContext";
import { useData } from "../../hooks/useData";
import { cn } from "../../lib/format";
import { LoadingBlock, ErrorBlock } from "../ui";
import { Empty, Heatmap, Panel, ReliabilityBadge, Seq, ShareBar, Warnings, ci, num, pct } from "./common";

type P = { fighter: string; filters: PatternFilters; fkey: string };
const tip = { contentStyle: { background: "#18181d", border: "1px solid #33333c", borderRadius: 6, fontSize: 12 }, labelStyle: { color: "#d6d6dc" }, itemStyle: { color: "#f1f1f4" }, cursor: { fill: "rgba(255,255,255,0.04)" } };

function Loading({ error, reload }: { error?: string; reload: () => void }) {
  return error ? <ErrorBlock message={error} onRetry={reload} /> : <LoadingBlock />;
}

const CLASS: Record<Combination["classification"], { label: string; cls: string; hint: string }> = {
  repeated_pattern: { label: "Repeated pattern", cls: "text-emerald-300", hint: "Seen in ≥2 fights and more often than chance (one-sided binomial test vs. how often the last action follows anything)" },
  single_fight: { label: "Single fight", cls: "text-violet-300", hint: "All occurrences come from one fight" },
  not_distinguishable_from_chance: { label: "Possibly coincidental", cls: "text-ink-400", hint: "Not more frequent than chance at the configured significance level" },
};

// ============================================================================ tendencies
export function TendenciesTab({ fighter, filters, fkey }: P) {
  const { data, error, reload } = useData(() => patternsApi.tendencies(fighter, filters), [fighter, fkey]);
  if (!data) return <Loading error={error} reload={reload} />;
  const t = data.tendencies;
  const dist = (items: typeof t.offensive.most_used, empty: string) =>
    items.length ? (
      <div className="space-y-2 px-5 py-4">
        {items.map((d) => (
          <div key={d.action} className="grid grid-cols-[minmax(0,10rem)_1fr_auto] items-center gap-3 text-sm">
            <span className="truncate text-ink-200">{d.label}</span>
            <ShareBar value={d.share} interval={d.ci95} />
            <span className="w-40 text-right text-xs text-ink-400 tabular">{d.count} · {pct(d.share)} <span className="text-ink-500">({ci(d.ci95)})</span>{d.per_minute != null && <> · {num(d.per_minute, 1)}/min</>}</span>
          </div>
        ))}
        <p className="pt-1 text-[11px] text-ink-500">Bar = share of this group; line = 95% Wilson interval.</p>
      </div>
    ) : <Empty>{empty}</Empty>;
  const r = t.defensive.reactions_to_opponent_attacks;
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Attacks / minute" value={num(t.offensive.attack_frequency_per_minute, 1)} hint={`${t.minutes_observed} min of footage`} />
        <Kpi label="Offensive share" value={pct(t.offensive.share_of_actions)} hint={`${t.group_counts.offensive ?? 0} of ${t.actions} actions`} />
        <Kpi label="Avg. attacks per sequence" value={num(t.offensive.average_combination_length, 2)} hint="sequences with ≥1 attack" />
        <Kpi label="Counter rate" value={pct(t.defensive.counterattack_rate)} hint={r.opponent_attacks ? `after ${r.opponent_attacks} opponent attacks · CI ${ci(t.defensive.counterattack_ci95)}` : "no attributed opponent attacks"} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Offensive tendencies · most used attacks">{dist(t.offensive.most_used, "No offensive actions.")}</Panel>
        <Panel title="Defensive tendencies · most used defences">{dist(t.defensive.most_used, "No defensive actions recognised for this fighter.")}</Panel>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Preferred attacking combinations">
          <ComboList items={t.offensive.top_combinations} empty="No attack-only combination reaches the minimum occurrence threshold." />
        </Panel>
        <Panel title={`Reactions to opponent attacks (within ${r.window_seconds}s)`}>
          {r.by_attack.length ? (
            <div className="divide-y divide-ink-700/60">
              {r.by_attack.map((a) => (
                <div key={a.opponent_attack} className="px-5 py-3 text-sm">
                  <div className="flex items-center justify-between"><span className="text-ink-100">After opponent's {a.label}</span><span className="flex items-center gap-2 text-xs text-ink-400">n={a.n} · {a.fights} fight(s) <ReliabilityBadge r={a.reliability} /></span></div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-300">
                    {a.responses.map((x) => <span key={x.action}>{x.label} <span className="tabular text-ink-400">{x.count} ({pct(x.probability)})</span></span>)}
                  </div>
                </div>
              ))}
            </div>
          ) : <Empty>No opponent actions are attributed in the same videos.</Empty>}
        </Panel>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Behaviour">
          <div className="space-y-2 px-5 py-4 text-sm">
            <Row k="Typical pause between actions" v={t.behavioral.typical_interval ? `${num(t.behavioral.typical_interval.median)} s (IQR ${num(t.behavioral.typical_interval.p25)}–${num(t.behavioral.typical_interval.p75)}, n=${t.behavioral.typical_interval.n})` : "—"} />
            <Row k="Distinct actions" v={t.behavioral.sequence_diversity.distinct_actions} />
            <Row k="Action entropy" v={`${t.behavioral.sequence_diversity.action_entropy_bits} bits`} />
            <Row k="Distinct transitions" v={t.behavioral.sequence_diversity.distinct_transitions} />
            <Row k="Transition entropy" v={`${t.behavioral.sequence_diversity.transition_entropy_bits} bits`} />
          </div>
        </Panel>
        <Panel title="Changes across fights" className="lg:col-span-2">
          {t.behavioral.changes_across_fights.length > 1 ? (
            <div className="px-5 py-4 text-sm">
              <table className="w-full text-left text-xs">
                <thead className="text-ink-500"><tr><th className="py-1">Fight</th><th>Actions</th><th>Most used</th><th className="text-right">JS distance from overall</th></tr></thead>
                <tbody className="text-ink-200">
                  {t.behavioral.changes_across_fights.map((c) => <tr key={c.fight_key} className="border-t border-ink-700/60"><td className="py-1.5 font-mono">{c.fight_key.slice(0, 16)}</td><td className="tabular">{c.actions}</td><td>{c.top.join(", ")}</td><td className="text-right tabular">{num(c.js_distance_from_overall, 3)}</td></tr>)}
                </tbody>
              </table>
              <p className="mt-2 text-[11px] text-ink-500">Homogeneity (chi-square): {t.behavioral.homogeneity_test.valid ? `χ²=${t.behavioral.homogeneity_test.chi2}, p=${t.behavioral.homogeneity_test.p_value}` : `not tested — ${t.behavioral.homogeneity_test.reason}`}</p>
            </div>
          ) : <Empty>Only one fight analysed — changes across fights can't be assessed.</Empty>}
        </Panel>
      </div>
    </div>
  );
}

function Kpi({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return <div className="panel px-4 py-3"><div className="label-caps">{label}</div><div className="mt-1 font-display text-3xl font-semibold text-white tabular">{value}</div>{hint && <div className="mt-1 text-[11px] text-ink-500">{hint}</div>}</div>;
}
function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return <div className="flex justify-between gap-3"><span className="text-ink-400">{k}</span><span className="text-right text-ink-100 tabular">{v}</span></div>;
}

function ComboList({ items, empty }: { items: Combination[]; empty: string }) {
  if (!items.length) return <Empty>{empty}</Empty>;
  return (
    <div className="divide-y divide-ink-700/60">
      {items.map((c) => (
        <div key={c.sequence.join(">")} className="flex flex-wrap items-center justify-between gap-2 px-5 py-2.5">
          <Seq labels={c.labels} />
          <span className="text-xs text-ink-400 tabular">×{c.occurrences} · {c.distinct_fights} fight(s)</span>
        </div>
      ))}
    </div>
  );
}

// ============================================================================ combinations
export function CombinationsTab({ fighter, filters, fkey }: P) {
  const { data, error, reload } = useData(() => patternsApi.combinations(fighter, filters), [fighter, fkey]);
  const [len, setLen] = useState(0);
  const [cls, setCls] = useState<string>("");
  if (!data) return <Loading error={error} reload={reload} />;
  const rows = data.combinations.filter((c) => (!len || (len === 4 ? c.length >= 4 : c.length === len)) && (!cls || c.classification === cls));
  return (
    <div className="space-y-4">
      <Panel title={`Combinations (${data.combinations.length})`} right={
        <div className="flex gap-2">
          <select className="input h-8 py-0 text-xs" value={len} onChange={(e) => setLen(+e.target.value)}><option value={0}>All lengths</option><option value={2}>2 actions</option><option value={3}>3 actions</option><option value={4}>4+ actions</option></select>
          <select className="input h-8 py-0 text-xs" value={cls} onChange={(e) => setCls(e.target.value)}><option value="">All</option>{Object.entries(CLASS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</select>
        </div>}>
        {rows.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[11px] uppercase tracking-wide text-ink-500">
                <tr className="border-b border-ink-700">
                  <th className="px-5 py-2">Sequence</th><th className="px-2" title="Total occurrences">Occ.</th><th className="px-2">Fights</th><th className="px-2" title="Occurrences per analysed fight">Per fight</th>
                  <th className="px-2" title="Average pause between consecutive actions">Avg gap</th><th className="px-2" title="Mean recognition confidence of the actions">Recog.</th>
                  <th className="px-2" title="Share of occurrences followed by another action">Followed</th><th className="px-2" title="P(last action | preceding actions) with 95% CI">P(last | prefix)</th><th className="px-2">Lift</th><th className="px-5">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.sequence.join(">")} className="border-b border-ink-700/50 hover:bg-ink-800/60">
                    <td className="px-5 py-2"><Seq labels={c.labels} /></td>
                    <td className="px-2 tabular whitespace-nowrap">{c.occurrences}</td>
                    <td className="px-2 tabular whitespace-nowrap">{c.distinct_fights}</td>
                    <td className="px-2 tabular whitespace-nowrap">{num(c.frequency_per_fight, 2)}</td>
                    <td className="px-2 tabular whitespace-nowrap">{c.avg_interval == null ? "—" : `${num(c.avg_interval)} s`}</td>
                    <td className="px-2 tabular whitespace-nowrap">{pct(c.avg_confidence)}</td>
                    <td className="px-2 tabular whitespace-nowrap">{pct(c.followed_by_action_pct)}</td>
                    <td className="px-2 text-xs tabular">{pct(c.confidence)} <span className="text-ink-500">({c.occurrences}/{c.confidence_n}, {ci(c.confidence_ci95)})</span></td>
                    <td className="px-2 tabular whitespace-nowrap">{num(c.lift, 2)}{c.p_value != null && <div className="text-[10px] text-ink-500">p={c.p_value}</div>}</td>
                    <td className={cn("px-5 text-xs", CLASS[c.classification].cls)} title={CLASS[c.classification].hint}>{CLASS[c.classification].label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty>No combination reaches the thresholds (≥{data.thresholds.min_occurrences} occurrences in ≥{data.thresholds.min_fights} fight(s)).</Empty>}
      </Panel>
      <Panel title="Sequential patterns (PrefixSpan — gaps allowed)">
        {data.sequential_patterns.length ? (
          <div className="divide-y divide-ink-700/60">
            {data.sequential_patterns.slice(0, 40).map((p) => (
              <div key={p.pattern.join(">")} className="flex flex-wrap items-center justify-between gap-2 px-5 py-2.5 text-sm">
                <Seq labels={p.labels} />
                <span className="text-xs text-ink-400 tabular">support {pct(p.support)} ({p.sequences_containing} seq.) · confidence {pct(p.confidence)} · {p.distinct_fights} fight(s){p.avg_time_between != null && ` · avg ${num(p.avg_time_between)} s between`}{!p.contiguous_only && <span className="ml-1 text-sky-300">· gapped</span>}{p.single_fight && <span className="ml-1 text-violet-300">· single fight</span>}</span>
              </div>
            ))}
          </div>
        ) : <Empty>No sequential pattern reaches the minimum support.</Empty>}
      </Panel>
    </div>
  );
}

// ============================================================================ transitions & probabilities
export function TransitionsTab({ fighter, filters, fkey, states, labels, maxOrder }: P & { states: string[]; labels: Record<string, string>; maxOrder: number }) {
  const [order, setOrder] = useState(1);
  const [mode, setMode] = useState<"raw" | "smoothed">("raw");
  const [def, setDef] = useState<Definition>("next");
  const [ctx, setCtx] = useState<string>("");
  const { data, error, reload } = useData(() => patternsApi.transitions(fighter, order, filters), [fighter, fkey, order]);
  const probs = useData(() => patternsApi.probabilities(fighter, def, 1, filters, ctx ? ctx.split(">") : undefined), [fighter, fkey, def, ctx]);
  const L = (k: string) => labels[k] ?? k;
  return (
    <div className="space-y-4">
      <Panel title="Transition heatmap" right={
        <div className="flex gap-2">
          <select className="input h-8 py-0 text-xs" value={order} onChange={(e) => setOrder(+e.target.value)}>{[1, 2, 3].filter((o) => o <= maxOrder).map((o) => <option key={o} value={o}>{o === 1 ? "1st order (previous action)" : `${o === 2 ? "2nd" : "3rd"} order (previous ${o} actions)`}</option>)}</select>
          <select className="input h-8 py-0 text-xs" value={mode} onChange={(e) => setMode(e.target.value as "raw" | "smoothed")}><option value="raw">Observed (raw)</option><option value="smoothed">Smoothed</option></select>
        </div>}>
        {data ? <Heatmap m={data.markov} mode={mode} /> : <Loading error={error} reload={reload} />}
      </Panel>
      {data && (data.markov.most_common.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Most common transitions">
            <div className="divide-y divide-ink-700/60">{data.markov.most_common.map((t) => <TRow key={t.from.join(">") + t.to} t={t} L={L} />)}</div>
          </Panel>
          <Panel title="Low-frequency transitions (seen once)">
            {data.markov.rare.length ? <div className="divide-y divide-ink-700/60">{data.markov.rare.map((t) => <TRow key={t.from.join(">") + t.to} t={t} L={L} />)}</div> : <Empty>None.</Empty>}
          </Panel>
        </div>
      )}
      <Panel title="Conditional probability  P(next | previous)" right={
        <div className="flex flex-wrap gap-2">
          <select className="input h-8 py-0 text-xs" value={def} onChange={(e) => setDef(e.target.value as Definition)}>{(Object.keys(DEFINITIONS) as Definition[]).map((d) => <option key={d} value={d}>{DEFINITIONS[d]}</option>)}</select>
          <select className="input h-8 py-0 text-xs" value={ctx} onChange={(e) => setCtx(e.target.value)}>
            <option value="">All previous actions</option>
            {states.map((s) => <option key={s} value={s}>After {L(s)}</option>)}
          </select>
        </div>}>
        {probs.data ? (
          <div>
            <p className="px-5 pt-3 text-xs text-ink-400">{probs.data.definition_text} Probabilities are Count(A→B) / Count(A); the three definitions are computed separately and never mixed. Minimum {probs.data.min_observations} observations for a reliable estimate.</p>
            {probs.data.probabilities.length ? (
              <table className="mt-2 w-full text-left text-sm">
                <thead className="text-[11px] uppercase tracking-wide text-ink-500"><tr className="border-b border-ink-700"><th className="px-5 py-2">After</th><th className="px-2">Next</th><th className="px-2">Count</th><th className="w-48 px-2">Probability (95% CI)</th><th className="px-2">Fights</th><th className="px-5">Reliability</th></tr></thead>
                <tbody>
                  {probs.data.probabilities.map((p) => (
                    <tr key={p.context.join(">") + p.next} className="border-b border-ink-700/50">
                      <td className="px-5 py-2 text-ink-200">{p.context_labels.join(" → ")}</td>
                      <td className="px-2 text-ink-100">{p.next_label}</td>
                      <td className="px-2 tabular text-ink-300">{p.count}/{p.context_count}</td>
                      <td className="px-2"><div className="flex items-center gap-2"><ShareBar value={p.probability} interval={p.ci95} /><span className="w-24 shrink-0 text-xs tabular">{pct(p.probability)} <span className="text-ink-500">{ci(p.ci95)}</span></span></div></td>
                      <td className="px-2 tabular text-ink-300">{p.fights}/{p.context_fights}</td>
                      <td className="px-5"><ReliabilityBadge r={p.reliability} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <Empty>No observations for this context.</Empty>}
          </div>
        ) : <Loading error={probs.error} reload={probs.reload} />}
      </Panel>
    </div>
  );
}

function TRow({ t, L }: { t: { from: string[]; to: string; count: number; probability: number }; L: (k: string) => string }) {
  return <div className="flex items-center justify-between px-5 py-2 text-sm"><span className="text-ink-200">{t.from.map(L).join(" → ")} <span className="text-ink-500">→</span> {L(t.to)}</span><span className="text-xs text-ink-400 tabular">{t.count} · {pct(t.probability)}</span></div>;
}

// ============================================================================ explorer
export function ExplorerTab({ fighter, filters, fkey }: P) {
  const [action, setAction] = useState("");
  const { data, error, reload } = useData(() => patternsApi.sequences(fighter, filters, action || undefined), [fighter, fkey, action]);
  useEffect(() => {
    if (data && !action && data.actions_available.length) setAction(data.actions_available[0].action);
  }, [data, action]);
  if (!data) return <Loading error={error} reload={reload} />;
  const ex = data.explorer;
  const list = (items: NonNullable<typeof ex>["before"]) => (
    <div className="space-y-2 px-5 py-4">
      {items.map((b) => <div key={b.action} className="grid grid-cols-[minmax(0,9rem)_1fr_auto] items-center gap-3 text-sm"><span className="truncate text-ink-200">{b.label}</span><ShareBar value={b.probability} interval={b.ci95} /><span className="w-20 text-right text-xs tabular text-ink-400">{b.count} · {pct(b.probability)}</span></div>)}
    </div>
  );
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="label-caps">Action</span>
        <select className="input h-9 w-64" value={action} onChange={(e) => setAction(e.target.value)}>{data.actions_available.map((a) => <option key={a.action} value={a.action}>{a.label}</option>)}</select>
        {ex && <span className="flex items-center gap-2 text-xs text-ink-400">{ex.occurrences} occurrences in {ex.fights} fight(s) <ReliabilityBadge r={ex.reliability} /></span>}
      </div>
      {ex && (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title={`What usually happens before ${ex.label}`}>{ex.before.length ? list(ex.before) : <Empty>Not observed.</Empty>}</Panel>
            <Panel title={`What usually follows ${ex.label}`}>{ex.after.length ? list(ex.after) : <Empty>Not observed.</Empty>}</Panel>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Most common 3-action combinations containing it">
              {ex.three_action.length ? <div className="divide-y divide-ink-700/60">{ex.three_action.slice(0, 5).map((g) => <div key={g.sequence.join(">")} className="flex items-center justify-between px-5 py-2"><Seq labels={g.labels} /><span className="text-xs tabular text-ink-400">×{g.count} · {g.fights} fight(s)</span></div>)}</div> : <Empty>No 3-action sequence contains this action.</Empty>}
            </Panel>
            <Panel title="Most common 5-action combinations containing it">
              {ex.five_action.length ? <div className="divide-y divide-ink-700/60">{ex.five_action.slice(0, 5).map((g) => <div key={g.sequence.join(">")} className="flex items-center justify-between px-5 py-2"><Seq labels={g.labels} /><span className="text-xs tabular text-ink-400">×{g.count} · {g.fights} fight(s)</span></div>)}</div> : <Empty>No 5-action sequence contains this action.</Empty>}
            </Panel>
          </div>
        </>
      )}
      <Panel title={`Chronological action sequences (${data.total_sequences})`}>
        {data.sequences.length ? (
          <div className="max-h-[32rem] divide-y divide-ink-700/60 overflow-y-auto">
            {data.sequences.map((s) => (
              <details key={s.id} className="group px-5 py-2.5">
                <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-2">
                  <Seq labels={s.labels} />
                  <span className="text-[11px] text-ink-500 tabular">{s.video_title} · {s.start.toFixed(2)}–{s.end.toFixed(2)} s{s.fight_date ? ` · ${s.fight_date}` : ""}</span>
                </summary>
                <table className="mt-2 w-full text-left text-xs">
                  <thead className="text-ink-500"><tr><th className="py-1">Action</th><th>Start</th><th>Dur.</th><th>Gap</th><th>Conf.</th><th>Source</th><th>Opponent action</th><th>Distance</th><th>Stance</th></tr></thead>
                  <tbody className="text-ink-200">
                    {s.actions.map((a) => (
                      <tr key={a.action_id} className="border-t border-ink-700/50">
                        <td className="py-1">{a.label}</td><td className="tabular">{a.start.toFixed(2)}</td><td className="tabular">{a.duration.toFixed(2)}</td>
                        <td className="tabular">{a.interval_prev == null ? "—" : a.interval_prev.toFixed(2)}</td><td className="tabular">{pct(a.confidence)}</td><td>{a.source}</td>
                        <td>{a.opponent_action ? `${a.opponent_action_label ?? a.opponent_action} (${(a.opponent_action_lag ?? 0) < 0.05 ? "simultaneous" : `${a.opponent_action_lag!.toFixed(2)} s before`})` : "—"}</td>
                        <td className="tabular">{a.distance == null ? "—" : `${a.distance.toFixed(2)} torso`}</td><td>{a.stance ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            ))}
          </div>
        ) : <Empty>No sequences.</Empty>}
      </Panel>
    </div>
  );
}

// ============================================================================ opponents
export function OpponentsTab({ fighter, filters, fkey }: P) {
  const { data, error, reload } = useData(() => patternsApi.opponents(fighter, filters), [fighter, fkey]);
  if (!data) return <Loading error={error} reload={reload} />;
  return (
    <div className="space-y-4">
      <div className="panel px-5 py-3 text-sm text-ink-300">
        {data.note} Opponent-specific statistics need ≥{data.min_actions_per_opponent} of this fighter's actions against an opponent.{" "}
        Difference between opponents: {data.difference_test.valid ? `χ²=${data.difference_test.chi2}, p=${data.difference_test.p_value}` : <span className="text-ink-400">not tested — {data.difference_test.reason}</span>}
        {data.findings.map((f) => <div key={f} className="mt-1 text-emerald-300">{f}</div>)}
      </div>
      {data.opponents.length ? (
        <div className="grid gap-4 lg:grid-cols-2">
          {data.opponents.map((o) => (
            <Panel key={o.opponent_key} title={`vs ${o.opponent}`} right={<span className={cn("text-[11px]", o.sufficient ? "text-emerald-300" : "text-ink-500")}>{o.sufficient ? "Sufficient data" : "Insufficient data — descriptive only"}</span>}>
              <div className="space-y-3 px-5 py-4 text-sm">
                <Row k="Fights / actions" v={`${o.fights} / ${o.actions}`} />
                <Row k="Opponent style (observed)" v={o.opponent_style.replace("_", " ")} />
                <Row k="Offensive share" v={pct(o.offensive_share)} />
                <Row k="Counter rate after opponent attacks" v={o.after_opponent_attacks.opponent_attacks ? `${pct(o.after_opponent_attacks.counter_rate)} (${ci(o.after_opponent_attacks.counter_ci95)}) of ${o.after_opponent_attacks.opponent_attacks}` : "—"} />
                <Row k="Defensive response rate" v={o.after_opponent_attacks.opponent_attacks ? `${pct(o.after_opponent_attacks.defensive_rate)} (${ci(o.after_opponent_attacks.defensive_ci95)})` : "—"} />
                <div className="text-xs text-ink-400">Actions: {o.action_distribution.map((d) => `${d.label} ${d.count}`).join(" · ")}</div>
                {o.combinations.length > 0 && <div className="space-y-1">{o.combinations.map((c) => <div key={c.sequence.join(">")} className="flex justify-between"><Seq labels={c.labels} /><span className="text-xs text-ink-400">×{c.occurrences}</span></div>)}</div>}
              </div>
            </Panel>
          ))}
        </div>
      ) : <Empty>No opponent is attributed for this fighter. Assign both fighters' tracks (Fighter Tracking) so opponents can be identified.</Empty>}
    </div>
  );
}

// ============================================================================ comparison
export function CompareTab({ fighters, filters, fkey, initial }: { fighters: FighterIndexRow[]; filters: PatternFilters; fkey: string; initial: string }) {
  const [a, setA] = useState(initial);
  const [b, setB] = useState(fighters.find((f) => f.fighter_key !== initial)?.fighter_key ?? "");
  useEffect(() => setA(initial), [initial]);
  const { data, error, reload } = useData(() => (a && b && a !== b ? patternsApi.compare(a, b, filters) : Promise.resolve(undefined)), [a, b, fkey]);
  const chart = useMemo(() => {
    if (!data) return [];
    const [fa, fb] = [data.fighters[a], data.fighters[b]];
    if (!fa || !fb) return [];
    const share = (list: typeof fa.most_used, k: string) => list.find((x) => x.action === k)?.count ?? 0;
    return data.states.map((s) => ({
      action: [...fa.most_used, ...fa.defensive, ...fb.most_used, ...fb.defensive].find((x) => x.action === s)?.label ?? s,
      [data.names[a]]: fa.actions ? (share(fa.most_used, s) + share(fa.defensive, s)) / fa.actions : 0,
      [data.names[b]]: fb.actions ? (share(fb.most_used, s) + share(fb.defensive, s)) / fb.actions : 0,
    }));
  }, [data, a, b]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select className="input h-9 w-64" value={a} onChange={(e) => setA(e.target.value)}>{fighters.map((f) => <option key={f.fighter_key} value={f.fighter_key}>{f.name}</option>)}</select>
        <span className="text-ink-500">vs</span>
        <select className="input h-9 w-64" value={b} onChange={(e) => setB(e.target.value)}><option value="">Choose…</option>{fighters.filter((f) => f.fighter_key !== a).map((f) => <option key={f.fighter_key} value={f.fighter_key}>{f.name}</option>)}</select>
      </div>
      {!b || a === b ? <Empty>Choose a second fighter to compare.</Empty> : !data ? <Loading error={error} reload={reload} /> : (
        <>
          <Warnings items={data.warnings} />
          <Panel title="Action distribution (share of each fighter's actions)">
            <div className="h-72 px-2 py-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chart} margin={{ left: -8, right: 8, top: 4, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke="#26262d" />
                  <XAxis dataKey="action" tick={{ fill: "#a3a3ad", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#33333c" }} interval={0} />
                  <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip {...tip} formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey={data.names[a]} fill="#e11d2e" radius={[4, 4, 0, 0]} maxBarSize={22} />
                  <Bar dataKey={data.names[b]} fill="#38bdf8" radius={[4, 4, 0, 0]} maxBarSize={22} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="px-5 pb-3 text-[11px] text-ink-500">Jensen–Shannon distance between the two distributions: {num(data.action_distribution_js_distance, 3)} (0 = identical, 1 = disjoint). Difference test: {data.difference_test.valid ? `χ²=${data.difference_test.chi2}, p=${data.difference_test.p_value}` : `not tested — ${data.difference_test.reason}`}.</p>
          </Panel>
          <div className="grid gap-4 lg:grid-cols-2">
            {[a, b].map((key) => {
              const f = data.fighters[key];
              const m = data.markov[key];
              return (
                <Panel key={key} title={data.names[key]}>
                  <div className="space-y-2 px-5 py-4 text-sm">
                    <Row k="Actions / fights / minutes" v={`${f.actions} / ${f.fights} / ${f.minutes_observed}`} />
                    <Row k="Attacks per minute" v={num(f.attack_frequency_per_minute, 1)} />
                    <Row k="Offensive / defensive share" v={`${pct(f.offensive_share)} / ${pct(f.defensive_share)}`} />
                    <Row k="Counter rate" v={pct(f.counterattack_rate)} />
                    <Row k="Defensive tendencies" v={f.defensive.length ? f.defensive.slice(0, 3).map((d) => d.label).join(", ") : "—"} />
                    <Row k="Sequence diversity" v={`${f.diversity.distinct_transitions} transitions · ${f.diversity.transition_entropy_bits} bits`} />
                    <div className="pt-2 label-caps">Top combinations</div>
                    {f.top_combinations.length ? f.top_combinations.slice(0, 5).map((c) => <div key={c.sequence.join(">")} className="flex justify-between"><Seq labels={c.labels} /><span className="text-xs text-ink-400">×{c.occurrences}</span></div>) : <div className="text-xs text-ink-500">None above threshold</div>}
                    <div className="pt-2 label-caps">Most likely next (1st order)</div>
                    {m.rows.slice(0, 6).map((r) => <div key={r.context.join()} className="flex justify-between text-xs"><span className="text-ink-300">{r.context_labels[0]} → {r.most_likely_next ? data.state_labels[r.most_likely_next] ?? r.most_likely_next : "—"}</span><span className="text-ink-500 tabular">n={r.n}</span></div>)}
                  </div>
                </Panel>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================================ fights & settings
export function FightsPanel({ onChanged }: { onChanged: () => void }) {
  const toast = useToast();
  const { data, reload } = useData(patternsApi.fights, []);
  const [edit, setEdit] = useState<Record<string, Partial<FightRow>>>({});
  if (!data) return <LoadingBlock />;
  const save = async (v: FightRow) => {
    const e = { ...v, ...edit[v.video_id] };
    try {
      await patternsApi.saveFight(v.video_id, { fight_key: e.fight_key || null, fight_date: e.fight_date || null, event: e.event || null, notes: e.notes || null });
      toast("success", "Fight information saved");
      setEdit((s) => ({ ...s, [v.video_id]: {} }));
      reload();
      onChanged();
    } catch (err) {
      toast("error", err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Panel title="Fights & dates">
      <p className="px-5 pt-3 text-xs text-ink-400">Videos with the same fight key are treated as one fight (several clips or camera angles of the same bout); identical sequences across them are counted once. Without a key, each video is its own fight. Dates drive the date filter and analysis period.</p>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full text-left text-sm">
          <thead className="text-[11px] uppercase tracking-wide text-ink-500"><tr className="border-b border-ink-700"><th className="px-5 py-2">Video</th><th className="px-2">Fight key</th><th className="px-2">Date</th><th className="px-2">Event</th><th className="px-5" /></tr></thead>
          <tbody>
            {data.map((v) => {
              const e = { ...v, ...edit[v.video_id] };
              const set = (k: keyof FightRow, val: string) => setEdit((s) => ({ ...s, [v.video_id]: { ...s[v.video_id], [k]: val } }));
              return (
                <tr key={v.video_id} className="border-b border-ink-700/50">
                  <td className="px-5 py-2 text-ink-200">{v.title}</td>
                  <td className="px-2"><input className="input h-8 w-44" placeholder={v.video_id.slice(0, 8)} value={e.fight_key ?? ""} onChange={(x) => set("fight_key", x.target.value)} /></td>
                  <td className="px-2"><input type="date" className="input h-8 w-40" value={e.fight_date ?? ""} onChange={(x) => set("fight_date", x.target.value)} /></td>
                  <td className="px-2"><input className="input h-8 w-48" value={e.event ?? ""} onChange={(x) => set("event", x.target.value)} /></td>
                  <td className="px-5 text-right"><button className="btn-secondary h-8 px-3 text-xs" disabled={!edit[v.video_id] || !Object.keys(edit[v.video_id]).length} onClick={() => save(v)}>Save</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

const SETTING_FIELDS: { key: keyof PatternSettings; label: string; step?: number; hint: string }[] = [
  { key: "min_confidence", label: "Min. detection confidence", step: 0.05, hint: "Unreviewed detections below this are excluded" },
  { key: "sequence_gap", label: "Sequence gap (s)", step: 0.1, hint: "Pause that ends a sequence" },
  { key: "window", label: "Observation window (s)", step: 0.1, hint: "For “next action within window”" },
  { key: "overlap_iou", label: "Duplicate overlap (IoU)", step: 0.05, hint: "Same-fighter detections overlapping more are merged" },
  { key: "min_occurrences", label: "Min. occurrences", hint: "For combinations / patterns" },
  { key: "min_fights", label: "Min. distinct fights", hint: "2 = cross-fight patterns only" },
  { key: "min_observations", label: "Min. observations", hint: "Below this a probability is “insufficient”" },
  { key: "min_context_count", label: "Min. higher-order context", hint: "Observations for 2nd/3rd-order rows" },
  { key: "max_n", label: "Longest combination", hint: "2–8 actions" },
  { key: "max_order", label: "Max. Markov order", hint: "1–3" },
  { key: "smoothing_alpha", label: "Smoothing α", step: 0.1, hint: "Additive smoothing; 0 = none" },
  { key: "significance", label: "Significance level", step: 0.01, hint: "For repeated-vs-coincidental tests" },
  { key: "opponent_reaction_window", label: "Reaction window (s)", step: 0.1, hint: "Response to opponent attack" },
  { key: "min_actions_per_opponent", label: "Min. actions per opponent", hint: "For opponent-specific findings" },
];

export function SettingsPanel({ onSaved }: { onSaved: () => void }) {
  const toast = useToast();
  const { data } = useData(patternsApi.settings, []);
  const [s, setS] = useState<PatternSettings>();
  useEffect(() => { if (data) setS(data); }, [data]);
  if (!s) return <LoadingBlock />;
  const save = async () => {
    try {
      setS(await patternsApi.saveSettings(s));
      toast("success", "Pattern settings saved — results recompute from the stored actions");
      onSaved();
    } catch (err) {
      toast("error", err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Panel title="Analysis settings" right={<button className="btn-primary h-8 px-3 text-xs" onClick={save}>Save settings</button>}>
      <div className="grid gap-4 px-5 py-4 sm:grid-cols-2 lg:grid-cols-4">
        <label className="text-sm"><div className="label-caps mb-1">Data source</div>
          <select className="input w-full" value={s.source} onChange={(e) => setS({ ...s, source: e.target.value as PatternSettings["source"] })}>
            <option value="best">Best available</option><option value="detections">Model detections</option><option value="annotations">Verified annotations</option>
          </select>
          <div className="mt-1 text-[11px] text-ink-500">Best = annotations for fully annotated videos, detections elsewhere</div>
        </label>
        <label className="flex items-start gap-2 pt-6 text-sm text-ink-200"><input type="checkbox" checked={s.include_low_confidence} onChange={(e) => setS({ ...s, include_low_confidence: e.target.checked })} />Include unreviewed low-confidence detections</label>
        {SETTING_FIELDS.map((f) => (
          <label key={f.key} className="text-sm"><div className="label-caps mb-1">{f.label}</div>
            <input type="number" step={f.step ?? 1} className="input w-full" value={s[f.key] as number} onChange={(e) => setS({ ...s, [f.key]: Number(e.target.value) })} />
            <div className="mt-1 text-[11px] text-ink-500">{f.hint}</div>
          </label>
        ))}
      </div>
    </Panel>
  );
}
