import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3, Eye, Sparkles, UserRound } from "lucide-react";
import { platformApi } from "../api/platform";
import { pct, tip } from "../components/platform/FightDashboard";
import { PredictionCard } from "../components/platform/PredictionCard";
import { Seq } from "../components/patterns/common";
import { EmptyState, ErrorBlock, LoadingBlock, PageHeader } from "../components/ui";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

function Section({ kind, title, children }: { kind: "observed" | "pattern" | "prediction"; title: string; children: React.ReactNode }) {
  const m = {
    observed: { icon: <Eye className="size-4" />, tag: "Observed behaviour", cls: "border-l-sky-400" },
    pattern: { icon: <BarChart3 className="size-4" />, tag: "Statistical pattern", cls: "border-l-emerald-400" },
    prediction: { icon: <Sparkles className="size-4" />, tag: "ML prediction — not a confirmed action", cls: "border-l-blood-500" },
  }[kind];
  return (
    <section className={cn("panel border-l-2", m.cls)}>
      <div className="panel-header"><h3 className="label-caps">{title}</h3><span className="flex items-center gap-1.5 text-[11px] text-ink-400">{m.icon}{m.tag}</span></div>
      {children}
    </section>
  );
}

export default function FighterIntelligencePage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const list = useData(platformApi.fighters, []);
  const [videos, setVideos] = useState<string[]>([]);
  useEffect(() => setVideos([]), [id]);
  const intel = useData(() => (id ? platformApi.intelligence(id, videos) : Promise.resolve(undefined)), [id, videos.join(",")]);
  const pred = useData(() => (id ? platformApi.predictions(id) : Promise.resolve(undefined)), [id]);
  useEffect(() => {
    if (!id && list.data?.length) navigate(`/intelligence/${list.data[0].fighter_key}`, { replace: true });
  }, [id, list.data, navigate]);

  if (list.error) return <ErrorBlock message={list.error} />;
  if (!list.data) return <LoadingBlock />;
  if (!list.data.length) return <><PageHeader title="Fighter Intelligence" /><div className="panel"><EmptyState icon={<UserRound className="size-5" />} title="No fighters yet" body={<>Create fighter profiles and assign tracks in <Link className="text-blood-300" to="/tracking">Computer Vision Analysis</Link>.</>} /></div></>;
  const d = intel.data;
  const o = d?.observed;
  const toggle = (v: string) => setVideos(videos.includes(v) ? videos.filter((x) => x !== v) : [...videos, v]);
  return (
    <>
      <PageHeader title="Fighter Intelligence" subtitle="Observed behaviour, statistically identified patterns and model predictions — kept visibly separate"
        actions={
          <select className="input w-72" value={id ?? ""} onChange={(e) => navigate(`/intelligence/${e.target.value}`)} aria-label="Fighter">
            {list.data.map((f) => <option key={f.fighter_key} value={f.fighter_key}>{f.name}{f.is_profile ? "" : " (unassigned)"} · {f.actions} actions</option>)}
          </select>
        } />
      {intel.error ? <ErrorBlock message={intel.error} onRetry={() => intel.reload()} /> : !d || !o ? <LoadingBlock /> : (
        <div className="space-y-4">
          <div className="panel flex flex-wrap items-center gap-x-8 gap-y-2 px-5 py-4">
            <div>
              <div className="font-display text-3xl font-bold uppercase tracking-wide text-white">{d.fighter.name}</div>
              <div className="text-xs text-ink-400">{[d.fighter.nickname && `“${d.fighter.nickname}”`, d.fighter.stance, d.fighter.weight_class, d.fighter.height_cm && `${d.fighter.height_cm} cm`, d.fighter.reach_cm && `reach ${d.fighter.reach_cm} cm`].filter(Boolean).join(" · ") || (d.fighter.is_profile ? "Profile" : "Unassigned track / imported athlete")}</div>
            </div>
            {[["Analysed time", o.analysed_seconds ? `${Math.round(o.analysed_seconds)} s` : "—", "visible in tracked footage"],
              ["Actions", o.actions, `${o.fights} fight(s)`],
              ["Avg. time between actions", o.mean_time_between_actions != null ? `${o.mean_time_between_actions.toFixed(2)} s` : "—", `median ${o.median_time_between_actions?.toFixed(2) ?? "—"} s · n=${o.interval_samples}`],
              ["Offensive / defensive", `${o.group_distribution.offensive} / ${o.group_distribution.defensive}`, `${o.group_distribution.movement} movement`]].map(([l, v, h]) => (
              <div key={l as string}><div className="label-caps">{l}</div><div className="font-display text-2xl text-white tabular">{v}</div><div className="text-[11px] text-ink-500">{h}</div></div>
            ))}
            {d.fighter.is_profile && <Link to={`/patterns/${d.fighter.fighter_key}`} className="btn-secondary ml-auto">Full pattern analysis</Link>}
          </div>
          {d.warnings.map((w) => <div key={w} className="rounded-md border border-amber-500/30 bg-amber-500/5 px-4 py-2 text-sm text-amber-200">{w}</div>)}

          <Section kind="observed" title={`Fight videos (${d.videos.length}) — select to compare or filter`}>
            {d.videos.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-[11px] uppercase tracking-wide text-ink-500"><tr className="border-b border-ink-700"><th className="px-5 py-2">Use</th><th>Video</th><th>Fight</th><th>Date</th><th>Actions</th><th>Attacks/min</th><th>Most used</th><th title="Jensen–Shannon distance of this video's action mix from the fighter's overall mix">Differs from usual</th><th /></tr></thead>
                  <tbody>
                    {d.videos.map((v) => (
                      <tr key={v.video_id} className="border-b border-ink-700/50">
                        <td className="px-5"><input type="checkbox" aria-label={`Filter ${v.title}`} checked={videos.includes(v.video_id)} onChange={() => toggle(v.video_id)} /></td>
                        <td className="py-2 text-ink-100">{v.title}</td><td className="font-mono text-xs text-ink-400">{v.fight_key?.slice(0, 14) ?? "—"}</td><td className="text-ink-400">{v.fight_date ?? "—"}</td>
                        <td className="tabular">{v.actions}</td><td className="tabular">{v.attacks_per_minute ?? "—"}</td>
                        <td className="text-xs text-ink-300">{v.distribution.slice(0, 2).map((x) => `${x.label} ${x.count}`).join(", ") || "—"}</td>
                        <td className="tabular">{v.js_distance_from_overall?.toFixed(2) ?? "—"}</td>
                        <td className="pr-5 text-right"><Link className="text-xs text-ink-300 hover:text-white" to={`/videos/${v.video_id}/analysis`}>Open analysis →</Link></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {videos.length > 0 && <p className="px-5 py-2 text-xs text-sky-300">Statistics below use {videos.length} selected video(s) only. <button className="underline" onClick={() => setVideos([])}>Show all</button></p>}
              </div>
            ) : <p className="px-5 py-6 text-sm text-ink-500">No videos linked to this fighter.</p>}
          </Section>

          <div className="grid gap-4 xl:grid-cols-2">
            <Section kind="observed" title="Strike frequency">
              {o.strike_frequency.length ? (
                <div className="h-60 px-2 py-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={o.strike_frequency} layout="vertical" margin={{ left: 20, right: 16 }}>
                      <CartesianGrid horizontal={false} stroke="#232b39" />
                      <XAxis type="number" tick={{ fill: "#8b93a3", fontSize: 11 }} />
                      <YAxis type="category" dataKey="label" width={130} tick={{ fill: "#c7cbd3", fontSize: 11 }} />
                      <Tooltip {...tip} formatter={(v, _n, p) => [`${v} (${pct((p as { payload: { share: number } }).payload.share)})`, "Count"]} />
                      <Bar dataKey="count" fill="#e11d2e" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : <p className="px-5 py-6 text-sm text-ink-500">No strikes recognised.</p>}
            </Section>
            <Section kind="observed" title="Openings, counters & movement">
              <div className="grid gap-4 px-5 py-4 text-sm sm:grid-cols-2">
                <div><div className="label-caps mb-1">Most common opening actions</div>{o.opening_actions.map((a) => <div key={a.action} className="flex justify-between"><span>{a.label}</span><span className="tabular text-ink-400">{a.count} · {pct(a.share)}</span></div>)}{!o.opening_actions.length && <span className="text-ink-500">—</span>}</div>
                <div><div className="label-caps mb-1">Most frequent counterattacks</div>{o.counterattacks.map((c, i) => <div key={i} className="flex justify-between"><span>{c.counter} <span className="text-ink-500">after {c.after}</span></span><span className="tabular text-ink-400">{c.count}</span></div>)}{!o.counterattacks.length && <span className="text-ink-500">None observed (needs opponent actions in the same video)</span>}</div>
                <div className="sm:col-span-2">
                  <div className="label-caps mb-1">Movement tendencies (pose tracking)</div>
                  {o.movement.samples ? (
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      <span className="text-ink-400">Time moving / still</span><span className="tabular">{pct(o.movement.state_share?.moving ?? 0)} / {pct(o.movement.state_share?.stationary ?? 0)}</span>
                      <span className="text-ink-400">Mean centre speed</span><span className="tabular">{o.movement.mean_center_speed ?? "—"}</span>
                      <span className="text-ink-400">Wrist speed L / R</span><span className="tabular">{o.movement.mean_wrist_speed?.left ?? "—"} / {o.movement.mean_wrist_speed?.right ?? "—"}</span>
                      <span className="text-ink-400">Ankle speed L / R</span><span className="tabular">{o.movement.mean_ankle_speed?.left ?? "—"} / {o.movement.mean_ankle_speed?.right ?? "—"}</span>
                      <span className="text-ink-400">Direction when moving</span><span>{Object.entries(o.movement.screen_direction_when_moving ?? {}).map(([k, v]) => `${k} ${pct(v)}`).join(" · ")}</span>
                      <span className="col-span-2 text-[11px] text-ink-500">{o.movement.units} · {o.movement.samples} pose samples</span>
                    </div>
                  ) : <span className="text-xs text-ink-500">No pose samples for tracks assigned to this fighter.</span>}
                </div>
              </div>
            </Section>
          </div>

          <Section kind="observed" title={`Historical action timeline (${o.timeline.length})`}>
            <div className="max-h-64 overflow-auto px-5 py-3">
              {o.timeline.length ? Object.entries(o.timeline.reduce<Record<string, typeof o.timeline>>((acc, a) => { (acc[a.video_id] ??= []).push(a); return acc; }, {})).map(([vid, acts]) => (
                <div key={vid} className="mb-3">
                  <div className="mb-1 text-xs text-ink-400">{d.videos.find((v) => v.video_id === vid)?.title ?? vid}</div>
                  <div className="flex flex-wrap gap-1">
                    {acts.map((a, i) => (
                      <Link key={i} to={`/videos/${vid}/analysis?t=${a.start}`} title={`${a.start.toFixed(2)} s · ${pct(a.confidence)} · ${a.source}`}
                        className={cn("rounded px-1.5 py-0.5 text-[11px] ring-1 hover:ring-white", a.group === "offensive" ? "bg-blood-500/15 text-blood-200 ring-blood-500/30" : a.group === "defensive" ? "bg-sky-500/15 text-sky-200 ring-sky-500/30" : "bg-ink-700 text-ink-200 ring-ink-600")}>
                        {a.start.toFixed(1)}s {a.label}
                      </Link>
                    ))}
                  </div>
                </div>
              )) : <span className="text-sm text-ink-500">No actions.</span>}
            </div>
          </Section>

          <div className="grid gap-4 xl:grid-cols-2">
            <Section kind="pattern" title="Fighting combinations">
              {d.patterns.combinations.length ? (
                <div className="divide-y divide-ink-700/60">
                  {d.patterns.combinations.map((c) => (
                    <div key={c.sequence.join(">")} className="flex flex-wrap items-center justify-between gap-2 px-5 py-2">
                      <Seq labels={c.labels} />
                      <span className="text-xs text-ink-400 tabular">×{c.occurrences} · {c.distinct_fights} fight(s) · P(last|prefix) {pct(c.confidence)} · {c.classification.replace(/_/g, " ")}</span>
                    </div>
                  ))}
                </div>
              ) : <p className="px-5 py-6 text-sm text-ink-500">No combination reaches the minimum number of occurrences.</p>}
            </Section>
            <Section kind="pattern" title="Tendencies & opponent-specific patterns">
              <div className="space-y-2 px-5 py-4 text-sm">
                {d.patterns.tendencies ? (
                  <>
                    <div className="flex justify-between"><span className="text-ink-400">Attacks per minute</span><span className="tabular">{d.patterns.tendencies.offensive.attack_frequency_per_minute ?? "—"}</span></div>
                    <div className="flex justify-between"><span className="text-ink-400">Counter rate after opponent attacks</span><span className="tabular">{pct(d.patterns.tendencies.defensive.counterattack_rate)}</span></div>
                    <div className="flex justify-between"><span className="text-ink-400">Distinct transitions / entropy</span><span className="tabular">{d.patterns.tendencies.behavioral.sequence_diversity.distinct_transitions} / {d.patterns.tendencies.behavioral.sequence_diversity.transition_entropy_bits} bits</span></div>
                  </>
                ) : <span className="text-ink-500">No data.</span>}
                {d.patterns.opponents?.opponents.map((op) => (
                  <div key={op.opponent_key} className="rounded border border-ink-700/60 px-3 py-2 text-xs">
                    <div className="flex justify-between"><span className="text-ink-100">vs {op.opponent}</span><span className={op.sufficient ? "text-emerald-300" : "text-ink-500"}>{op.sufficient ? "sufficient data" : "insufficient data"}</span></div>
                    <div className="text-ink-400">{op.actions} actions · offensive {pct(op.offensive_share)} · counter {pct(op.after_opponent_attacks.counter_rate)}</div>
                  </div>
                ))}
                {d.patterns.opponents && <p className="text-[11px] text-ink-500">{d.patterns.opponents.difference_test.valid ? `Difference between opponents: p=${d.patterns.opponents.difference_test.p_value}` : `Opponent difference not tested — ${d.patterns.opponents.difference_test.reason}`}</p>}
              </div>
            </Section>
          </div>

          <Section kind="prediction" title="Next-move prediction from the latest recognised actions">
            {pred.error ? <ErrorBlock message={pred.error} /> : !pred.data ? <LoadingBlock /> : <PredictionCard data={pred.data} />}
          </Section>
        </div>
      )}
    </>
  );
}
