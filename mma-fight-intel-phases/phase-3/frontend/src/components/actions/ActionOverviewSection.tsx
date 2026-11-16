import { Link } from "react-router-dom";
import { Brain, ListChecks, Swords, Tags } from "lucide-react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ALGO_LABELS, GROUP_COLORS, actionsApi } from "../../api/actions";
import { useData } from "../../hooks/useData";
import { EmptyState, ProgressBar, StatCard } from "../ui";
import { pct } from "./common";

export default function ActionOverviewSection() {
  const { data: o } = useData(actionsApi.overview, [], (d) => (d && (d.active_jobs.length || d.training) ? 2000 : 20000));
  if (!o) return null;
  const m = o.active_model;
  return (
    <section className="mt-8">
      <div className="mb-3 flex items-end justify-between">
        <div>
          <h2 className="font-display text-2xl font-bold uppercase tracking-wide text-white">MMA Action Recognition</h2>
          <p className="text-sm text-ink-400">Temporal strike & movement recognition (Phase 3)</p>
        </div>
        <div className="flex gap-2">
          <Link to="/actions/models" className="btn-secondary"><Brain className="size-4" />Models</Link>
          <Link to="/actions" className="btn-secondary"><Swords className="size-4" />Open</Link>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Actions detected" value={o.actions_detected} icon={<Swords className="size-4" />} hint={`${o.videos_analysed} video${o.videos_analysed === 1 ? "" : "s"} analysed`} accent />
        <StatCard label="Annotations" value={o.annotations.total} icon={<Tags className="size-4" />} hint={`${o.annotations.verified} verified · ${Object.keys(o.annotations.by_category).length} categories`} />
        <StatCard label="Active model" value={m ? (ALGO_LABELS[m.algorithm] ?? m.algorithm) : "—"} icon={<Brain className="size-4" />} hint={m ? `${m.classes.filter((c) => c !== "background").length} trained classes` : "train a model to start"} />
        <StatCard label="Validated macro F1" value={m ? pct(m.f1_macro) : "—"} icon={<ListChecks className="size-4" />} hint={m ? `${m.evaluation_strategy ?? ""} · n=${m.n_evaluated ?? 0}` : "no evaluation yet"} />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <section className="panel lg:col-span-2">
          <div className="panel-header"><h3 className="label-caps">Detected actions by category</h3></div>
          {o.by_category.length ? (
            <div className="h-56 px-2 py-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={o.by_category.slice(0, 12)} margin={{ left: -16, right: 8, top: 4, bottom: 0 }}>
                  <XAxis dataKey="label" tick={{ fill: "#a3a3ad", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#33333c" }} interval={0} />
                  <YAxis allowDecimals={false} tick={{ fill: "#75757f", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: "#18181d", border: "1px solid #33333c", fontSize: 12 }} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar dataKey="count" name="Actions" radius={[4, 4, 0, 0]} maxBarSize={36}>{o.by_category.slice(0, 12).map((c) => <Cell key={c.key} fill={GROUP_COLORS[c.group]} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState icon={<Swords className="size-5" />} title="No actions recognised yet" body="Annotate examples, train a model, then analyse a video." />}
        </section>
        <section className="panel">
          <div className="panel-header"><h3 className="label-caps">Jobs</h3></div>
          <div className="space-y-3 px-5 py-4 text-sm">
            {o.training && (
              <div><div className="text-xs text-ink-300">Training · {o.training.current_stage ?? "queued"}</div><ProgressBar value={o.training.progress} tone="sky" className="mt-1" /></div>
            )}
            {o.active_jobs.map((j) => (
              <div key={j.id}><Link to={`/actions/${j.video_id}`} className="text-xs text-ink-200 hover:underline">{j.video_title}</Link><ProgressBar value={j.progress} tone="sky" className="mt-1" /></div>
            ))}
            {!o.training && !o.active_jobs.length && <p className="text-xs text-ink-500">Nothing running. {o.job_counts.completed ?? 0} analyses completed{o.job_counts.failed ? `, ${o.job_counts.failed} failed` : ""}.</p>}
            {m && <p className="border-t border-ink-700/60 pt-3 text-[11px] text-ink-500">Active model recognises: {m.class_labels.filter((l) => l !== "No action").join(", ")}. Other categories need annotated training data.</p>}
          </div>
        </section>
      </div>
    </section>
  );
}
