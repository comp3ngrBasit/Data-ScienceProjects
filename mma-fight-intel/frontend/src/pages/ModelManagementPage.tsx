import { useState } from "react";
import { Link } from "react-router-dom";
import { CheckCircle2, History, Play, RotateCcw, ShieldCheck, XCircle } from "lucide-react";
import { ApiError } from "../api/client";
import { actionsApi } from "../api/actions";
import { platformApi, type ModelVersion, type Validation } from "../api/platform";
import { predictionsApi } from "../api/predictions";
import { pct } from "../components/platform/FightDashboard";
import { ConfirmDialog, ErrorBlock, LoadingBlock, PageHeader, ProgressBar, Spinner } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn } from "../lib/format";

const METRIC_COLS: Record<"action" | "prediction", [string, string, (v: number | null | undefined) => string][]> = {
  action: [["accuracy", "Accuracy", (v) => pct(v, 1)], ["f1_macro", "Macro F1", (v) => v?.toFixed(3) ?? "—"], ["precision_macro", "Precision", (v) => v?.toFixed(3) ?? "—"], ["recall_macro", "Recall", (v) => v?.toFixed(3) ?? "—"], ["n_evaluated", "Held-out n", (v) => String(v ?? "—")]],
  prediction: [["top1_accuracy", "Top-1", (v) => pct(v, 1)], ["top3_accuracy", "Top-3", (v) => pct(v, 1)], ["log_loss", "Log loss", (v) => v?.toFixed(3) ?? "—"], ["brier", "Brier", (v) => v?.toFixed(3) ?? "—"], ["ece", "ECE", (v) => v?.toFixed(3) ?? "—"], ["n", "Held-out n", (v) => String(v ?? "—")]],
};

function ValidationBadge({ v }: { v: Validation | null }) {
  if (!v) return <span className="text-[11px] text-ink-500">not validated</span>;
  return (
    <span className={cn("inline-flex items-center gap-1 text-[11px]", v.passed ? "text-emerald-300" : "text-amber-300")} title={v.checks.map((c) => `${c.passed ? "✓" : "✗"} ${c.check}: ${c.detail}`).join("\n")}>
      {v.passed ? <CheckCircle2 className="size-3.5" /> : <XCircle className="size-3.5" />}{v.passed ? "passed checks" : "failed checks"}
    </span>
  );
}

export default function ModelManagementPage() {
  const toast = useToast();
  const [kind, setKind] = useState<"prediction" | "action">("prediction");
  const [cmp, setCmp] = useState<string[]>([]);
  const [force, setForce] = useState<{ m: ModelVersion; reason: "manual" | "rollback"; validation?: Validation } | null>(null);
  const actJobs = useData(actionsApi.modelsStatus, [], (d) => (d?.training_active ? 2000 : 0));
  const predJobs = useData(predictionsApi.jobs, [], (d) => (d?.some((j) => j.status === "queued" || j.status === "processing") ? 2000 : 0));
  const running = kind === "action" ? actJobs.data?.training_active : predJobs.data?.some((j) => j.status === "queued" || j.status === "processing");
  const vers = useData(platformApi.versions, [running ? 1 : 0]);
  const ds = useData(platformApi.datasets, []);
  if (vers.error) return <ErrorBlock message={vers.error} onRetry={() => vers.reload()} />;
  if (!vers.data) return <LoadingBlock />;
  const list = vers.data[kind];
  const active = list.find((m) => m.is_active);
  const deps = vers.data.deployments.filter((d) => d.kind === kind);
  const previous = deps.find((d) => d.model_id === active?.id)?.previous_model_id;
  const cmpModels = list.filter((m) => cmp.includes(m.id));

  const train = async () => {
    try { await platformApi.train(kind); toast("info", "Training started — the new model goes live only if it passes the validation checks"); kind === "action" ? actJobs.reload(true) : predJobs.reload(true); }
    catch (e) { toast("error", e instanceof ApiError ? e.message : "Request failed"); }
  };
  const activate = async (m: ModelVersion, reason: "manual" | "rollback", f = false) => {
    try {
      const r = await platformApi.activate(kind, m.id, { force: f, reason });
      toast("success", `${m.name} is now active${r.validation.passed ? "" : " (validation checks were overridden)"}`);
      setForce(null);
      vers.reload(true);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setForce({ m, reason, validation: ((e.body as { detail?: { validation?: Validation } })?.detail)?.validation });
      else toast("error", e instanceof ApiError ? e.message : "Request failed");
    }
  };
  const job = kind === "action" ? actJobs.data?.training_jobs?.[0] : predJobs.data?.[0];
  return (
    <>
      <PageHeader title="Model Management" subtitle="Versions, training, validation-gated activation and rollback for the action-recognition and next-move models"
        actions={
          <div className="flex gap-2">
            <div className="flex rounded-md border border-ink-600 bg-ink-900 p-0.5 text-xs">
              {(["prediction", "action"] as const).map((k) => <button key={k} onClick={() => { setKind(k); setCmp([]); }} className={cn("rounded px-3 py-1.5", kind === k ? "bg-ink-700 text-white" : "text-ink-400")}>{k === "prediction" ? "Next-move models" : "Action-recognition models"}</button>)}
            </div>
            <button className="btn-primary" onClick={train} disabled={!!running}>{running ? <Spinner className="size-4" /> : <Play className="size-4" />}Start training</button>
          </div>
        } />
      {job && (
        <div className="mb-4 panel px-5 py-3 text-xs">
          <div className="flex justify-between text-ink-300"><span>Latest training: {job.status} · {job.current_stage ?? ""}</span><span className="tabular">{Math.round((job.progress ?? 0) * 100)}%</span></div>
          {(job.status === "queued" || job.status === "processing") && <ProgressBar value={job.progress} tone="sky" className="mt-1" />}
          {job.error_message && <div className="mt-1 text-blood-300">{job.error_message}</div>}
        </div>
      )}
      <div className="grid gap-4 xl:grid-cols-3">
        <section className="panel">
          <div className="panel-header"><h3 className="label-caps flex items-center gap-2"><ShieldCheck className="size-4" />Active model</h3></div>
          {active ? (
            <div className="space-y-2 px-5 py-4 text-sm">
              <div className="font-mono text-white">{active.name}</div>
              <div className="text-xs text-ink-400">{active.architecture} · trained {new Date(active.created_at).toLocaleString()} · dataset {active.dataset_version}</div>
              <ValidationBadge v={active.validation} />
              {active.validation && <ul className="space-y-0.5 text-[11px]">{active.validation.checks.map((c) => <li key={c.check} className={c.passed ? "text-ink-300" : "text-amber-300"}>{c.passed ? "✓" : "✗"} {c.detail}</li>)}</ul>}
              <div className="text-[11px] text-ink-500">File: {active.path}</div>
              {previous && <button className="btn-secondary h-8 text-xs" onClick={() => { const pm = list.find((m) => m.id === previous); if (pm) void activate(pm, "rollback"); }}><RotateCcw className="size-3.5" />Roll back to previous ({list.find((m) => m.id === previous)?.name ?? previous.slice(0, 8)})</button>}
            </div>
          ) : <p className="px-5 py-6 text-sm text-ink-500">No active {kind === "action" ? "action-recognition" : "next-move"} model.</p>}
        </section>
        <section className="panel">
          <div className="panel-header"><h3 className="label-caps">Training dataset</h3></div>
          <div className="space-y-1 px-5 py-4 text-sm">
            {kind === "prediction" ? (ds.data ? (
              <>
                <div className="flex justify-between"><span className="text-ink-400">Next-action examples</span><span className="tabular">{ds.data.prediction.examples}</span></div>
                <div className="flex justify-between"><span className="text-ink-400">Unique fights / fighters</span><span className="tabular">{ds.data.prediction.fights} / {ds.data.prediction.fighters}</span></div>
                <div className="flex justify-between"><span className="text-ink-400">Sources</span><span className="text-xs">{Object.entries(ds.data.prediction.sources).map(([k, v]) => `${v} ${k}`).join(" · ")}</span></div>
                <div className="pt-1 text-xs text-ink-400">{Object.entries(ds.data.prediction.targets).slice(0, 8).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>
                <Link to="/predictions" className="text-xs text-ink-300 hover:text-white">Import sequences / settings →</Link>
              </>
            ) : <LoadingBlock />) : (
              <><p className="text-xs text-ink-400">{ds.data?.action.note}</p><Link to="/actions/models" className="text-xs text-ink-300 hover:text-white">Open dataset preview & annotation →</Link></>
            )}
          </div>
        </section>
        <section className="panel">
          <div className="panel-header"><h3 className="label-caps flex items-center gap-2"><History className="size-4" />Deployment history</h3></div>
          <div className="max-h-60 divide-y divide-ink-700/60 overflow-auto">
            {deps.length ? deps.map((d) => (
              <div key={d.id} className="px-5 py-2 text-xs">
                <div className="flex justify-between"><span className="text-ink-100">{list.find((m) => m.id === d.model_id)?.name ?? d.model_id.slice(0, 10)}</span><span className="text-ink-500">{new Date(d.created_at).toLocaleString()}</span></div>
                <div className="text-ink-400">{d.reason}{d.forced ? " · validation overridden" : ""}{d.previous_model_id ? ` · replaced ${list.find((m) => m.id === d.previous_model_id)?.name ?? d.previous_model_id.slice(0, 10)}` : ""}</div>
              </div>
            )) : <p className="px-5 py-6 text-sm text-ink-500">No activations recorded yet.</p>}
          </div>
        </section>
      </div>

      <section className="panel mt-4">
        <div className="panel-header"><h3 className="label-caps">Model versions ({list.length})</h3><span className="text-xs text-ink-500">tick two or more to compare · metrics are held-out</span></div>
        {list.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[11px] uppercase tracking-wide text-ink-500"><tr className="border-b border-ink-700"><th className="px-5 py-2">Compare</th><th>Model</th><th>Trained</th><th>Dataset</th>{METRIC_COLS[kind].map(([, l]) => <th key={l} className="px-2">{l}</th>)}<th>Validation</th><th className="px-5 text-right">Status</th></tr></thead>
              <tbody>
                {list.map((m) => (
                  <tr key={m.id} className={cn("border-b border-ink-700/50", m.is_active && "bg-emerald-500/5")}>
                    <td className="px-5"><input type="checkbox" aria-label={`Compare ${m.name}`} checked={cmp.includes(m.id)} onChange={() => setCmp(cmp.includes(m.id) ? cmp.filter((x) => x !== m.id) : [...cmp, m.id])} /></td>
                    <td className="py-2 font-mono text-xs text-white">{m.name}{m.experimental && <span className="ml-1 font-sans text-[10px] uppercase text-amber-300">experimental</span>}</td>
                    <td className="text-xs text-ink-400">{new Date(m.created_at).toLocaleString()}</td><td className="font-mono text-xs text-ink-400">{m.dataset_version}</td>
                    {METRIC_COLS[kind].map(([k, l, f]) => <td key={l} className="px-2 tabular">{f(m.metrics[k])}</td>)}
                    <td><ValidationBadge v={m.validation} /></td>
                    <td className="px-5 text-right">{m.is_active ? <span className="text-xs text-emerald-300">active</span> : <button className="btn-ghost h-7 text-xs" onClick={() => activate(m, "manual")}>Activate</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="px-5 py-6 text-sm text-ink-500">No models trained yet. Click “Start training”.</p>}
      </section>

      {cmpModels.length >= 2 && (
        <section className="panel mt-4">
          <div className="panel-header"><h3 className="label-caps">Version comparison</h3><button className="text-xs text-ink-400 hover:text-white" onClick={() => setCmp([])}>Clear</button></div>
          <table className="w-full text-left text-sm">
            <thead className="text-[11px] uppercase text-ink-500"><tr><th className="px-5 py-2">Metric</th>{cmpModels.map((m) => <th key={m.id} className="font-mono normal-case">{m.name}</th>)}</tr></thead>
            <tbody>
              {METRIC_COLS[kind].map(([k, l, f]) => {
                const vals = cmpModels.map((m) => m.metrics[k]).filter((v): v is number => v != null);
                const lowerBetter = ["log_loss", "brier", "ece"].includes(k);
                const best = vals.length ? (lowerBetter ? Math.min(...vals) : Math.max(...vals)) : null;
                return <tr key={k} className="border-t border-ink-700/50"><td className="px-5 py-1.5 text-ink-400">{l}</td>{cmpModels.map((m) => <td key={m.id} className={cn("tabular", m.metrics[k] === best && k !== "n" && k !== "n_evaluated" && "text-emerald-300")}>{f(m.metrics[k])}</td>)}</tr>;
              })}
              <tr className="border-t border-ink-700/50"><td className="px-5 py-1.5 text-ink-400">Dataset</td>{cmpModels.map((m) => <td key={m.id} className="font-mono text-xs">{m.dataset_version}</td>)}</tr>
            </tbody>
          </table>
          <p className="px-5 py-2 text-[11px] text-ink-500">Versions trained on different dataset versions were evaluated on different held-out data; compare with care.</p>
        </section>
      )}

      <ConfirmDialog open={!!force} title="Model failed validation" confirmLabel="Activate anyway"
        body={<div className="space-y-2 text-sm"><p>{force?.m.name} did not pass the deployment checks:</p><ul className="text-xs">{force?.validation?.checks.map((c) => <li key={c.check} className={c.passed ? "text-ink-300" : "text-amber-300"}>{c.passed ? "✓" : "✗"} {c.detail}</li>)}</ul><p className="text-xs text-ink-400">Activating anyway is recorded in the deployment history.</p></div>}
        onConfirm={() => force && activate(force.m, force.reason, true)} onClose={() => setForce(null)} />
    </>
  );
}
