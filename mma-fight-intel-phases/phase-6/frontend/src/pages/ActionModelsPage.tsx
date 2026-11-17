import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, CircleCheck, Play, Trash2, TriangleAlert } from "lucide-react";
import { ApiError } from "../api/client";
import { ALGO_LABELS, actionsApi, type ActionSettings, type ModelDetail } from "../api/actions";
import { pct, refreshTaxonomy } from "../components/actions/common";
import { ConfirmDialog, EmptyState, Field, KV, LoadingBlock, PageHeader, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn, formatDate } from "../lib/format";

function ConfusionMatrix({ cm }: { cm: NonNullable<ModelDetail["metrics"]["confusion_matrix"]> }) {
  const max = Math.max(1, ...cm.matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1 text-right text-[10px] font-normal text-ink-500">true ↓ / predicted →</th>
            {cm.display_labels.map((l) => <th key={l} className="max-w-[80px] p-1 align-bottom text-[10px] font-normal text-ink-400"><div className="w-16 truncate" title={l}>{l}</div></th>)}
          </tr>
        </thead>
        <tbody>
          {cm.matrix.map((row, i) => (
            <tr key={cm.labels[i]}>
              <th className="whitespace-nowrap p-1 text-right font-normal text-ink-300">{cm.display_labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="p-0.5">
                  <div className={cn("grid size-12 place-items-center rounded text-xs tabular", i === j ? "text-white" : "text-ink-200")}
                    style={{ background: v ? (i === j ? `rgba(74,222,128,${0.15 + 0.6 * v / max})` : `rgba(239,68,68,${0.12 + 0.6 * v / max})`) : "#131317" }}
                    title={`${cm.display_labels[i]} predicted as ${cm.display_labels[j]}: ${v}`}>{v}</div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Evaluation({ version }: { version: string }) {
  const { data: m, loading } = useData(() => actionsApi.evaluation(version), [version]);
  if (loading && !m) return <LoadingBlock />;
  if (!m) return null;
  const x = m.metrics;
  const ds = m.training_info?.dataset;
  if (x.error) return <p className="flex gap-2 p-5 text-sm text-amber-200"><TriangleAlert className="size-4" />{x.error}</p>;
  return (
    <div className="space-y-5 p-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["Accuracy", pct(x.accuracy, 1), x.accuracy_ci95 ? `95% CI ${pct(x.accuracy_ci95[0])}–${pct(x.accuracy_ci95[1])}` : ""],
          ["Macro F1", pct(x.f1_macro, 1), "action classes"],
          ["Macro precision", pct(x.precision_macro, 1), `weighted ${pct(x.precision_weighted, 1)}`],
          ["Macro recall", pct(x.recall_macro, 1), `weighted ${pct(x.recall_weighted, 1)}`],
        ].map(([k, v, h]) => (
          <div key={k} className="rounded-md border border-ink-700 bg-ink-900 px-3 py-2">
            <div className="label-caps">{k}</div>
            <div className="font-display text-2xl font-semibold text-white tabular">{v}</div>
            <div className="text-[10px] text-ink-500">{h}</div>
          </div>
        ))}
      </div>
      <p className="text-xs text-ink-400">
        Evaluated with <b className="text-ink-200">{x.evaluation?.strategy}</b> on {x.n_evaluated} held-out examples (split by video/fight — no example's video is ever in both training and test).
        Always-predict-the-most-common-class would score {pct(x.majority_class_baseline, 1)}.
      </p>
      {(x.n_evaluated ?? 0) < 200 && (
        <p className="flex gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">
          <TriangleAlert className="size-4 shrink-0" />Only {x.n_evaluated} evaluation examples. With so few, these figures can swing by tens of percentage points — they verify the pipeline, not real-world accuracy.
        </p>
      )}
      <div>
        <div className="label-caps mb-2">Per-action performance</div>
        <table className="w-full text-xs">
          <thead><tr className="text-left text-ink-500">{["Action", "Precision", "Recall", "F1", "Support", `Precision when ≥ ${pct(x.threshold)} confident`].map((h) => <th key={h} className="py-1 pr-3 font-normal">{h}</th>)}</tr></thead>
          <tbody className="divide-y divide-ink-700/50">
            {Object.entries(x.per_class ?? {}).map(([k, c]) => (
              <tr key={k}><td className="py-1.5 pr-3 text-ink-100">{c.label}</td><td className="tabular">{pct(c.precision)}</td><td className="tabular">{pct(c.recall)}</td><td className="tabular">{pct(c.f1)}</td><td className="tabular">{c.support}</td><td className="tabular">{c.precision_at_threshold != null ? `${pct(c.precision_at_threshold)} (n=${c.confident_predictions})` : "—"}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      {x.confusion_matrix && <div><div className="label-caps mb-2">Confusion matrix</div><ConfusionMatrix cm={x.confusion_matrix} /></div>}
      {ds && (
        <dl className="grid gap-x-6 sm:grid-cols-2">
          <KV k="Dataset version" v={<span className="font-mono text-xs">{m.dataset_version}</span>} />
          <KV k="Annotated examples" v={ds.annotations_used} />
          <KV k="Videos / groups" v={`${ds.videos} / ${ds.groups}`} />
          <KV k="Training time" v={`${m.training_info.train_seconds}s on ${m.training_info.device}`} />
          {Object.entries(m.training_info.dropped_classes ?? {}).map(([k, why]) => <KV key={k} k={`Not trained: ${k}`} v={why} />)}
        </dl>
      )}
    </div>
  );
}

export default function ActionModelsPage() {
  const toast = useToast();
  const { data, reload } = useData(actionsApi.modelsStatus, [], (d) => (d?.training_active ? 2000 : undefined));
  const { data: ds, reload: reloadDs } = useData(actionsApi.dataset, []);
  const { data: settings } = useData(actionsApi.settings, []);
  const [s, setS] = useState<ActionSettings>();
  const [algos, setAlgos] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [del, setDel] = useState<string | null>(null);
  useEffect(() => { if (settings) { setS(settings); setAlgos(settings.algorithms); } }, [settings]);
  useEffect(() => { if (!selected && data?.active_model) setSelected(data.active_model.version); }, [data, selected]);
  const wasTraining = data?.training_active;
  useEffect(() => { if (wasTraining === false) { void reloadDs(true); refreshTaxonomy(); } }, [wasTraining, reloadDs]);

  if (!data || !s) return <><PageHeader title="Action Models" /><LoadingBlock /></>;
  const tj = data.training_jobs[0];
  const set = <K extends keyof ActionSettings>(k: K, v: ActionSettings[K]) => setS({ ...s, [k]: v });
  const num = (k: keyof ActionSettings, min: number, max: number, step = 0.05) => (
    <input className="input" type="number" min={min} max={max} step={step} value={(s[k] as number) ?? ""} onChange={(e) => set(k, Number(e.target.value) as never)} />
  );

  const train = async () => {
    setBusy(true);
    try {
      await actionsApi.saveSettings({ ...s, algorithms: algos });
      await actionsApi.train({ algorithms: algos, activate_best: true });
      toast("info", "Training started in the background");
      await reload(true);
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Could not start training");
    } finally {
      setBusy(false);
    }
  };
  const saveSettings = async () => {
    try { await actionsApi.saveSettings({ ...s, algorithms: algos }); toast("success", "Settings saved"); } catch (e) { toast("error", e instanceof Error ? e.message : "Save failed"); }
  };

  return (
    <>
      <Link to="/actions" className="btn-ghost mb-3 -ml-3"><ArrowLeft className="size-4" />Action recognition</Link>
      <PageHeader title="Action Models" subtitle="Train baselines (Random Forest, XGBoost, SVM) and temporal models (Bi-LSTM, TCN) on your verified annotations, evaluated by video." />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Training dataset (from verified annotations)</h2><span className="font-mono text-[11px] text-ink-500">v {ds?.version}</span></div>
            {!ds ? <LoadingBlock /> : (
              <div className="p-5">
                <div className="mb-3 grid grid-cols-3 gap-3 text-center">
                  {[["Examples", ds.annotations_used], ["Videos", ds.videos], ["Video/fight groups", ds.groups]].map(([k, v]) => (
                    <div key={k as string} className="rounded-md border border-ink-700 bg-ink-900 py-2"><div className="font-display text-2xl font-semibold text-white">{v}</div><div className="text-[10px] uppercase tracking-wider text-ink-500">{k}</div></div>
                  ))}
                </div>
                {ds.class_table.length === 0 ? <EmptyState title="No usable annotations yet" body="Open a video in Action Recognition → Annotate, label examples and mark them verified." /> : (
                  <table className="w-full text-xs">
                    <thead><tr className="text-left text-ink-500"><th className="py-1 font-normal">Class</th><th className="font-normal">Examples</th><th className="font-normal">From videos/groups</th><th className="font-normal">Trainable</th></tr></thead>
                    <tbody className="divide-y divide-ink-700/50">
                      {ds.class_table.map((c) => (
                        <tr key={c.key}><td className="py-1.5 text-ink-100">{c.label}</td><td className="tabular">{c.examples}</td><td className="tabular">{c.groups}</td>
                          <td>{c.trainable ? <CircleCheck className="size-4 text-emerald-400" /> : <span className="text-amber-200">needs ≥{ds.requirements.min_examples} examples from ≥{ds.requirements.min_groups} videos</span>}</td></tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {Object.keys(ds.skipped).length > 0 && <p className="mt-2 text-[11px] text-ink-500">Skipped: {Object.entries(ds.skipped).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ")}</p>}
                {ds.warnings.slice(0, 4).map((w) => <p key={w} className="mt-1 flex gap-2 text-[11px] text-amber-200"><TriangleAlert className="size-3 shrink-0" />{w}</p>)}
              </div>
            )}
          </section>

          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Model versions</h2></div>
            {data.models.length === 0 ? <EmptyState title="No trained models yet" body="Train once you have enough verified annotations." /> : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead><tr className="border-b border-ink-700 text-left">{["Model", "Accuracy", "Macro F1", "Evaluation", "Created", ""].map((h) => <th key={h} className="label-caps px-4 py-2.5 font-semibold">{h}</th>)}</tr></thead>
                  <tbody className="divide-y divide-ink-700/50">
                    {data.models.map((m) => (
                      <tr key={m.version} className={cn("cursor-pointer hover:bg-ink-800/50", selected === m.version && "bg-ink-800/60")} onClick={() => setSelected(m.version)}>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2"><span className="text-ink-100">{ALGO_LABELS[m.algorithm] ?? m.algorithm}</span>{m.is_active && <span className="rounded bg-emerald-500/15 px-1.5 py-px text-[10px] font-medium uppercase text-emerald-300">active</span>}</div>
                          <div className="font-mono text-[10px] text-ink-500">{m.version}</div>
                        </td>
                        <td className="px-4 tabular">{pct(m.accuracy, 1)}</td>
                        <td className="px-4 tabular">{pct(m.f1_macro, 1)}</td>
                        <td className="px-4 text-xs text-ink-400">{m.error ?? `${m.evaluation_strategy ?? "—"} · n=${m.n_evaluated ?? 0}`}</td>
                        <td className="px-4 text-xs text-ink-400">{formatDate(m.created_at)}</td>
                        <td className="px-4 text-right" onClick={(e) => e.stopPropagation()}>
                          {!m.is_active && <button className="btn-ghost px-2 py-1 text-xs" onClick={async () => { await actionsApi.activate(m.version); refreshTaxonomy(); void reload(true); }}>Activate</button>}
                          <button className="btn-ghost p-1.5 hover:text-blood-400" onClick={() => setDel(m.version)}><Trash2 className="size-3.5" /></button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {selected && (
            <section className="panel">
              <div className="panel-header"><h2 className="label-caps">Evaluation · <span className="font-mono normal-case tracking-normal">{selected}</span></h2></div>
              <Evaluation version={selected} />
            </section>
          )}
        </div>

        <aside className="space-y-4">
          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Train</h2></div>
            <div className="space-y-3 px-5 py-4 text-sm">
              <div className="space-y-1.5">
                {Object.entries(ALGO_LABELS).map(([k, l]) => (
                  <label key={k} className="flex items-center gap-2 text-ink-200">
                    <input type="checkbox" className="accent-[#e11d2e]" checked={algos.includes(k)} onChange={(e) => setAlgos(e.target.checked ? [...algos, k] : algos.filter((a) => a !== k))} />
                    {l} <span className="text-[11px] text-ink-500">{["lstm", "tcn"].includes(k) ? "temporal · PyTorch" : "baseline · window statistics"}</span>
                  </label>
                ))}
              </div>
              <button className="btn-primary w-full" disabled={busy || data.training_active || !algos.length} onClick={train}><Play className="size-4" />Train & evaluate</button>
              <p className="text-[11px] text-ink-500">The best model by macro F1 is activated automatically. Training runs in the background.</p>
              {tj && (
                <div className="rounded-md border border-ink-700 bg-ink-900 p-3 text-xs">
                  <div className="flex items-center justify-between"><StatusBadge status={tj.status} /><span className="text-ink-500">{formatDate(tj.created_at)}</span></div>
                  {tj.status === "processing" || tj.status === "queued" ? (<><div className="mt-2 text-ink-300">{tj.current_stage ?? "Queued"}</div><ProgressBar value={tj.progress} tone="sky" className="mt-1" /></>) : null}
                  {tj.error_message && <p className="mt-2 text-blood-300">{tj.error_message}</p>}
                  {tj.results && (
                    <table className="mt-2 w-full">
                      <tbody>{Object.entries(tj.results.comparison).map(([a, r]) => (
                        <tr key={a} className={r.version === tj.results?.best ? "text-emerald-300" : "text-ink-300"}><td className="py-0.5">{ALGO_LABELS[a]}</td><td className="text-right tabular">acc {pct(r.accuracy)}</td><td className="text-right tabular">F1 {pct(r.f1_macro)}</td></tr>
                      ))}</tbody>
                    </table>
                  )}
                  {tj.warnings?.slice(-2).map((w) => <p key={w} className="mt-2 text-amber-200">{w}</p>)}
                </div>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Settings</h2><button className="btn-ghost px-2 py-1 text-xs" onClick={saveSettings}>Save</button></div>
            <div className="grid grid-cols-2 gap-3 px-5 py-4">
              <Field label="Confidence threshold" hint="Below = not reported">{num("confidence_threshold", 0.05, 0.99)}</Field>
              <Field label="Motion gate (torso/s)" hint="Idle windows skipped">{num("motion_gate", 0, 50, 0.5)}</Field>
              <Field label="'Unknown' above (torso/s)">{num("unknown_motion", 0, 100, 0.5)}</Field>
              <Field label="Window hop (fraction)">{num("hop_fraction", 0.05, 1)}</Field>
              <Field label="Sequence gap (s)">{num("sequence_gap", 0.2, 30, 0.1)}</Field>
              <Field label="De-dup IoU">{num("nms_iou", 0, 1)}</Field>
              <Field label="Sequence length (steps)">{num("seq_len", 6, 64, 1)}</Field>
              <Field label="Context (s)">{num("context", 0, 1)}</Field>
              <Field label="Min examples / class">{num("min_examples", 1, 1000, 1)}</Field>
              <Field label="Min videos / class">{num("min_groups", 1, 100, 1)}</Field>
              <Field label="Augment copies">{num("augment", 0, 20, 1)}</Field>
              <Field label="Epochs (LSTM/TCN)">{num("epochs", 5, 500, 5)}</Field>
              <Field label="Device" className="col-span-2">
                <select className="input" value={s.device} onChange={(e) => set("device", e.target.value)}>{["auto", "cpu", "cuda", "mps"].map((d) => <option key={d}>{d}</option>)}</select>
              </Field>
              <label className="col-span-2 flex items-center gap-2 text-xs text-ink-300"><input type="checkbox" className="accent-[#e11d2e]" checked={s.fighters_only} onChange={(e) => set("fighters_only", e.target.checked)} />Analyse fighter tracks only</label>
              <label className="col-span-2 flex items-center gap-2 text-xs text-ink-300"><input type="checkbox" className="accent-[#e11d2e]" checked={s.include_draft_annotations} onChange={(e) => set("include_draft_annotations", e.target.checked)} />Also train on draft annotations</label>
            </div>
          </section>
        </aside>
      </div>
      <ConfirmDialog open={!!del} title="Delete model" onClose={() => setDel(null)} onConfirm={async () => { await actionsApi.deleteModel(del!); setDel(null); if (selected === del) setSelected(null); void reload(true); }}
        body={<>Delete model <span className="font-mono">{del}</span>? Results already produced with it are kept.</>} />
      {busy && <Spinner className="hidden" />}
    </>
  );
}
