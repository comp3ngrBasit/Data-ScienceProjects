import { useMemo, useState } from "react";
import { Check, Pencil, X } from "lucide-react";
import { actionsApi, type Category, type Lane, type RecognizedAction } from "../../api/actions";
import { useToast } from "../../context/ToastContext";
import { cn } from "../../lib/format";
import { Marker, fmtT, pct } from "./common";

function ConfidenceCell({ a }: { a: RecognizedAction }) {
  return (
    <div className="text-right tabular">
      <div className={cn("font-medium", a.low_confidence ? "text-amber-200" : "text-ink-100")}>{pct(a.confidence)}</div>
      <div className="text-[10px] text-ink-500" title="Precision of this class on held-out videos when the model was at least this confident">
        {a.validated_precision != null ? `validated ${pct(a.validated_precision)} (n=${a.validated_support})` : "not validated"}
      </div>
    </div>
  );
}

export default function ActionHistory({ lanes, categories, onSeek, onChanged }: {
  lanes: Lane[];
  categories: Category[];
  onSeek: (t: number) => void;
  onChanged: () => void;
}) {
  const toast = useToast();
  const withActions = lanes.filter((l) => l.actions.length);
  const [laneKey, setLaneKey] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [hideLow, setHideLow] = useState(false);
  const lane = withActions.find((l) => l.key === laneKey) ?? withActions[0];
  const fam = useMemo(() => Object.fromEntries(categories.map((c) => [c.key, c.family])), [categories]);
  const grouped = useMemo(() => {
    const g: Record<string, Category[]> = {};
    categories.forEach((c) => (g[c.group] = [...(g[c.group] ?? []), c]));
    return g;
  }, [categories]);

  const review = async (a: RecognizedAction, decision: "confirm" | "correct" | "reject", category?: string) => {
    try {
      await actionsApi.review(a.id, decision, category);
      toast("success", decision === "reject" ? "Marked as wrong" : decision === "confirm" ? "Confirmed — added to training data" : "Corrected — added to training data");
      setEditing(null);
      onChanged();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Review failed");
    }
  };

  if (!withActions.length) return <p className="px-5 py-8 text-center text-sm text-ink-500">No actions were recognised. See the analysis notes for why windows were skipped.</p>;
  const rows = lane.actions.filter((a) => !hideLow || !a.low_confidence);
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 border-b border-ink-700/70 px-4 py-2">
        {withActions.map((l) => (
          <button key={l.key} onClick={() => setLaneKey(l.key)}
            className={cn("rounded-full border px-3 py-1 text-xs", l.key === lane.key ? "border-blood-500 bg-blood-900/30 text-white" : "border-ink-600 text-ink-300 hover:text-white")}>
            {l.name} <span className="text-ink-400 tabular">{l.actions.length}</span>
          </button>
        ))}
        <label className="ml-auto flex items-center gap-1.5 text-xs text-ink-400">
          <input type="checkbox" className="accent-[#e11d2e]" checked={hideLow} onChange={(e) => setHideLow(e.target.checked)} />Hide low-confidence
        </label>
      </div>
      <div className="max-h-[420px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-ink-850">
            <tr className="text-left">{["Time", "Action", "Track", "Confidence", "Review"].map((h) => <th key={h} className="label-caps px-3 py-2 font-semibold">{h}</th>)}</tr>
          </thead>
          <tbody className="divide-y divide-ink-700/50">
            {rows.map((a) => (
              <tr key={a.id} className={cn("hover:bg-ink-800/50", a.review_status === "rejected" && "opacity-40")}>
                <td className="px-3 py-2"><button className="font-mono text-xs text-ink-200 hover:text-white hover:underline" onClick={() => onSeek(a.start_time)}>{fmtT(a.start_time)}</button><div className="text-[10px] text-ink-500 tabular">{a.duration.toFixed(2)}s</div></td>
                <td className="px-3 py-2">
                  {editing === a.id ? (
                    <select autoFocus className="input py-1 text-xs" defaultValue="" onChange={(e) => e.target.value && review(a, "correct", e.target.value)} onBlur={() => setEditing(null)}>
                      <option value="">Correct to…</option>
                      {Object.entries(grouped).map(([g, cs]) => (
                        <optgroup key={g} label={g}>{cs.filter((c) => c.key !== "unknown").map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}</optgroup>
                      ))}
                    </select>
                  ) : (
                    <div className="flex items-center gap-2">
                      <Marker family={fam[a.category] ?? "unknown"} group={a.group} />
                      <span className={cn("text-ink-100", a.category === "unknown" && "italic text-amber-200")}>{a.label}</span>
                      {a.review_status === "corrected" && <span className="text-[10px] text-ink-500">(model said {a.model_category.replace(/_/g, " ")})</span>}
                    </div>
                  )}
                  {a.runner_up_label && a.category !== "unknown" && <div className="text-[10px] text-ink-500">runner-up {a.runner_up_label} {pct(a.runner_up_confidence)}</div>}
                </td>
                <td className="px-3 py-2 text-xs text-ink-400 tabular">ID {a.track_id}{a.limb && <div className="text-[10px]">{a.limb.replace("_", " ")}</div>}</td>
                <td className="px-3 py-2"><ConfidenceCell a={a} /></td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-0.5">
                    {a.review_status !== "unreviewed" && <span className="mr-1 text-[10px] uppercase text-ink-500">{a.review_status}</span>}
                    <button title="Correct" className={cn("btn-ghost p-1", a.review_status === "confirmed" && "text-emerald-400")} onClick={() => review(a, "confirm")} disabled={a.category === "unknown"}><Check className="size-3.5" /></button>
                    <button title="Wrong label — choose the right one" className="btn-ghost p-1" onClick={() => setEditing(a.id)}><Pencil className="size-3.5" /></button>
                    <button title="Not an action / false detection" className={cn("btn-ghost p-1 hover:text-blood-400", a.review_status === "rejected" && "text-blood-400")} onClick={() => review(a, "reject")}><X className="size-3.5" /></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
