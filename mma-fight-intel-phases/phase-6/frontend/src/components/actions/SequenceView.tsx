import { useMemo, useState } from "react";
import { ArrowRight, Download } from "lucide-react";
import { actionsApi, GROUP_COLORS, type Category, type Sequence } from "../../api/actions";
import { useData } from "../../hooks/useData";
import { cn } from "../../lib/format";
import { Spinner } from "../ui";
import { Marker, fmtT, pct } from "./common";

export default function SequenceView({ videoId, jobId, categories, onSeek }: { videoId: string; jobId: string | null; categories: Category[]; onSeek: (t: number) => void }) {
  const [source, setSource] = useState<"model" | "annotations">(jobId ? "model" : "annotations");
  const { data, loading, error } = useData(() => actionsApi.sequences(videoId, source), [videoId, source, jobId]);
  const fam = useMemo(() => Object.fromEntries(categories.map((c) => [c.key, c.family])), [categories]);
  const byFighter = useMemo(() => {
    const m = new Map<string, Sequence[]>();
    data?.sequences.forEach((s) => m.set(s.fighter_name, [...(m.get(s.fighter_name) ?? []), s]));
    return [...m.entries()];
  }, [data]);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 border-b border-ink-700/70 px-4 py-2 text-xs">
        <div className="flex rounded-md border border-ink-600 bg-ink-900 p-0.5">
          {(["model", "annotations"] as const).map((s) => (
            <button key={s} disabled={s === "model" && !jobId} onClick={() => setSource(s)}
              className={cn("rounded px-2.5 py-1 disabled:opacity-40", source === s ? "bg-ink-700 text-white" : "text-ink-400")}>
              {s === "model" ? "From recognised actions" : "From manual annotations"}
            </button>
          ))}
        </div>
        {source === "model" && jobId && (
          <div className="ml-auto flex gap-1">
            <a className="btn-ghost px-2 py-1 text-xs" href={actionsApi.exportUrl(videoId, "sequences", "json")}><Download className="size-3.5" />JSON</a>
            <a className="btn-ghost px-2 py-1 text-xs" href={actionsApi.exportUrl(videoId, "sequences", "csv")}><Download className="size-3.5" />CSV</a>
          </div>
        )}
      </div>
      {loading && !data ? <div className="p-6 text-center"><Spinner /></div> : error ? <p className="p-5 text-sm text-ink-500">{error}</p> : !data?.sequences.length ? (
        <p className="px-5 py-8 text-center text-sm text-ink-500">No sequences yet.</p>
      ) : (
        <div className="max-h-[460px] space-y-5 overflow-y-auto p-4">
          {byFighter.map(([name, seqs]) => (
            <div key={name}>
              <div className="mb-2 text-sm font-medium text-white">{name} <span className="text-xs font-normal text-ink-500">· {seqs.length} sequence{seqs.length > 1 ? "s" : ""}</span></div>
              <div className="space-y-2">
                {seqs.map((s) => (
                  <div key={`${s.fighter_key}-${s.sequence_index}`} className="rounded-md border border-ink-700 bg-ink-900 px-3 py-2">
                    <div className="mb-1.5 flex items-center justify-between text-[11px] text-ink-500 tabular">
                      <button className="hover:text-ink-200" onClick={() => onSeek(s.start_time)}>#{s.sequence_index + 1} · {fmtT(s.start_time)}–{fmtT(s.end_time)}</button>
                      <span>{s.n_actions} action{s.n_actions > 1 ? "s" : ""}</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1">
                      {s.actions.map((a, i) => (
                        <span key={a.action_id} className="flex items-center gap-1">
                          {i > 0 && (
                            <span className="flex flex-col items-center px-0.5 text-[9px] text-ink-500 tabular">
                              <ArrowRight className="size-3" />{s.intervals[i - 1].toFixed(2)}s
                            </span>
                          )}
                          <button onClick={() => onSeek(a.start)} title={`${fmtT(a.start)} · ${pct(a.confidence)} · ${a.source}`}
                            className="flex items-center gap-1.5 rounded border px-2 py-1 text-xs text-ink-100 hover:brightness-125"
                            style={{ borderColor: `${GROUP_COLORS[a.group]}88`, background: `${GROUP_COLORS[a.group]}1f` }}>
                            <Marker family={fam[a.category] ?? "unknown"} group={a.group} size={10} />{a.label}
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <p className="text-[11px] text-ink-500">{data.format_note}</p>
        </div>
      )}
    </div>
  );
}
