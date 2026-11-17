import { useEffect, useMemo, useRef, useState } from "react";
import { Pencil, Trash2, Upload } from "lucide-react";
import { actionsApi, type Annotation, type Category } from "../../api/actions";
import type { Fighter } from "../../api/types";
import type { Track } from "../../api/vision";
import { useToast } from "../../context/ToastContext";
import { useData } from "../../hooks/useData";
import { cn } from "../../lib/format";
import { Field, Spinner } from "../ui";
import { Marker, fmtT } from "./common";

const REACTIONS = ["", "defensive_block", "slip", "duck", "parry", "backward_evasion", "counterattack", "high_guard", "takedown_defense", "sprawl"];

/**
 * Manual labelling: play/pause in the player, press "Set start" (I) and "Set end" (O) at the current
 * playhead, choose action + fighter, save. Annotations can be edited, re-timed, verified or deleted.
 */
export default function AnnotationPanel({ videoId, time, fighters, tracks, categories, onSeek, onChanged, version }: {
  videoId: string;
  time: number;
  fighters: Fighter[];
  tracks: Track[];
  categories: Category[];
  onSeek: (t: number) => void;
  onChanged: () => void;
  version: number;
}) {
  const toast = useToast();
  const { data, reload } = useData(() => actionsApi.annotations(videoId), [videoId, version]);
  const [editId, setEditId] = useState<string | null>(null);
  const [start, setStart] = useState<number | null>(null);
  const [end, setEnd] = useState<number | null>(null);
  const [category, setCategory] = useState("");
  const [fighterId, setFighterId] = useState("");
  const [trackId, setTrackId] = useState<string>("");
  const [conf, setConf] = useState(1);
  const [reaction, setReaction] = useState("");
  const [statusV, setStatusV] = useState<Annotation["status"]>("verified");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const timeRef = useRef(time);
  timeRef.current = time;

  const grouped = useMemo(() => {
    const g: Record<string, Category[]> = {};
    categories.forEach((c) => (g[c.group] = [...(g[c.group] ?? []), c]));
    return g;
  }, [categories]);
  const fam = useMemo(() => Object.fromEntries(categories.map((c) => [c.key, c.family])), [categories]);
  const fighterTracks = tracks.filter((t) => t.status !== "fragment");

  // keyboard: I = in, O = out
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "i" || e.key === "I") setStart(Number(timeRef.current.toFixed(3)));
      if (e.key === "o" || e.key === "O") setEnd(Number(timeRef.current.toFixed(3)));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const reset = () => {
    setEditId(null); setStart(null); setEnd(null); setCategory(""); setConf(1); setReaction(""); setNotes(""); setStatusV("verified");
  };
  const loadForEdit = (a: Annotation) => {
    setEditId(a.id); setStart(a.start_time); setEnd(a.end_time); setCategory(a.category); setFighterId(a.fighter_id ?? "");
    setTrackId(a.track_id != null ? String(a.track_id) : ""); setConf(a.confidence); setReaction(a.opponent_reaction ?? "");
    setStatusV(a.status); setNotes(a.notes ?? ""); onSeek(a.start_time);
  };

  const save = async () => {
    if (start == null || end == null || !category) return toast("error", "Set a start, an end and an action category");
    if (end <= start) return toast("error", "End must be after start");
    setBusy(true);
    const body = { category, start_time: start, end_time: end, fighter_id: fighterId || null,
      track_id: trackId ? Number(trackId) : null, confidence: conf, opponent_reaction: reaction || null, status: statusV, notes: notes || null };
    try {
      if (editId) await actionsApi.updateAnnotation(editId, body);
      else await actionsApi.createAnnotation(videoId, body);
      toast("success", editId ? "Annotation updated" : "Annotation saved");
      reset();
      void reload(true);
      onChanged();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  };
  const remove = async (id: string) => {
    await actionsApi.deleteAnnotation(id).catch(() => undefined);
    if (editId === id) reset();
    void reload(true);
    onChanged();
  };
  const setMeta = async (m: { fully_annotated?: boolean; split?: string }) => {
    await actionsApi.setMeta(videoId, m).catch(() => undefined);
    void reload(true);
  };
  const importFile = async (f: File) => {
    const fd = new FormData();
    fd.append("file", f);
    const isFrames = window.confirm("Are start/end given in FRAMES? (OK = frames, Cancel = seconds)");
    fd.append("units", isFrames ? "frames" : "seconds");
    const boxing = f.name.toLowerCase().endsWith(".xlsx") && window.confirm("Use the BoxingVI label mapping (orthodox stance: jab→left jab, cross→right cross, lead hook→left hook …)?");
    fd.append("label_map", boxing ? "boxingvi_orthodox" : "none");
    if (fighterId) fd.append("fighter_id", fighterId);
    try {
      const r = await actionsApi.importFile(videoId, fd);
      toast(r.error_count ? "info" : "success", `Imported ${r.created} annotation(s)${r.error_count ? `, ${r.error_count} row(s) skipped: ${r.errors[0]}` : ""}`);
      void reload(true);
      onChanged();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Import failed");
    }
  };

  const meta = data?.meta;
  return (
    <div className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-white">{editId ? "Edit annotation" : "New annotation"}</h3>
          <span className="font-mono text-xs text-ink-400">playhead {fmtT(time)}</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <button className="btn-secondary w-full py-1.5 text-xs" onClick={() => setStart(Number(time.toFixed(3)))}>Set start (I)</button>
            <input className="input mt-1 py-1 font-mono text-xs" type="number" step="0.01" min={0} value={start ?? ""} placeholder="start s" onChange={(e) => setStart(e.target.value === "" ? null : Number(e.target.value))} />
          </div>
          <div>
            <button className="btn-secondary w-full py-1.5 text-xs" onClick={() => setEnd(Number(time.toFixed(3)))}>Set end (O)</button>
            <input className="input mt-1 py-1 font-mono text-xs" type="number" step="0.01" min={0} value={end ?? ""} placeholder="end s" onChange={(e) => setEnd(e.target.value === "" ? null : Number(e.target.value))} />
          </div>
        </div>
        {start != null && end != null && end > start && <p className="text-[11px] text-ink-500 tabular">Duration {(end - start).toFixed(2)} s · <button className="underline" onClick={() => onSeek(start)}>preview from start</button></p>}
        <Field label="Action category">
          <select className="input" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">Choose…</option>
            {Object.entries(grouped).map(([g, cs]) => (
              <optgroup key={g} label={g[0].toUpperCase() + g.slice(1)}>{cs.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}</optgroup>
            ))}
          </select>
        </Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Fighter">
            <select className="input" value={fighterId} onChange={(e) => setFighterId(e.target.value)}>
              <option value="">— none —</option>
              {fighters.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
            </select>
          </Field>
          <Field label="Track ID" hint="Who performs it in this video">
            <select className="input" value={trackId} onChange={(e) => setTrackId(e.target.value)}>
              <option value="">auto</option>
              {fighterTracks.map((t) => <option key={t.track_id} value={t.track_id}>ID {t.track_id}{t.assigned_fighter ? ` · ${t.assigned_fighter.name}` : t.role !== "unreviewed" ? ` · ${t.role}` : ""}</option>)}
            </select>
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Field label={`Your certainty: ${Math.round(conf * 100)}%`}>
            <input type="range" min={0} max={1} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} className="w-full accent-[#e11d2e]" />
          </Field>
          <Field label="Opponent reaction (optional)">
            <select className="input" value={reaction} onChange={(e) => setReaction(e.target.value)}>
              {REACTIONS.map((r) => <option key={r} value={r}>{r ? r.replace(/_/g, " ") : "—"}</option>)}
            </select>
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Status">
            <select className="input" value={statusV} onChange={(e) => setStatusV(e.target.value as Annotation["status"])}>
              <option value="verified">Verified (used for training)</option>
              <option value="draft">Draft</option>
              <option value="needs_review">Needs review</option>
              <option value="rejected">Rejected</option>
            </select>
          </Field>
          <Field label="Notes"><input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} /></Field>
        </div>
        <div className="flex gap-2">
          <button className="btn-primary flex-1" onClick={save} disabled={busy}>{busy && <Spinner className="text-white" />}{editId ? "Update annotation" : "Save annotation"}</button>
          {(editId || start != null) && <button className="btn-secondary" onClick={reset}>Clear</button>}
        </div>
        <div className="space-y-2 rounded-md border border-ink-700 bg-ink-900 p-3 text-xs">
          <div className="label-caps">Dataset settings for this video</div>
          <label className="flex items-start gap-2 text-ink-300">
            <input type="checkbox" className="mt-0.5 accent-[#e11d2e]" checked={meta?.fully_annotated ?? false} onChange={(e) => setMeta({ fully_annotated: e.target.checked })} />
            <span>Every action of the labelled fighter(s) is annotated — unlabelled moments can be used as “no action” examples.</span>
          </label>
          <label className="flex items-center gap-2 text-ink-300">Split
            <select className="input w-auto py-1 text-xs" value={meta?.split ?? "auto"} onChange={(e) => setMeta({ split: e.target.value })}>
              <option value="auto">auto (cross-validation by video)</option>
              <option value="train">training only</option>
              <option value="test">hold-out test</option>
              <option value="exclude">exclude from training</option>
            </select>
          </label>
          <button className="btn-ghost px-2 py-1 text-xs" onClick={() => fileRef.current?.click()}><Upload className="size-3.5" />Import CSV / XLSX (BoxingVI format supported)</button>
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls,.tsv,.txt" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void importFile(f); e.target.value = ""; }} />
        </div>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between text-xs text-ink-400">
          <span className="label-caps">Annotations ({data?.items.length ?? 0})</span>
          <span className="tabular">{Object.entries(data?.counts ?? {}).length} categor{Object.entries(data?.counts ?? {}).length === 1 ? "y" : "ies"}</span>
        </div>
        <div className="max-h-[480px] divide-y divide-ink-700/50 overflow-y-auto rounded-md border border-ink-700">
          {!data?.items.length && <p className="px-4 py-8 text-center text-xs text-ink-500">No annotations yet. Pause on an action, press I at its start and O at its end, choose a category and save.</p>}
          {data?.items.map((a) => (
            <div key={a.id} className={cn("flex items-center gap-2 px-3 py-2 text-xs", editId === a.id && "bg-ink-800")}>
              <Marker family={fam[a.category] ?? "unknown"} group={a.group} size={11} />
              <button className="font-mono text-ink-300 hover:text-white" onClick={() => onSeek(a.start_time)}>{fmtT(a.start_time)}</button>
              <span className="min-w-0 flex-1 truncate text-ink-100">{a.label}<span className="text-ink-500"> · {a.duration.toFixed(2)}s{a.fighter_name ? ` · ${a.fighter_name}` : a.track_id != null ? ` · ID ${a.track_id}` : ""}</span></span>
              <span className={cn("rounded px-1.5 py-px text-[10px] uppercase", a.status === "verified" ? "bg-emerald-500/10 text-emerald-300" : a.status === "rejected" ? "bg-blood-900/40 text-blood-300" : "bg-ink-700 text-ink-300")}>{a.status.replace("_", " ")}</span>
              {a.source !== "manual" && <span className="text-[10px] text-ink-500">{a.source}</span>}
              <button className="btn-ghost p-1" title="Edit" onClick={() => loadForEdit(a)}><Pencil className="size-3.5" /></button>
              <button className="btn-ghost p-1 hover:text-blood-400" title="Delete" onClick={() => remove(a.id)}><Trash2 className="size-3.5" /></button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
