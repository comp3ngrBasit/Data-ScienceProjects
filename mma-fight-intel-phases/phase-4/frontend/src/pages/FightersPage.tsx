import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Film, Pencil, Plus, Ruler, Search, Target, Trash2, Users } from "lucide-react";
import { api } from "../api/client";
import type { Fighter, FighterInput, Stance } from "../api/types";
import { ConfirmDialog, EmptyState, ErrorBlock, Field, LoadingBlock, Modal, PageHeader, Spinner, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cmToFtIn, cmToIn, cn, formatDate, formatDuration } from "../lib/format";

const WEIGHT_CLASSES = [
  "Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
  "Middleweight", "Light Heavyweight", "Heavyweight",
  "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
];
const STANCES: Stance[] = ["Orthodox", "Southpaw", "Unknown"];

const empty: FighterInput = { name: "", nickname: null, stance: "Unknown", height_cm: null, reach_cm: null, weight_class: null, notes: null };

function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((p) => p[0]?.toUpperCase()).join("");
}

function FighterForm({ open, initial, onClose, onSaved }: { open: boolean; initial: Fighter | null; onClose: () => void; onSaved: (id: string) => void }) {
  const [form, setForm] = useState<FighterInput>(empty);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string>();
  const toast = useToast();

  useEffect(() => {
    if (open) {
      setErr(undefined);
      setForm(initial ? { name: initial.name, nickname: initial.nickname, stance: initial.stance, height_cm: initial.height_cm, reach_cm: initial.reach_cm, weight_class: initial.weight_class, notes: initial.notes } : empty);
    }
  }, [open, initial]);

  const set = <K extends keyof FighterInput>(k: K, v: FighterInput[K]) => setForm((f) => ({ ...f, [k]: v }));
  const num = (s: string) => (s.trim() === "" ? null : Number(s));

  const submit = async () => {
    if (!form.name.trim()) return setErr("Name is required");
    if (form.height_cm != null && (form.height_cm <= 100 || form.height_cm >= 250)) return setErr("Height must be between 100 and 250 cm");
    if (form.reach_cm != null && (form.reach_cm <= 100 || form.reach_cm >= 260)) return setErr("Reach must be between 100 and 260 cm");
    setBusy(true);
    setErr(undefined);
    try {
      const saved = initial ? await api.updateFighter(initial.id, form) : await api.createFighter(form);
      toast("success", initial ? "Fighter updated" : `Added ${saved.name}`);
      onSaved(saved.id);
      onClose();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={initial ? "Edit fighter" : "Add fighter"}
      footer={
        <>
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={submit} disabled={busy}>{busy && <Spinner className="text-white" />}{initial ? "Save changes" : "Add fighter"}</button>
        </>
      }
    >
      <form className="grid grid-cols-2 gap-4" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
        <Field label="Name *" className="col-span-2 sm:col-span-1">
          <input className="input" value={form.name} maxLength={120} onChange={(e) => set("name", e.target.value)} placeholder="Full name" autoFocus />
        </Field>
        <Field label="Nickname" className="col-span-2 sm:col-span-1">
          <input className="input" value={form.nickname ?? ""} maxLength={120} onChange={(e) => set("nickname", e.target.value || null)} placeholder="Optional" />
        </Field>
        <div className="col-span-2">
          <span className="mb-1.5 block text-xs font-medium text-ink-300">Fighting stance</span>
          <div className="grid grid-cols-3 gap-2">
            {STANCES.map((s) => (
              <button key={s} type="button" onClick={() => set("stance", s)} className={cn("rounded-md border px-3 py-2 text-sm transition-colors", form.stance === s ? "border-blood-500 bg-blood-900/30 text-white" : "border-ink-600 text-ink-300 hover:border-ink-500")}>
                {s}
              </button>
            ))}
          </div>
        </div>
        <Field label="Height (cm)" hint={cmToFtIn(form.height_cm) ?? "Optional"}>
          <input className="input" type="number" step="0.5" min={100} max={250} value={form.height_cm ?? ""} onChange={(e) => set("height_cm", num(e.target.value))} placeholder="e.g. 180" />
        </Field>
        <Field label="Reach (cm)" hint={cmToIn(form.reach_cm) ?? "Optional"}>
          <input className="input" type="number" step="0.5" min={100} max={260} value={form.reach_cm ?? ""} onChange={(e) => set("reach_cm", num(e.target.value))} placeholder="e.g. 188" />
        </Field>
        <Field label="Weight class" className="col-span-2">
          <input className="input" list="weight-classes" value={form.weight_class ?? ""} maxLength={60} onChange={(e) => set("weight_class", e.target.value || null)} placeholder="Select or type" />
          <datalist id="weight-classes">{WEIGHT_CLASSES.map((w) => <option key={w} value={w} />)}</datalist>
        </Field>
        <Field label="Notes" className="col-span-2">
          <textarea className="input min-h-[70px] resize-y" value={form.notes ?? ""} maxLength={5000} onChange={(e) => set("notes", e.target.value || null)} placeholder="Style, tendencies, camp…" />
        </Field>
        {err && <p className="col-span-2 text-sm text-blood-300">{err}</p>}
        <button type="submit" className="hidden" />
      </form>
    </Modal>
  );
}

function FighterDetailPanel({ id, onEdit, onDelete }: { id: string; onEdit: (f: Fighter) => void; onDelete: (f: Fighter) => void }) {
  const { data: f, error, loading } = useData(() => api.getFighter(id), [id]);
  if (loading && !f) return <div className="panel"><LoadingBlock /></div>;
  if (error && !f) return <ErrorBlock message={error} />;
  if (!f) return null;
  return (
    <div className="panel">
      <div className="flex items-start gap-4 border-b border-ink-700/80 px-5 py-5">
        <div className="grid size-14 shrink-0 place-items-center rounded-lg border border-blood-700/50 bg-blood-900/30 font-display text-2xl font-bold text-blood-300">{initials(f.name)}</div>
        <div className="min-w-0 flex-1">
          <h2 className="truncate font-display text-2xl font-bold uppercase tracking-wide text-white">{f.name}</h2>
          {f.nickname && <div className="text-sm text-ink-400">“{f.nickname}”</div>}
        </div>
        <div className="flex gap-1">
          <button className="btn-ghost p-2" onClick={() => onEdit(f)} title="Edit"><Pencil className="size-4" /></button>
          <button className="btn-ghost p-2 hover:text-blood-400" onClick={() => onDelete(f)} title="Delete"><Trash2 className="size-4" /></button>
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-px bg-ink-700/60 sm:grid-cols-4">
        {[
          ["Stance", f.stance],
          ["Height", f.height_cm ? `${f.height_cm} cm · ${cmToFtIn(f.height_cm)}` : "—"],
          ["Reach", f.reach_cm ? `${f.reach_cm} cm · ${cmToIn(f.reach_cm)}` : "—"],
          ["Weight class", f.weight_class ?? "—"],
        ].map(([k, v]) => (
          <div key={k} className="bg-ink-850 px-5 py-3">
            <dt className="label-caps">{k}</dt>
            <dd className="mt-1 text-sm text-ink-100">{v}</dd>
          </div>
        ))}
      </dl>
      {f.notes && <p className="whitespace-pre-wrap border-t border-ink-700/80 px-5 py-3 text-sm text-ink-300">{f.notes}</p>}
      <div className="panel-header border-t">
        <h3 className="label-caps">Associated fight videos</h3>
        <span className="text-xs text-ink-400">{f.videos.length} video{f.videos.length === 1 ? "" : "s"} · {f.target_video_count} as analysis target</span>
      </div>
      {f.videos.length === 0 ? (
        <EmptyState icon={<Film className="size-5" />} title="No videos assigned" body="Open a video and assign this fighter from the “Fighters in this video” panel." action={<Link to="/videos" className="btn-secondary">Go to library</Link>} />
      ) : (
        <ul className="divide-y divide-ink-700/60">
          {f.videos.map((v) => (
            <li key={v.id}>
              <Link to={`/videos/${v.id}`} className="flex items-center gap-3 px-5 py-3 hover:bg-ink-800/60">
                <div className="h-10 w-16 shrink-0 overflow-hidden rounded bg-ink-950">
                  {v.has_thumbnail && <img src={`/api/videos/${v.id}/thumbnail`} alt="" className="size-full object-cover" loading="lazy" />}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm text-ink-100">{v.title}</span>
                    {v.is_target && <span className="flex shrink-0 items-center gap-1 rounded bg-blood-900/50 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-blood-300"><Target className="size-3" />Target</span>}
                  </div>
                  <div className="text-xs text-ink-500 tabular">{formatDuration(v.duration_seconds)} · {formatDate(v.uploaded_at, false)}</div>
                </div>
                <StatusBadge status={v.status} />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function FightersPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const [search, setSearch] = useState("");
  const { data, error, loading, reload } = useData(() => api.listFighters(), []);
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<Fighter | null>(null);
  const [deleting, setDeleting] = useState<Fighter | null>(null);
  const [busy, setBusy] = useState(false);
  const [detailKey, setDetailKey] = useState(0);

  const list = (data ?? []).filter((f) => !search || `${f.name} ${f.nickname ?? ""} ${f.weight_class ?? ""}`.toLowerCase().includes(search.toLowerCase()));
  const selected = id ?? (data && data.length > 0 ? undefined : undefined);

  const doDelete = async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.deleteFighter(deleting.id);
      toast("success", `Deleted ${deleting.name}`);
      if (id === deleting.id) navigate("/fighters");
      setDeleting(null);
      void reload(true);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Delete failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Fighters"
        subtitle="Fighter profiles are created and assigned manually in Phase 1 — no automatic recognition."
        actions={<button className="btn-primary" onClick={() => { setEditing(null); setFormOpen(true); }}><Plus className="size-4" />Add fighter</button>}
      />
      {loading && !data ? (
        <LoadingBlock />
      ) : error && !data ? (
        <ErrorBlock message={error} onRetry={() => reload()} />
      ) : data && data.length === 0 ? (
        <div className="panel">
          <EmptyState icon={<Users className="size-5" />} title="No fighters yet" body="Create a profile for each athlete you want to analyze, then assign them to their fight videos." action={<button className="btn-primary" onClick={() => { setEditing(null); setFormOpen(true); }}><Plus className="size-4" />Add your first fighter</button>} />
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_1fr]">
          <div className="space-y-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-500" />
              <input className="input pl-9" placeholder="Search fighters…" value={search} onChange={(e) => setSearch(e.target.value)} />
            </div>
            <ul className="space-y-1.5">
              {list.map((f) => (
                <li key={f.id}>
                  <Link
                    to={`/fighters/${f.id}`}
                    className={cn("panel flex items-center gap-3 px-4 py-3 transition-colors hover:border-ink-500", selected === f.id && "border-blood-600/70 bg-ink-800")}
                  >
                    <div className={cn("grid size-10 shrink-0 place-items-center rounded-md font-display text-lg font-bold", selected === f.id ? "bg-blood-500 text-white" : "bg-ink-700 text-ink-200")}>{initials(f.name)}</div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-ink-100">{f.name}</div>
                      <div className="flex flex-wrap items-center gap-x-2 text-xs text-ink-400">
                        <span>{f.stance}</span>
                        {f.weight_class && <><span className="text-ink-600">·</span><span className="truncate">{f.weight_class}</span></>}
                        {f.height_cm && <><span className="text-ink-600">·</span><span className="flex items-center gap-0.5"><Ruler className="size-3" />{cmToFtIn(f.height_cm)}</span></>}
                      </div>
                    </div>
                    <span className="flex shrink-0 items-center gap-1 text-xs text-ink-400 tabular"><Film className="size-3.5" />{f.video_count}</span>
                  </Link>
                </li>
              ))}
              {list.length === 0 && <li className="px-2 py-4 text-sm text-ink-500">No fighters match “{search}”.</li>}
            </ul>
          </div>
          <div>
            {selected ? (
              <FighterDetailPanel key={`${selected}-${detailKey}`} id={selected} onEdit={(f) => { setEditing(f); setFormOpen(true); }} onDelete={setDeleting} />
            ) : (
              <div className="panel"><EmptyState icon={<Users className="size-5" />} title="Select a fighter" body="View profile details and every fight video they're assigned to." /></div>
            )}
          </div>
        </div>
      )}
      <FighterForm
        open={formOpen}
        initial={editing}
        onClose={() => setFormOpen(false)}
        onSaved={(fid) => { void reload(true); setDetailKey((k) => k + 1); navigate(`/fighters/${fid}`); }}
      />
      <ConfirmDialog
        open={!!deleting}
        title="Delete fighter"
        busy={busy}
        onClose={() => setDeleting(null)}
        onConfirm={doDelete}
        body={<>Delete <span className="font-medium text-white">{deleting?.name}</span>? Their videos stay in the library but will be unassigned.</>}
      />
    </>
  );
}
