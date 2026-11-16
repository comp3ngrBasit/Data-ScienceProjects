import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { LayoutGrid, List, Pencil, RotateCw, Search, Trash2, Upload, Volume2 } from "lucide-react";
import { ApiError, api, type VideoQuery } from "../api/client";
import type { Fighter, VideoStatus, VideoSummary } from "../api/types";
import { ConfirmDialog, EmptyState, ErrorBlock, Field, LoadingBlock, Modal, PageHeader, Spinner, StatusBadge, Thumb } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useUploads } from "../context/UploadContext";
import { useData } from "../hooks/useData";
import { cn, formatBytes, formatDate, formatDuration, formatFps, formatResolution } from "../lib/format";

const PAGE_SIZE = 48;

export function RenameDialog({ video, onClose, onSaved }: { video: VideoSummary | null; onClose: () => void; onSaved: () => void }) {
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  useEffect(() => setTitle(video?.title ?? ""), [video]);
  const save = async () => {
    if (!video || !title.trim()) return;
    setBusy(true);
    try {
      await api.updateVideo(video.id, { title: title.trim() });
      toast("success", "Title updated");
      onSaved();
      onClose();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Rename failed");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal
      open={!!video}
      onClose={onClose}
      title="Rename video"
      footer={
        <>
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={save} disabled={busy || !title.trim()}>{busy && <Spinner className="text-white" />}Save</button>
        </>
      }
    >
      <form onSubmit={(e) => { e.preventDefault(); void save(); }}>
        <Field label="Display title" hint={video ? `Original file: ${video.original_filename}` : undefined}>
          <input className="input" value={title} maxLength={255} onChange={(e) => setTitle(e.target.value)} autoFocus />
        </Field>
      </form>
    </Modal>
  );
}

export function useVideoActions(onChange: () => void) {
  const toast = useToast();
  const [renaming, setRenaming] = useState<VideoSummary | null>(null);
  const [deleting, setDeleting] = useState<VideoSummary | null>(null);
  const [busy, setBusy] = useState(false);

  const process = async (v: VideoSummary) => {
    try {
      await api.processVideo(v.id);
      toast("info", `Queued “${v.title}” for processing`);
      onChange();
    } catch (e) {
      toast("error", e instanceof ApiError ? e.message : "Could not queue processing");
    }
  };
  const confirmDelete = async (after?: () => void) => {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.deleteVideo(deleting.id);
      toast("success", `Deleted “${deleting.title}”`);
      setDeleting(null);
      onChange();
      after?.();
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Delete failed");
    } finally {
      setBusy(false);
    }
  };
  const dialogs = (after?: () => void) => (
    <>
      <RenameDialog video={renaming} onClose={() => setRenaming(null)} onSaved={onChange} />
      <ConfirmDialog
        open={!!deleting}
        title="Delete video"
        busy={busy}
        onClose={() => setDeleting(null)}
        onConfirm={() => confirmDelete(after)}
        body={
          <>
            Permanently delete <span className="font-medium text-white">“{deleting?.title}”</span>? The original file, thumbnail, sampled
            frames, preview and processing history will be removed. This cannot be undone.
          </>
        }
      />
    </>
  );
  return { process, setRenaming, setDeleting, dialogs };
}

function VideoCard({ v, actions }: { v: VideoSummary; actions: ReturnType<typeof useVideoActions> }) {
  return (
    <div className="panel group flex flex-col overflow-hidden transition-colors hover:border-ink-500">
      <Link to={`/videos/${v.id}`} className="relative block aspect-video">
        <Thumb url={v.thumbnail_url} alt={v.title} status={v.status} className="size-full" />
        <div className="absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/80 to-transparent px-2.5 pb-2 pt-8">
          <StatusBadge status={v.status} className="bg-black/60 backdrop-blur" />
          {v.duration_seconds != null && (
            <span className="rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white tabular">{formatDuration(v.duration_seconds)}</span>
          )}
        </div>
      </Link>
      <div className="flex flex-1 flex-col px-3.5 py-3">
        <Link to={`/videos/${v.id}`} className="truncate text-sm font-medium text-ink-100 hover:text-white" title={v.title}>
          {v.title}
        </Link>
        <div className="mt-1 flex flex-wrap gap-x-2 text-xs text-ink-400 tabular">
          {v.width ? (
            <>
              <span>{formatResolution(v.width, v.height)}</span>
              <span className="text-ink-600">·</span>
              <span>{formatFps(v.fps)}</span>
            </>
          ) : (
            <span className="text-ink-500">{v.status === "failed" ? "No metadata" : "Metadata pending"}</span>
          )}
          <span className="text-ink-600">·</span>
          <span>{formatBytes(v.file_size)}</span>
          {v.has_audio && <Volume2 className="size-3 self-center text-ink-500" aria-label="Has audio" />}
        </div>
        {v.target_fighter && (
          <div className="mt-2 truncate text-xs text-ink-300">
            <span className="text-blood-400">◎</span> {v.target_fighter.name}
          </div>
        )}
        {v.status === "failed" && v.error_message && <p className="mt-2 line-clamp-2 text-xs text-blood-300">{v.error_message}</p>}
        <div className="mt-auto flex items-center justify-between pt-3 text-[11px] text-ink-500">
          <span>{formatDate(v.uploaded_at)}</span>
          <div className="flex gap-0.5 opacity-80 group-hover:opacity-100">
            {(v.status === "failed" || v.status === "uploaded") && (
              <button className="btn-ghost p-1.5" title={v.status === "failed" ? "Retry processing" : "Process"} onClick={() => actions.process(v)}>
                <RotateCw className="size-3.5" />
              </button>
            )}
            <button className="btn-ghost p-1.5" title="Rename" onClick={() => actions.setRenaming(v)}>
              <Pencil className="size-3.5" />
            </button>
            <button className="btn-ghost p-1.5 hover:text-blood-400" title="Delete" onClick={() => actions.setDeleting(v)}>
              <Trash2 className="size-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function VideoTable({ items, actions }: { items: VideoSummary[]; actions: ReturnType<typeof useVideoActions> }) {
  const navigate = useNavigate();
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full min-w-[900px] text-sm">
        <thead>
          <tr className="border-b border-ink-700 text-left">
            {["", "Title", "Duration", "Resolution", "Frame rate", "Size", "Uploaded", "Status", ""].map((h, i) => (
              <th key={i} className="label-caps px-3 py-3 font-semibold">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-ink-700/50">
          {items.map((v) => (
            <tr key={v.id} className="cursor-pointer hover:bg-ink-800/60" onClick={() => navigate(`/videos/${v.id}`)}>
              <td className="py-2 pl-3"><Thumb url={v.thumbnail_url} alt="" status={v.status} className="h-10 w-16 rounded" /></td>
              <td className="max-w-[280px] px-3">
                <div className="truncate font-medium text-ink-100">{v.title}</div>
                <div className="truncate text-xs text-ink-500">{v.original_filename}</div>
              </td>
              <td className="px-3 text-ink-300 tabular">{formatDuration(v.duration_seconds)}</td>
              <td className="px-3 text-ink-300 tabular">{formatResolution(v.width, v.height)}</td>
              <td className="px-3 text-ink-300 tabular">{formatFps(v.fps)}</td>
              <td className="px-3 text-ink-300 tabular">{formatBytes(v.file_size)}</td>
              <td className="px-3 text-ink-400">{formatDate(v.uploaded_at)}</td>
              <td className="px-3"><StatusBadge status={v.status} /></td>
              <td className="px-3" onClick={(e) => e.stopPropagation()}>
                <div className="flex justify-end gap-0.5">
                  {(v.status === "failed" || v.status === "uploaded") && (
                    <button className="btn-ghost p-1.5" title="Process / retry" onClick={() => actions.process(v)}><RotateCw className="size-3.5" /></button>
                  )}
                  <button className="btn-ghost p-1.5" title="Rename" onClick={() => actions.setRenaming(v)}><Pencil className="size-3.5" /></button>
                  <button className="btn-ghost p-1.5 hover:text-blood-400" title="Delete" onClick={() => actions.setDeleting(v)}><Trash2 className="size-3.5" /></button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function VideoLibraryPage() {
  const [params, setParams] = useSearchParams();
  const [searchInput, setSearchInput] = useState(params.get("q") ?? "");
  const [view, setView] = useState<"grid" | "table">(() => {
    try {
      return (localStorage.getItem("library-view") as "grid" | "table") || "grid";
    } catch {
      return "grid";
    }
  });
  const [page, setPage] = useState(0);
  const { onUploaded } = useUploads();

  const query: VideoQuery = {
    search: params.get("q") ?? "",
    status: (params.get("status") as VideoStatus) || "",
    fighter_id: params.get("fighter") ?? undefined,
    unassigned: params.get("fighter") === "__none" || undefined,
    sort: (params.get("sort") as VideoQuery["sort"]) || "uploaded_at",
    order: (params.get("order") as "asc" | "desc") || "desc",
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  if (query.fighter_id === "__none") query.fighter_id = undefined;
  const key = JSON.stringify(query);

  const { data, error, loading, reload } = useData(() => api.listVideos(query), [key], (d) =>
    d?.items.some((v) => v.status === "processing" || v.status === "queued") ? 2500 : undefined,
  );
  const { data: fighters } = useData<Fighter[]>(() => api.listFighters(), []);
  const actions = useVideoActions(() => reload(true));

  useEffect(() => onUploaded(() => reload(true)), [onUploaded, reload]);

  // Debounced search -> URL
  useEffect(() => {
    const t = window.setTimeout(() => {
      if ((params.get("q") ?? "") !== searchInput) setParam("q", searchInput);
    }, 300);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchInput]);

  function setParam(k: string, v: string) {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v);
    else next.delete(k);
    setParams(next, { replace: true });
    setPage(0);
  }
  const changeView = (v: "grid" | "table") => {
    setView(v);
    try {
      localStorage.setItem("library-view", v);
    } catch {
      /* storage unavailable */
    }
  };

  const filtersActive = !!(query.search || query.status || params.get("fighter"));
  const pages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  return (
    <>
      <PageHeader
        title="Video Library"
        subtitle={data ? `${data.total} video${data.total === 1 ? "" : "s"}${filtersActive ? " matching filters" : ""}` : "Manage uploaded fight footage"}
        actions={<Link to="/upload" className="btn-primary"><Upload className="size-4" /> Upload</Link>}
      />

      <div className="mb-4 flex flex-col gap-2 lg:flex-row lg:items-center">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-500" />
          <input className="input pl-9" placeholder="Search by title or filename…" value={searchInput} onChange={(e) => setSearchInput(e.target.value)} />
        </div>
        <div className="flex flex-wrap gap-2">
          <select className="input w-auto" value={query.status} onChange={(e) => setParam("status", e.target.value)} aria-label="Filter by status">
            <option value="">All statuses</option>
            {["uploaded", "queued", "processing", "completed", "failed"].map((s) => (
              <option key={s} value={s}>{s[0].toUpperCase() + s.slice(1)}</option>
            ))}
          </select>
          <select className="input w-auto max-w-[200px]" value={params.get("fighter") ?? ""} onChange={(e) => setParam("fighter", e.target.value)} aria-label="Filter by fighter">
            <option value="">All fighters</option>
            <option value="__none">Unassigned</option>
            {fighters?.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
          <select
            className="input w-auto"
            value={`${query.sort}:${query.order}`}
            onChange={(e) => {
              const [s, o] = e.target.value.split(":");
              const next = new URLSearchParams(params);
              next.set("sort", s);
              next.set("order", o);
              setParams(next, { replace: true });
            }}
            aria-label="Sort"
          >
            <option value="uploaded_at:desc">Newest first</option>
            <option value="uploaded_at:asc">Oldest first</option>
            <option value="title:asc">Title A–Z</option>
            <option value="title:desc">Title Z–A</option>
            <option value="duration:desc">Longest</option>
            <option value="duration:asc">Shortest</option>
            <option value="file_size:desc">Largest file</option>
          </select>
          <div className="flex rounded-md border border-ink-600 bg-ink-900 p-0.5">
            {(["grid", "table"] as const).map((m) => (
              <button
                key={m}
                onClick={() => changeView(m)}
                className={cn("rounded px-2.5 py-1.5", view === m ? "bg-ink-700 text-white" : "text-ink-400 hover:text-ink-100")}
                aria-label={`${m} view`}
              >
                {m === "grid" ? <LayoutGrid className="size-4" /> : <List className="size-4" />}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading && !data ? (
        <LoadingBlock />
      ) : error && !data ? (
        <ErrorBlock message={error} onRetry={() => reload()} />
      ) : data && data.items.length === 0 ? (
        <div className="panel">
          {filtersActive ? (
            <EmptyState icon={<Search className="size-5" />} title="No videos match your filters" action={<button className="btn-secondary" onClick={() => { setSearchInput(""); setParams({}, { replace: true }); }}>Clear filters</button>} />
          ) : (
            <EmptyState title="Your library is empty" body="Upload fight footage to see it here." action={<Link to="/upload" className="btn-primary"><Upload className="size-4" />Upload footage</Link>} />
          )}
        </div>
      ) : data ? (
        <>
          {view === "grid" ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
              {data.items.map((v) => <VideoCard key={v.id} v={v} actions={actions} />)}
            </div>
          ) : (
            <VideoTable items={data.items} actions={actions} />
          )}
          {pages > 1 && (
            <div className="mt-5 flex items-center justify-center gap-3 text-sm">
              <button className="btn-secondary" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Previous</button>
              <span className="text-ink-400 tabular">Page {page + 1} of {pages}</span>
              <button className="btn-secondary" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>Next</button>
            </div>
          )}
        </>
      ) : null}
      {actions.dialogs()}
    </>
  );
}
