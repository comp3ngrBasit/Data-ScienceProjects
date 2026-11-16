import { useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";
import { CircleAlert, CircleCheck, CopyX, FileVideo, RotateCw, Upload, X } from "lucide-react";
import { api } from "../api/client";
import { PageHeader, ProgressBar, Spinner } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { ALLOWED_EXTENSIONS, useUploads, type UploadItem } from "../context/UploadContext";
import { useData } from "../hooks/useData";
import { cn, formatBytes } from "../lib/format";

function ItemRow({ item }: { item: UploadItem }) {
  const { cancel, remove, retry } = useUploads();
  const active = item.state === "uploading" || item.state === "finalizing" || item.state === "pending";
  return (
    <li className="flex items-center gap-4 px-5 py-3.5">
      <div
        className={cn(
          "grid size-10 shrink-0 place-items-center rounded-md border",
          item.state === "done" && "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
          item.state === "error" && "border-blood-600/40 bg-blood-900/40 text-blood-400",
          item.state === "duplicate" && "border-amber-500/30 bg-amber-500/10 text-amber-300",
          (active || item.state === "cancelled") && "border-ink-600 bg-ink-800 text-ink-300",
        )}
      >
        {item.state === "done" ? <CircleCheck className="size-5" /> : item.state === "error" ? <CircleAlert className="size-5" /> : item.state === "duplicate" ? <CopyX className="size-5" /> : <FileVideo className="size-5" />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <span className="truncate text-sm font-medium text-ink-100" title={item.name}>{item.name}</span>
          <span className="shrink-0 text-xs text-ink-400 tabular">{formatBytes(item.size)}</span>
        </div>
        {item.state === "uploading" || item.state === "finalizing" ? (
          <div className="mt-2 flex items-center gap-3">
            <ProgressBar value={item.progress} />
            <span className="w-24 shrink-0 text-right text-xs text-ink-400 tabular">
              {item.state === "finalizing" ? "Verifying…" : `${Math.round(item.progress * 100)}% · ${formatBytes(item.size * item.progress)}`}
            </span>
          </div>
        ) : (
          <div className="mt-1 text-xs">
            {item.state === "pending" && <span className="text-ink-400">Waiting…</span>}
            {item.state === "done" && (
              <span className="text-emerald-300">
                Uploaded{" "}
                {item.videoId && <Link to={`/videos/${item.videoId}`} className="ml-1 text-ink-200 underline decoration-ink-500 underline-offset-2 hover:text-white">View video</Link>}
              </span>
            )}
            {item.state === "duplicate" && (
              <span className="text-amber-200">
                {item.error}
                {item.existingVideoId && <Link to={`/videos/${item.existingVideoId}`} className="ml-2 text-ink-200 underline underline-offset-2 hover:text-white">Open existing</Link>}
              </span>
            )}
            {(item.state === "error" || item.state === "cancelled") && <span className="text-blood-300">{item.error}</span>}
          </div>
        )}
      </div>
      <div className="flex shrink-0 gap-1">
        {(item.state === "error" || item.state === "cancelled") && ALLOWED_EXTENSIONS.some((e) => item.name.toLowerCase().endsWith(e)) && (
          <button className="btn-ghost p-1.5" title="Retry upload" onClick={() => retry(item.id)}><RotateCw className="size-4" /></button>
        )}
        {active ? (
          <button className="btn-ghost p-1.5" title="Cancel" onClick={() => cancel(item.id)}><X className="size-4" /></button>
        ) : (
          <button className="btn-ghost p-1.5" title="Remove from list" onClick={() => remove(item.id)}><X className="size-4" /></button>
        )}
      </div>
    </li>
  );
}

export default function UploadPage() {
  const { items, addFiles, clearFinished, maxSizeMb, activeCount } = useUploads();
  const { data: settings } = useData(api.getSettings, []);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const toast = useToast();

  const accept = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const { accepted, rejected } = addFiles(files);
    if (rejected) toast("error", `${rejected} file${rejected > 1 ? "s" : ""} rejected — see details below`);
    else if (accepted) toast("info", `${accepted} file${accepted > 1 ? "s" : ""} added to the upload queue`);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    accept(e.dataTransfer.files);
  };

  const done = items.filter((i) => i.state === "done").length;
  const finished = items.length - activeCount;

  return (
    <>
      <PageHeader title="Upload Videos" subtitle="Add UFC / MMA fight footage to the library. Uploads continue while you browse other pages." />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <div
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            onDragEnter={(e) => { e.preventDefault(); dragDepth.current++; setDragging(true); }}
            onDragOver={(e) => e.preventDefault()}
            onDragLeave={() => { dragDepth.current--; if (dragDepth.current <= 0) setDragging(false); }}
            onDrop={onDrop}
            className={cn(
              "relative flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-lg border-2 border-dashed px-6 py-16 text-center transition-colors",
              dragging ? "border-blood-500 bg-blood-900/20" : "border-ink-600 bg-ink-850 hover:border-ink-500 hover:bg-ink-800/70",
            )}
          >
            <div className={cn("grid size-14 place-items-center rounded-full border transition-colors", dragging ? "border-blood-500 bg-blood-500 text-white" : "border-ink-600 bg-ink-800 text-blood-400")}>
              <Upload className="size-6" />
            </div>
            <p className="mt-4 font-display text-xl font-semibold uppercase tracking-wide text-white">
              {dragging ? "Release to upload" : "Drag & drop fight videos"}
            </p>
            <p className="mt-1 text-sm text-ink-400">or click to browse — multiple files supported</p>
            <div className="mt-4 flex flex-wrap justify-center gap-1.5">
              {ALLOWED_EXTENSIONS.map((e) => (
                <span key={e} className="rounded border border-ink-600 bg-ink-900 px-2 py-0.5 font-mono text-xs uppercase text-ink-300">{e.slice(1)}</span>
              ))}
              {maxSizeMb && <span className="rounded border border-ink-600 bg-ink-900 px-2 py-0.5 text-xs text-ink-300">max {maxSizeMb >= 1024 ? `${(maxSizeMb / 1024).toFixed(maxSizeMb % 1024 ? 1 : 0)} GB` : `${maxSizeMb} MB`} each</span>}
            </div>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
              className="hidden"
              onChange={(e) => { accept(e.target.files); e.target.value = ""; }}
            />
          </div>

          <section className="panel mt-4">
            <div className="panel-header">
              <h2 className="label-caps">Upload queue</h2>
              <div className="flex items-center gap-3 text-xs text-ink-400">
                {activeCount > 0 && <span className="flex items-center gap-1.5"><Spinner className="size-3" /> {activeCount} in progress</span>}
                {done > 0 && <span className="tabular">{done} uploaded</span>}
                {finished > 0 && <button className="btn-ghost px-2 py-1 text-xs" onClick={clearFinished}>Clear finished</button>}
              </div>
            </div>
            {items.length === 0 ? (
              <p className="px-5 py-10 text-center text-sm text-ink-500">No files queued yet.</p>
            ) : (
              <ul className="divide-y divide-ink-700/60">{[...items].reverse().map((i) => <ItemRow key={i.id} item={i} />)}</ul>
            )}
          </section>
        </div>

        <aside className="space-y-4">
          <section className="panel px-5 py-4">
            <h2 className="label-caps mb-3">What happens after upload</h2>
            <ol className="space-y-3 text-sm text-ink-300">
              {[
                ["Validated & stored", "File type, container signature and size are checked; duplicates (same file content) are detected."],
                [settings?.auto_process_on_upload === false ? "Waits for you" : "Queued automatically", settings?.auto_process_on_upload === false ? "Auto-processing is off — start processing from the video page." : "A background job starts; you can keep working."],
                ["Metadata extracted", "Duration, frame count, FPS, resolution, codecs and audio via FFmpeg + OpenCV."],
                ["Thumbnail & frames", `Thumbnail plus sampled frames (${settings ? (settings.sampling_mode === "interval" ? `every ${settings.sample_interval_seconds}s` : `${settings.uniform_frame_count} evenly spaced`) : "configurable"}, max ${settings?.max_sampled_frames ?? "—"}).`],
              ].map(([t, d], i) => (
                <li key={t} className="flex gap-3">
                  <span className="grid size-5 shrink-0 place-items-center rounded-full bg-ink-700 text-[11px] font-semibold text-ink-200">{i + 1}</span>
                  <div><div className="font-medium text-ink-100">{t}</div><div className="text-xs text-ink-400">{d}</div></div>
                </li>
              ))}
            </ol>
            <Link to="/settings" className="mt-4 inline-block text-xs text-ink-400 underline decoration-ink-600 underline-offset-2 hover:text-ink-200">Change processing settings</Link>
          </section>
          <section className="panel px-5 py-4 text-xs text-ink-400">
            <h2 className="label-caps mb-2">Tips</h2>
            <ul className="list-disc space-y-1.5 pl-4">
              <li>AVI and MOV files get an MP4 preview for in-browser playback; the original is kept untouched for analysis.</li>
              <li>Higher resolution and 30+ fps footage gives later detection phases more to work with.</li>
              <li>Keep this tab open until uploads finish — closing it cancels in-flight uploads.</li>
            </ul>
          </section>
        </aside>
      </div>
    </>
  );
}
