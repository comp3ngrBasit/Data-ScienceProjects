import { useEffect, useState } from "react";
import { CircleCheck, TriangleAlert } from "lucide-react";
import { api } from "../api/client";
import type { ProcessingSettings } from "../api/types";
import { ErrorBlock, Field, KV, LoadingBlock, PageHeader, Spinner } from "../components/ui";
import { useToast } from "../context/ToastContext";
import { useData } from "../hooks/useData";
import { cn, formatBytes } from "../lib/format";
import VisionSettingsPanel from "../components/vision/VisionSettingsPanel";

function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4 py-2">
      <span>
        <span className="block text-sm text-ink-100">{label}</span>
        {hint && <span className="block text-xs text-ink-500">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn("relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors", checked ? "bg-blood-500" : "bg-ink-600")}
      >
        <span className={cn("absolute top-0.5 size-4 rounded-full bg-white transition-all", checked ? "left-[18px]" : "left-0.5")} />
      </button>
    </label>
  );
}

export default function SettingsPage() {
  const toast = useToast();
  const { data, error, loading, reload } = useData(api.getSettings, []);
  const { data: sys } = useData(api.system, []);
  const [s, setS] = useState<ProcessingSettings>();
  const [busy, setBusy] = useState(false);
  useEffect(() => setS(data), [data]);

  if (loading && !data) return <><PageHeader title="Settings" /><LoadingBlock /></>;
  if (error && !data) return <><PageHeader title="Settings" /><ErrorBlock message={error} onRetry={() => reload()} /></>;
  if (!s || !data) return null;

  const set = <K extends keyof ProcessingSettings>(k: K, v: ProcessingSettings[K]) => setS({ ...s, [k]: v });
  const dirty = JSON.stringify(s) !== JSON.stringify(data);
  const numberInput = (k: keyof ProcessingSettings, props: { min: number; max: number; step?: number }) => (
    <input
      className="input"
      type="number"
      {...props}
      value={s[k] as number}
      onChange={(e) => set(k, (e.target.value === "" ? props.min : Number(e.target.value)) as never)}
    />
  );

  const save = async () => {
    setBusy(true);
    try {
      await api.saveSettings(s);
      toast("success", "Settings saved — they apply to new processing jobs");
      void reload(true);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  };

  // rough per-video estimate for a 15-minute fight at the current settings
  const est = s.sampling_mode === "interval" ? Math.min(Math.ceil(900 / s.sample_interval_seconds), s.max_sampled_frames) : Math.min(s.uniform_frame_count, s.max_sampled_frames);
  const estBytes = s.save_frame_images ? est * s.frame_max_width * s.frame_max_width * 0.5625 * 3 * (s.frame_jpeg_quality / 100) * 0.08 : 0;

  return (
    <>
      <PageHeader
        title="Settings"
        subtitle="Processing options are stored in the database and applied to new jobs. Reprocess a video to apply changes to it."
        actions={
          <>
            {dirty && <button className="btn-ghost" onClick={() => setS(data)}>Discard</button>}
            <button className="btn-primary" disabled={!dirty || busy} onClick={save}>{busy && <Spinner className="text-white" />}Save settings</button>
          </>
        }
      />
      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Frame sampling</h2></div>
            <div className="space-y-4 px-5 py-4">
              <div className="grid grid-cols-2 gap-2">
                {(["interval", "uniform"] as const).map((m) => (
                  <button key={m} type="button" onClick={() => set("sampling_mode", m)} className={cn("rounded-md border px-4 py-3 text-left transition-colors", s.sampling_mode === m ? "border-blood-500 bg-blood-900/20" : "border-ink-600 hover:border-ink-500")}>
                    <div className="text-sm font-medium text-ink-100">{m === "interval" ? "Fixed interval" : "Evenly spaced"}</div>
                    <div className="text-xs text-ink-400">{m === "interval" ? "One frame every N seconds" : "A fixed number of frames across the video"}</div>
                  </button>
                ))}
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {s.sampling_mode === "interval" ? (
                  <Field label="Interval (seconds)" hint="0.04 – 600">{numberInput("sample_interval_seconds", { min: 0.04, max: 600, step: 0.1 })}</Field>
                ) : (
                  <Field label="Frames per video" hint="1 – 5000">{numberInput("uniform_frame_count", { min: 1, max: 5000 })}</Field>
                )}
                <Field label="Maximum frames per video" hint="Hard cap — keeps storage bounded for long fights">{numberInput("max_sampled_frames", { min: 1, max: 5000 })}</Field>
                <Field label="Stored frame width (px)" hint="Frames are downscaled to this width">{numberInput("frame_max_width", { min: 64, max: 3840, step: 16 })}</Field>
                <Field label="JPEG quality" hint="30 – 100">{numberInput("frame_jpeg_quality", { min: 30, max: 100 })}</Field>
              </div>
              <Toggle checked={s.save_frame_images} onChange={(v) => set("save_frame_images", v)} label="Save frame images" hint="When off, only frame timestamps and quality stats are stored (no JPEGs)." />
              <p className="rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-xs text-ink-400">
                Estimate for a 15-minute fight: <span className="text-ink-200 tabular">{est} frames</span>
                {s.save_frame_images && <> · roughly <span className="text-ink-200 tabular">{formatBytes(estBytes)}</span> of JPEGs (varies with content)</>}
              </p>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">Uploads & processing</h2></div>
            <div className="space-y-2 px-5 py-4">
              <Toggle checked={s.auto_process_on_upload} onChange={(v) => set("auto_process_on_upload", v)} label="Process automatically after upload" hint="Otherwise videos stay “Uploaded” until you start processing." />
              <Toggle checked={s.generate_preview_proxy} onChange={(v) => set("generate_preview_proxy", v)} label="Generate browser preview for AVI / MOV / non-H.264" hint="Transcodes a 720p-ish H.264 MP4 for playback. The original is never modified." />
              <div className="grid gap-4 pt-2 sm:grid-cols-2">
                <Field label="Max upload size (MB)" hint="Per file">{numberInput("max_upload_size_mb", { min: 1, max: 20480 })}</Field>
                <Field label="Thumbnail width (px)">{numberInput("thumbnail_width", { min: 64, max: 1920, step: 16 })}</Field>
              </div>
            </div>
          </section>

          <VisionSettingsPanel />
        </div>

        <aside className="space-y-4">
          <section className="panel">
            <div className="panel-header"><h2 className="label-caps">System</h2></div>
            {sys ? (
              <div className="px-5 py-3">
                <div className={cn("mb-3 flex items-start gap-2 rounded-md border px-3 py-2 text-xs", sys.ffmpeg_path ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-200" : "border-amber-500/30 bg-amber-500/5 text-amber-200")}>
                  {sys.ffmpeg_path ? <CircleCheck className="size-4 shrink-0" /> : <TriangleAlert className="size-4 shrink-0" />}
                  <span>
                    {sys.ffmpeg_path ? `FFmpeg available (${sys.ffmpeg_source})` : "FFmpeg not found — previews disabled, metadata limited to OpenCV"}
                    {sys.ffmpeg_path && !sys.ffprobe_path && " · no ffprobe, using ffmpeg output parsing"}
                  </span>
                </div>
                <dl>
                  <KV k="Version" v={sys.version} />
                  <KV k="OpenCV" v={sys.opencv_version} />
                  <KV k="FFmpeg" v={sys.ffmpeg_path ?? "—"} mono />
                  <KV k="FFprobe" v={sys.ffprobe_path ?? "—"} mono />
                  <KV k="Workers" v={sys.worker_threads} />
                  <KV k="Formats" v={sys.allowed_extensions.join(" ")} />
                  <KV k="Data folder" v={sys.data_dir} mono />
                </dl>
              </div>
            ) : (
              <LoadingBlock />
            )}
          </section>
          {sys && (
            <section className="panel">
              <div className="panel-header"><h2 className="label-caps">Storage</h2></div>
              <dl className="px-5 py-3">
                {Object.entries(sys.storage_bytes).map(([k, v]) => <KV key={k} k={k[0].toUpperCase() + k.slice(1)} v={formatBytes(v)} />)}
              </dl>
            </section>
          )}
        </aside>
      </div>
    </>
  );
}
