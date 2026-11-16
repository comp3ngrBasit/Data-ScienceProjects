import { useEffect, useState } from "react";
import { CircleCheck, Cpu, TriangleAlert } from "lucide-react";
import { visionApi, type VisionSettings } from "../../api/vision";
import { useToast } from "../../context/ToastContext";
import { useData } from "../../hooks/useData";
import { cn } from "../../lib/format";
import { Field, KV, LoadingBlock, Spinner } from "../ui";

export default function VisionSettingsPanel() {
  const toast = useToast();
  const { data, reload } = useData(visionApi.settings, []);
  const { data: sys } = useData(visionApi.system, []);
  const [s, setS] = useState<VisionSettings>();
  const [busy, setBusy] = useState(false);
  useEffect(() => setS(data), [data]);
  if (!s || !data) return <section className="panel"><LoadingBlock /></section>;

  const set = <K extends keyof VisionSettings>(k: K, v: VisionSettings[K]) => setS({ ...s, [k]: v });
  const dirty = JSON.stringify(s) !== JSON.stringify(data);
  const n = (k: keyof VisionSettings, min: number, max: number, step = 1) => (
    <input className="input" type="number" min={min} max={max} step={step} value={s[k] as number}
      onChange={(e) => set(k, (e.target.value === "" ? min : Number(e.target.value)) as never)} />
  );
  const save = async () => {
    setBusy(true);
    try {
      await visionApi.saveSettings(s);
      toast("success", "Vision settings saved — they apply to new analyses");
      void reload(true);
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(false);
    }
  };
  const t = sys?.torch;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="label-caps">Computer vision (Phase 2)</h2>
        <div className="flex gap-2">
          {dirty && <button className="btn-ghost px-2 py-1 text-xs" onClick={() => setS(data)}>Discard</button>}
          <button className="btn-primary px-3 py-1.5 text-xs" disabled={!dirty || busy} onClick={save}>{busy && <Spinner className="size-3 text-white" />}Save vision settings</button>
        </div>
      </div>
      <div className="space-y-5 px-5 py-4">
        {t && (
          <div className={cn("flex items-start gap-2 rounded-md border px-3 py-2 text-xs", t.cuda_available ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-200" : "border-ink-600 bg-ink-900 text-ink-300")}>
            {t.cuda_available ? <CircleCheck className="size-4 shrink-0" /> : <Cpu className="size-4 shrink-0" />}
            <span>
              PyTorch {t.torch ?? "not installed"} · {t.cuda_available ? `CUDA ${t.cuda_version} — ${t.gpus.join(", ")}. "auto" will use the GPU.` : "No CUDA GPU visible to PyTorch — analysis runs on the CPU (slower). See README → GPU acceleration."}
              {t.mps_available && " Apple GPU (MPS) available."}
            </span>
          </div>
        )}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Pose model" hint="n = fastest · x = most accurate. Downloaded on first use.">
            <select className="input" value={s.model} onChange={(e) => set("model", e.target.value)}>
              {sys?.available_models.map((m) => <option key={m} value={m}>{m}{sys.downloaded_models.includes(m) ? " ✓" : ""}</option>)}
            </select>
          </Field>
          <Field label="Device" hint="auto = CUDA if available, else CPU">
            <select className="input" value={s.device} onChange={(e) => set("device", e.target.value)}>
              {["auto", "cpu", "cuda", "cuda:1", "mps"].map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </Field>
          <Field label="Tracker" hint="BoT-SORT adds camera-motion compensation (slower)">
            <select className="input" value={s.tracker} onChange={(e) => set("tracker", e.target.value as VisionSettings["tracker"])}>
              <option value="bytetrack">ByteTrack</option>
              <option value="botsort">BoT-SORT</option>
            </select>
          </Field>
          <Field label="Analysis rate (frames/s of video)" hint="0 = every frame. 10 is a good balance.">{n("analysis_fps", 0, 240, 1)}</Field>
          <Field label="Inference size (px)" hint="Larger finds smaller people; slower">{n("image_size", 320, 1920, 32)}</Field>
          <Field label="Detection confidence">{n("detection_confidence", 0.05, 0.95, 0.05)}</Field>
          <Field label="Landmark confidence" hint="Below = treated as missing">{n("keypoint_confidence", 0.05, 0.95, 0.05)}</Field>
          <Field label="NMS IoU">{n("iou_threshold", 0.1, 0.95, 0.05)}</Field>
          <Field label="Occlusion memory (s)" hint="How long a lost track keeps its ID">{n("track_buffer_seconds", 0.1, 30, 0.5)}</Field>
          <Field label="Min person height" hint="Fraction of frame height — filters the crowd">{n("min_box_height", 0, 0.9, 0.01)}</Field>
          <Field label="Max people per frame" hint="Largest N kept">{n("max_people_per_frame", 1, 50)}</Field>
          <Field label="Smoothing window (frames)">{n("smoothing_window", 1, 31)}</Field>
          <Field label="Fill landmark gaps up to (frames)">{n("max_gap_interpolation", 0, 30)}</Field>
          <Field label="Stationary threshold" hint="Torso lengths / second">{n("stationary_threshold", 0, 5, 0.05)}</Field>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="flex cursor-pointer items-center justify-between gap-3 text-sm">
            <span><span className="block text-ink-100">Generate annotated video</span><span className="text-xs text-ink-500">H.264 MP4 with boxes, IDs and skeletons</span></span>
            <input type="checkbox" className="size-4 accent-[#e11d2e]" checked={s.generate_annotated_video} onChange={(e) => set("generate_annotated_video", e.target.checked)} />
          </label>
          <label className="flex cursor-pointer items-center justify-between gap-3 text-sm">
            <span><span className="block text-ink-100">FP16 on CUDA</span><span className="text-xs text-ink-500">Faster on NVIDIA GPUs; ignored on CPU</span></span>
            <input type="checkbox" className="size-4 accent-[#e11d2e]" checked={s.half_precision} onChange={(e) => set("half_precision", e.target.checked)} />
          </label>
        </div>
        {s.analysis_fps === 0 && <p className="flex items-center gap-2 text-xs text-amber-200"><TriangleAlert className="size-3.5" />Every frame will be analysed — expect long processing times on CPU.</p>}
        {t && (
          <dl className="border-t border-ink-700/60 pt-3">
            <KV k="CPU cores" v={t.cpu_threads ?? "—"} />
            <KV k="Downloaded models" v={sys?.downloaded_models.join(", ") || "none yet"} />
          </dl>
        )}
      </div>
    </section>
  );
}
