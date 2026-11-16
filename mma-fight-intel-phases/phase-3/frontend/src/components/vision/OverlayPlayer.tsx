import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Pause, Play } from "lucide-react";
import { apiUrl } from "../../api/client";
import { SKELETON, trackColor, visionApi, type OverlayPerson, type Track } from "../../api/vision";
import { cn, formatTimestamp } from "../../lib/format";

export interface OverlayToggles {
  boxes: boolean;
  ids: boolean;
  skeleton: boolean;
  landmarks: boolean;
  fightersOnly: boolean;
}

export interface PlayerHandle {
  seekToFrame: (frame: number) => void;
  pause: () => void;
}

interface Props {
  videoId: string;
  originalUrl: string;
  annotatedUrl: string | null;
  source: "original" | "annotated";
  fps: number;
  stride: number;
  totalFrames: number;
  tracks: Track[];
  toggles: OverlayToggles;
  kpThreshold: number;
  onFrame: (frame: number) => void;
}

const CHUNK_ANALYSED = 300; // analysed frames per overlay request

function useOverlayData(videoId: string, stride: number) {
  const cache = useRef(new Map<number, Map<number, OverlayPerson[]>>());
  const pending = useRef(new Set<number>());
  const [, bump] = useState(0);
  useEffect(() => {
    cache.current.clear();
    pending.current.clear();
  }, [videoId, stride]);
  const chunkSize = CHUNK_ANALYSED * Math.max(stride, 1);

  const load = useCallback(
    (chunk: number) => {
      if (chunk < 0 || cache.current.has(chunk) || pending.current.has(chunk)) return;
      pending.current.add(chunk);
      visionApi
        .overlay(videoId, chunk * chunkSize, (chunk + 1) * chunkSize - 1)
        .then((d) => {
          const m = new Map<number, OverlayPerson[]>();
          d.frames.forEach(([f, people]) => m.set(f, people));
          cache.current.set(chunk, m);
          bump((x) => x + 1);
        })
        .catch(() => undefined)
        .finally(() => pending.current.delete(chunk));
    },
    [videoId, chunkSize],
  );

  const get = useCallback(
    (frame: number): OverlayPerson[] | undefined => {
      const analysed = frame - (frame % Math.max(stride, 1));
      const chunk = Math.floor(analysed / chunkSize);
      load(chunk);
      if (analysed % chunkSize > chunkSize * 0.7) load(chunk + 1);
      const m = cache.current.get(chunk);
      if (!m) return undefined; // not loaded yet
      return m.get(analysed) ?? [];
    },
    [stride, chunkSize, load],
  );
  return get;
}

const OverlayPlayer = forwardRef<PlayerHandle, Props>(function OverlayPlayer(
  { videoId, originalUrl, annotatedUrl, source, fps, stride, totalFrames, tracks, toggles, kpThreshold, onFrame },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const frameRef = useRef(0);
  const getPeople = useOverlayData(videoId, stride);
  const trackMap = useRef(new Map<number, Track>());
  trackMap.current = new Map(tracks.map((t) => [t.track_id, t]));

  const src = source === "annotated" && annotatedUrl ? annotatedUrl : originalUrl;

  const draw = useCallback(
    (f: number) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const wrap = wrapRef.current;
      if (!canvas || !video || !wrap) return;
      const dpr = window.devicePixelRatio || 1;
      const W = wrap.clientWidth;
      const H = wrap.clientHeight;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      if (source === "annotated") return; // burned into the video already
      const vw = video.videoWidth || 16;
      const vh = video.videoHeight || 9;
      const s = Math.min(W / vw, H / vh);
      const cw = vw * s;
      const ch = vh * s;
      const ox = (W - cw) / 2;
      const oy = (H - ch) / 2;
      const people = getPeople(f);
      if (!people) return;
      const lw = Math.max(1.5, cw / 600);
      for (const [tid, conf, x1, y1, x2, y2, kp] of people) {
        const t = trackMap.current.get(tid);
        if (toggles.fightersOnly && t?.role !== "fighter") continue;
        const color = trackColor(tid);
        const X = (x: number) => ox + x * cw;
        const Y = (y: number) => oy + y * ch;
        if (toggles.boxes) {
          ctx.strokeStyle = color;
          ctx.lineWidth = lw;
          ctx.strokeRect(X(x1), Y(y1), (x2 - x1) * cw, (y2 - y1) * ch);
        }
        if (toggles.ids) {
          const label = `ID ${tid}${t?.assigned_fighter ? ` · ${t.assigned_fighter.name}` : t?.role && t.role !== "unreviewed" ? ` · ${t.role}` : ""}  ${conf.toFixed(2)}`;
          ctx.font = `600 ${Math.max(11, Math.round(cw / 70))}px Inter Variable, system-ui, sans-serif`;
          const tw = ctx.measureText(label).width;
          const th = Math.max(16, Math.round(cw / 50));
          const ly = Math.max(Y(y1) - th, oy);
          ctx.fillStyle = color;
          ctx.fillRect(X(x1), ly, tw + 10, th);
          ctx.fillStyle = "#0e0e11";
          ctx.fillText(label, X(x1) + 5, ly + th * 0.75);
        }
        if (kp.length === 51) {
          const ok = (i: number) => kp[i * 3 + 2] >= kpThreshold;
          if (toggles.skeleton) {
            ctx.strokeStyle = color;
            ctx.lineWidth = lw * 1.3;
            ctx.lineCap = "round";
            for (const [a, b] of SKELETON) {
              if (!ok(a) || !ok(b)) continue;
              ctx.beginPath();
              ctx.moveTo(X(kp[a * 3]), Y(kp[a * 3 + 1]));
              ctx.lineTo(X(kp[b * 3]), Y(kp[b * 3 + 1]));
              ctx.stroke();
            }
          }
          if (toggles.landmarks) {
            for (let i = 0; i < 17; i++) {
              if (!ok(i)) continue;
              ctx.beginPath();
              ctx.arc(X(kp[i * 3]), Y(kp[i * 3 + 1]), lw * 2.2, 0, Math.PI * 2);
              ctx.fillStyle = "#fff";
              ctx.fill();
              ctx.beginPath();
              ctx.arc(X(kp[i * 3]), Y(kp[i * 3 + 1]), lw * 1.4, 0, Math.PI * 2);
              ctx.fillStyle = color;
              ctx.fill();
            }
          }
        }
      }
    },
    [getPeople, toggles, source, kpThreshold],
  );

  // Frame clock: requestVideoFrameCallback when available (exact presented frame), else rAF.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let handle = 0;
    let stopped = false;
    const update = (mediaTime: number) => {
      const f = Math.min(Math.max(0, Math.round(mediaTime * fps)), Math.max(totalFrames - 1, 0));
      if (f !== frameRef.current) {
        frameRef.current = f;
        setFrame(f);
        onFrame(f);
      }
      draw(f);
    };
    const hasRVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
    if (hasRVFC) {
      const cb = (_now: number, meta: { mediaTime: number }) => {
        if (stopped) return;
        update(meta.mediaTime);
        handle = (video as HTMLVideoElement & { requestVideoFrameCallback: (c: typeof cb) => number }).requestVideoFrameCallback(cb);
      };
      handle = (video as HTMLVideoElement & { requestVideoFrameCallback: (c: typeof cb) => number }).requestVideoFrameCallback(cb);
    } else {
      const loop = () => {
        if (stopped) return;
        update(video.currentTime);
        handle = requestAnimationFrame(loop);
      };
      handle = requestAnimationFrame(loop);
    }
    const onSeek = () => update(video.currentTime);
    video.addEventListener("seeked", onSeek);
    video.addEventListener("loadeddata", onSeek);
    return () => {
      stopped = true;
      if (hasRVFC) (video as HTMLVideoElement & { cancelVideoFrameCallback: (h: number) => void }).cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
      video.removeEventListener("seeked", onSeek);
      video.removeEventListener("loadeddata", onSeek);
    };
  }, [fps, totalFrames, draw, onFrame, src]);

  // Redraw on toggle / data changes / resize while paused
  useEffect(() => {
    draw(frameRef.current);
    const ro = new ResizeObserver(() => draw(frameRef.current));
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [draw]);

  // Keep playback position when switching original <-> annotated
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const restore = () => {
      if (frameRef.current > 0) v.currentTime = (frameRef.current + 0.5) / fps;
    };
    v.addEventListener("loadedmetadata", restore, { once: true });
    return () => v.removeEventListener("loadedmetadata", restore);
  }, [src, fps]);

  const seekToFrame = useCallback(
    (f: number) => {
      const v = videoRef.current;
      if (!v) return;
      const clamped = Math.min(Math.max(0, f), Math.max(totalFrames - 1, 0));
      v.currentTime = (clamped + 0.5) / fps; // land in the middle of the frame
    },
    [fps, totalFrames],
  );
  useImperativeHandle(ref, () => ({ seekToFrame, pause: () => videoRef.current?.pause() }), [seekToFrame]);

  const step = (n: number) => {
    videoRef.current?.pause();
    seekToFrame(frameRef.current + n);
  };
  const stepAnalysed = (dir: 1 | -1) => {
    videoRef.current?.pause();
    const s = Math.max(stride, 1);
    const cur = frameRef.current - (frameRef.current % s);
    seekToFrame(dir > 0 ? cur + s : cur === frameRef.current ? cur - s : cur);
  };
  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        if (e.shiftKey) stepAnalysed(1);
        else step(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (e.shiftKey) stepAnalysed(-1);
        else step(-1);
      } else if (e.key === " ") {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const analysed = frame - (frame % Math.max(stride, 1));
  return (
    <div className="panel overflow-hidden">
      <div ref={wrapRef} className="relative aspect-video bg-black">
        <video
          ref={videoRef}
          src={apiUrl(src)}
          className="absolute inset-0 size-full object-contain"
          controls
          preload="auto"
          muted
          playsInline
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" />
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-ink-700/70 px-3 py-2">
        <button className="btn-ghost p-1.5" onClick={() => stepAnalysed(-1)} title="Previous analysed frame (Shift+←)"><ChevronsLeft className="size-4" /></button>
        <button className="btn-ghost p-1.5" onClick={() => step(-1)} title="Previous frame (←)"><ChevronLeft className="size-4" /></button>
        <button className="btn-secondary px-2.5 py-1.5" onClick={toggle} title="Play / pause (Space)">{playing ? <Pause className="size-4" /> : <Play className="size-4" />}</button>
        <button className="btn-ghost p-1.5" onClick={() => step(1)} title="Next frame (→)"><ChevronRight className="size-4" /></button>
        <button className="btn-ghost p-1.5" onClick={() => stepAnalysed(1)} title="Next analysed frame (Shift+→)"><ChevronsRight className="size-4" /></button>
        <div className="ml-2 font-mono text-xs text-ink-300 tabular">
          frame <span className="text-white">{frame}</span>
          <span className="text-ink-500"> / {Math.max(totalFrames - 1, 0)}</span>
          <span className="ml-3">{formatTimestamp(frame / fps)}</span>
        </div>
        <div className={cn("ml-auto rounded px-2 py-0.5 text-[11px]", frame === analysed ? "bg-emerald-500/10 text-emerald-300" : "bg-ink-800 text-ink-400")}>
          {frame === analysed ? "analysed frame" : `showing results of frame ${analysed}`}
        </div>
      </div>
    </div>
  );
});

export default OverlayPlayer;
