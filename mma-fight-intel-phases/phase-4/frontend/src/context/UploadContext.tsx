import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError, api, uploadFile } from "../api/client";
import type { UploadResponse } from "../api/types";
import { formatBytes } from "../lib/format";

export const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi"];
const CONCURRENCY = 2;

export type UploadState = "pending" | "uploading" | "finalizing" | "done" | "duplicate" | "error" | "cancelled";

export interface UploadItem {
  id: string;
  file: File;
  name: string;
  size: number;
  state: UploadState;
  progress: number;
  error?: string;
  videoId?: string;
  existingVideoId?: string;
}

interface UploadCtx {
  items: UploadItem[];
  maxSizeMb: number | null;
  addFiles: (files: FileList | File[]) => { accepted: number; rejected: number };
  cancel: (id: string) => void;
  remove: (id: string) => void;
  retry: (id: string) => void;
  clearFinished: () => void;
  activeCount: number;
  onUploaded: (cb: () => void) => () => void;
}

const Ctx = createContext<UploadCtx | null>(null);

const extOf = (name: string) => {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
};

export function UploadProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [maxSizeMb, setMaxSizeMb] = useState<number | null>(null);
  const aborters = useRef(new Map<string, () => void>());
  const listeners = useRef(new Set<() => void>());
  const running = useRef(new Set<string>());

  useEffect(() => {
    api.getSettings().then((s) => setMaxSizeMb(s.max_upload_size_mb)).catch(() => setMaxSizeMb(null));
  }, []);

  const patch = useCallback((id: string, p: Partial<UploadItem>) => {
    setItems((list) => list.map((it) => (it.id === id ? { ...it, ...p } : it)));
  }, []);

  const start = useCallback(
    async (item: UploadItem) => {
      running.current.add(item.id);
      patch(item.id, { state: "uploading", progress: 0, error: undefined });
      const { promise, abort } = uploadFile(item.file, (f) =>
        patch(item.id, f >= 1 ? { progress: 1, state: "finalizing" } : { progress: f }),
      );
      aborters.current.set(item.id, abort);
      try {
        const { status, body } = await promise;
        const res = (body as UploadResponse | null)?.results?.[0];
        if (status >= 200 && status < 300 && res?.status === "created") {
          patch(item.id, { state: "done", progress: 1, videoId: res.video?.id });
          listeners.current.forEach((cb) => cb());
        } else if (status === 409 || res?.status === "duplicate") {
          patch(item.id, { state: "duplicate", error: res?.error ?? "Already uploaded", existingVideoId: res?.existing_video_id ?? undefined });
        } else {
          const msg = res?.error ?? (body as { detail?: string } | null)?.detail ?? `Upload failed (HTTP ${status})`;
          patch(item.id, { state: "error", error: msg });
        }
      } catch (e) {
        if (e instanceof ApiError && e.status === -1) patch(item.id, { state: "cancelled", error: "Cancelled" });
        else patch(item.id, { state: "error", error: e instanceof Error ? e.message : String(e) });
      } finally {
        aborters.current.delete(item.id);
        running.current.delete(item.id);
        setItems((l) => [...l]); // trigger scheduler
      }
    },
    [patch],
  );

  // Simple scheduler: keep up to CONCURRENCY uploads running.
  useEffect(() => {
    const free = CONCURRENCY - running.current.size;
    if (free <= 0) return;
    items
      .filter((i) => i.state === "pending" && !running.current.has(i.id))
      .slice(0, free)
      .forEach((i) => void start(i));
  }, [items, start]);

  const itemsRef = useRef(items);
  itemsRef.current = items;

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const prev = itemsRef.current;
      const added: UploadItem[] = [];
      let accepted = 0;
      let rejected = 0;
      for (const file of Array.from(files)) {
        const id = `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 8)}`;
        const base: UploadItem = { id, file, name: file.name, size: file.size, state: "pending", progress: 0 };
        const ext = extOf(file.name);
        const sameInQueue = [...prev, ...added].some(
          (p) => p.name === file.name && p.size === file.size && ["pending", "uploading", "finalizing", "done"].includes(p.state),
        );
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
          added.push({ ...base, state: "error", error: `Unsupported file type ${ext || "(none)"} — use MP4, MOV or AVI` });
          rejected++;
        } else if (maxSizeMb && file.size > maxSizeMb * 1024 * 1024) {
          added.push({ ...base, state: "error", error: `${formatBytes(file.size)} exceeds the ${maxSizeMb} MB limit` });
          rejected++;
        } else if (file.size === 0) {
          added.push({ ...base, state: "error", error: "File is empty" });
          rejected++;
        } else if (sameInQueue) {
          added.push({ ...base, state: "duplicate", error: "Already in this upload queue" });
          rejected++;
        } else {
          added.push(base);
          accepted++;
        }
      }
      setItems((list) => [...list, ...added]);
      return { accepted, rejected };
    },
    [maxSizeMb],
  );

  const cancel = (id: string) => {
    const abort = aborters.current.get(id);
    if (abort) abort();
    else patch(id, { state: "cancelled", error: "Cancelled" });
  };
  const remove = (id: string) => {
    aborters.current.get(id)?.();
    setItems((l) => l.filter((i) => i.id !== id));
  };
  const retry = (id: string) => patch(id, { state: "pending", progress: 0, error: undefined });
  const clearFinished = () =>
    setItems((l) => l.filter((i) => ["pending", "uploading", "finalizing"].includes(i.state)));
  const onUploaded = useCallback((cb: () => void) => {
    listeners.current.add(cb);
    return () => {
      listeners.current.delete(cb);
    };
  }, []);

  const activeCount = items.filter((i) => ["pending", "uploading", "finalizing"].includes(i.state)).length;

  return (
    <Ctx.Provider value={{ items, maxSizeMb, addFiles, cancel, remove, retry, clearFinished, activeCount, onUploaded }}>
      {children}
    </Ctx.Provider>
  );
}

export function useUploads() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useUploads must be used inside UploadProvider");
  return c;
}
