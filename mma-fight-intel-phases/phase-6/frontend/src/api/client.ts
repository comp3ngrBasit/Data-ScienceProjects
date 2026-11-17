import type {
  Fighter,
  FighterDetail,
  FighterInput,
  FrameList,
  Job,
  JobList,
  JobStatus,
  Overview,
  Paged,
  ProcessingSettings,
  SystemInfo,
  UploadResponse,
  VideoDetail,
  VideoStatus,
  VideoSummary,
} from "./types";

export type { FrameList };

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export function apiUrl(path: string) {
  return `${BASE}${path}`;
}

function detailMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d))
      return d
        .map((e: { loc?: unknown[]; msg?: string }) => {
          const field = Array.isArray(e.loc) ? e.loc.filter((x) => x !== "body").join(".") : "";
          return field ? `${field}: ${e.msg}` : e.msg;
        })
        .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: { ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Is the backend running on port 8000?");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = undefined;
  try {
    body = text ? JSON.parse(text) : undefined;
  } catch {
    body = text;
  }
  if (!res.ok) throw new ApiError(res.status, detailMessage(body, `Request failed (${res.status})`), body);
  return body as T;
}

const qs = (params: Record<string, string | number | boolean | undefined | null>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
};

export interface VideoQuery {
  search?: string;
  status?: VideoStatus | "";
  fighter_id?: string;
  unassigned?: boolean;
  sort?: "uploaded_at" | "title" | "duration" | "file_size" | "status";
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
}

export const api = {
  // Videos
  listVideos: (q: VideoQuery = {}) =>
    request<Paged<VideoSummary>>(`/api/videos${qs({ ...q, unassigned: q.unassigned || undefined })}`),
  getVideo: (id: string) => request<VideoDetail>(`/api/videos/${id}`),
  updateVideo: (
    id: string,
    patch: Partial<{
      title: string;
      notes: string | null;
      target_fighter_id: string | null;
      target_fighter_notes: string | null;
      fighter_ids: string[];
    }>,
  ) => request<VideoDetail>(`/api/videos/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  deleteVideo: (id: string) => request<void>(`/api/videos/${id}`, { method: "DELETE" }),
  processVideo: (id: string) => request<Job>(`/api/videos/${id}/process`, { method: "POST" }),
  listFrames: (id: string, limit = 200, offset = 0) =>
    request<FrameList>(`/api/videos/${id}/frames${qs({ limit, offset })}`),

  // Jobs
  listJobs: (q: { status?: JobStatus | ""; limit?: number } = {}) => request<JobList>(`/api/jobs${qs(q)}`),
  retryJob: (id: string) => request<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),

  // Fighters
  listFighters: (search?: string) => request<Fighter[]>(`/api/fighters${qs({ search })}`),
  getFighter: (id: string) => request<FighterDetail>(`/api/fighters/${id}`),
  createFighter: (data: FighterInput) =>
    request<FighterDetail>(`/api/fighters`, { method: "POST", body: JSON.stringify(data) }),
  updateFighter: (id: string, data: FighterInput) =>
    request<FighterDetail>(`/api/fighters/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteFighter: (id: string) => request<void>(`/api/fighters/${id}`, { method: "DELETE" }),

  // System
  overview: () => request<Overview>(`/api/stats/overview`),
  getSettings: () => request<ProcessingSettings>(`/api/settings`),
  saveSettings: (s: ProcessingSettings) =>
    request<ProcessingSettings>(`/api/settings`, { method: "PUT", body: JSON.stringify(s) }),
  system: () => request<SystemInfo>(`/api/system`),
};

/** Upload a single file with progress reporting (XHR, since fetch has no upload progress). */
export function uploadFile(
  file: File,
  onProgress: (fraction: number) => void,
): { promise: Promise<{ status: number; body: UploadResponse | { detail: string } }>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{ status: number; body: UploadResponse | { detail: string } }>((resolve, reject) => {
    xhr.open("POST", apiUrl("/api/videos/upload"));
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => resolve({ status: xhr.status, body: xhr.response });
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload — is the backend running?"));
    xhr.onabort = () => reject(new ApiError(-1, "Upload cancelled"));
    const fd = new FormData();
    fd.append("files", file, file.name);
    xhr.send(fd);
  });
  return { promise, abort: () => xhr.abort() };
}
