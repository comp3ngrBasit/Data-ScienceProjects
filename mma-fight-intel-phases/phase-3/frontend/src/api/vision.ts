import { ApiError, apiUrl } from "./client";
import type { JobStatus } from "./types";

export interface VisionJob {
  id: string;
  video_id: string;
  video_title: string | null;
  status: JobStatus;
  attempt: number;
  progress: number;
  current_stage: string | null;
  error_message: string | null;
  warnings: string[] | null;
  model_name: string | null;
  device: string | null;
  stats: VisionStats | null;
  settings_snapshot: Record<string, unknown> | null;
  has_annotated_video: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
}

export interface VisionStats {
  frames_total: number;
  frames_analysed: number;
  frames_missing: number;
  stride: number;
  analysis_fps: number;
  frames_with_people: number;
  max_people_in_frame: number;
  detections: number;
  landmarks: number;
  tracks: number;
  tracks_non_fragment: number;
  fragment_tracks: number;
  suggested_fighter_tracks: number[];
  avg_detection_confidence: number | null;
  avg_pose_confidence: number | null;
  fighter_track_coverage: number | null;
  fighter_track_interruptions: number;
  scene_cuts: number[];
  processing_seconds: number;
  inference_ms_per_frame: number;
  realtime_factor: number | null;
  model: string;
  device: string;
  tracker: string;
  width: number;
  height: number;
  fps: number;
  superseded_by?: string;
}

export interface VisionStatus {
  video_id: string;
  video_title: string;
  video_status: string;
  can_analyze: boolean;
  reason: string | null;
  active_job: VisionJob | null;
  current: VisionJob | null;
  annotated_video_url: string | null;
  history: VisionJob[];
}

export type Role = "fighter" | "referee" | "other" | "unreviewed";

export interface Track {
  track_id: number;
  color: string;
  role: Role;
  suggested_role: "fighter" | "other";
  fighter_score: number;
  assigned_fighter: { id: string; name: string; stance: string } | null;
  first_frame: number;
  last_frame: number;
  first_timestamp: number;
  last_timestamp: number;
  frames_detected: number;
  coverage: number;
  video_coverage: number;
  interruptions: number;
  segments: [number, number][];
  avg_confidence: number;
  avg_box_height: number;
  avg_pose_confidence: number;
  status: "continuous" | "interrupted" | "fragment";
  reviewed: boolean;
}

export interface Tracking {
  job_id: string;
  analysed_frames: number;
  stride: number;
  fps: number;
  scene_cuts: number[];
  tracks: Track[];
  consistency: {
    tracks_total: number;
    tracks_non_fragment: number;
    fragment_tracks: number;
    max_people_in_frame: number;
    fighter_tracks_basis: string;
    fighter_coverage: number | null;
    fighter_video_coverage: number | null;
    fighter_interruptions: number;
    note: string;
  };
}

/** [track_id, conf, x1, y1, x2, y2, [x,y,c]*17] — normalised */
export type OverlayPerson = [number, number, number, number, number, number, number[]];
export interface OverlayChunk {
  job_id: string;
  start_frame: number;
  end_frame: number;
  frames: [number, OverlayPerson[]][];
}

export interface MovementRow {
  track_id: number;
  frame_number: number;
  timestamp: number;
  state: "moving" | "stationary" | "low_confidence" | "missing";
  valid_landmarks: number;
  center_x: number | null;
  center_y: number | null;
  torso_px: number | null;
  center_speed: number | null;
  direction_deg: number | null;
  left_wrist_speed: number | null;
  right_wrist_speed: number | null;
  left_elbow_speed: number | null;
  right_elbow_speed: number | null;
  left_shoulder_speed: number | null;
  right_shoulder_speed: number | null;
  left_hip_speed: number | null;
  right_hip_speed: number | null;
  left_knee_speed: number | null;
  right_knee_speed: number | null;
  left_ankle_speed: number | null;
  right_ankle_speed: number | null;
  left_elbow_angle: number | null;
  right_elbow_angle: number | null;
  left_knee_angle: number | null;
  right_knee_angle: number | null;
  torso_lean_deg: number | null;
  stance_width: number | null;
  guard_height: number | null;
  posture_change: number | null;
}

export interface FrameDetail {
  requested_frame: number;
  analysed_frame: number;
  timestamp?: number | null;
  people: {
    track_id: number;
    confidence: number;
    role: Role | null;
    fighter_name: string | null;
    bbox: number[];
    keypoints: { landmark: string; x: number; y: number; confidence: number }[];
    movement: MovementRow | null;
  }[];
  distances: { track_a: number; track_b: number; distance_frame: number; distance_torso: number | null; closing_speed: number | null }[];
}

export interface DistanceSeries {
  pairs: { track_a: number; track_b: number; frames: number }[];
  items: { frame_number: number; timestamp: number; distance_frame: number; distance_torso: number | null; closing_speed: number | null }[];
}

export interface VisionSettings {
  model: string;
  device: string;
  half_precision: boolean;
  image_size: number;
  analysis_fps: number;
  detection_confidence: number;
  iou_threshold: number;
  keypoint_confidence: number;
  tracker: "bytetrack" | "botsort";
  track_buffer_seconds: number;
  min_box_height: number;
  max_people_per_frame: number;
  generate_annotated_video: boolean;
  annotated_max_width: number;
  smoothing_window: number;
  max_gap_interpolation: number;
  stationary_threshold: number;
}

export interface VisionSystem {
  torch: { torch: string | null; cuda_available: boolean; cuda_version: string | null; gpus: string[]; mps_available: boolean; cpu_threads?: number; error?: string };
  available_models: string[];
  downloaded_models: string[];
  keypoints: string[];
}

export interface VisionOverview {
  videos_analysed: number;
  videos_eligible: number;
  job_counts: Record<JobStatus, number>;
  total_frames_analysed: number;
  fighter_tracks_assigned: number;
  fighter_tracks_suggested: number;
  distinct_fighters_identified: number;
  avg_detection_confidence: number | null;
  avg_pose_confidence: number | null;
  tracking: { avg_fighter_track_coverage: number | null; fighter_track_interruptions: number; fighter_tracks: number };
  active_jobs: VisionJob[];
  recent: VisionJob[];
  latest_annotated: { video_id: string; title: string; url: string } | null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, headers: init?.body ? { "Content-Type": "application/json" } : undefined });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Is the backend running?");
  }
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => undefined);
  if (!res.ok) {
    const d = (body as { detail?: unknown })?.detail;
    throw new ApiError(res.status, typeof d === "string" ? d : `Request failed (${res.status})`, body);
  }
  return body as T;
}

export const visionApi = {
  status: (id: string) => req<VisionStatus>(`/api/vision/${id}/status`),
  analyze: (id: string) => req<VisionJob>(`/api/vision/${id}/analyze`, { method: "POST" }),
  deleteAnalysis: (id: string) => req<void>(`/api/vision/${id}/analysis`, { method: "DELETE" }),
  retry: (jobId: string) => req<VisionJob>(`/api/vision/jobs/${jobId}/retry`, { method: "POST" }),
  jobs: (status?: string) => req<{ items: VisionJob[]; counts: Record<JobStatus, number> }>(`/api/vision/jobs${status ? `?status=${status}` : ""}`),
  tracking: (id: string) => req<Tracking>(`/api/vision/${id}/tracking`),
  assign: (id: string, assignments: { track_id: number; fighter_id?: string | null; role?: Role | null }[]) =>
    req<Tracking>(`/api/vision/${id}/assign-fighter`, { method: "POST", body: JSON.stringify({ assignments }) }),
  overlay: (id: string, start: number, end: number) => req<OverlayChunk>(`/api/vision/${id}/overlay?start_frame=${start}&end_frame=${end}`),
  movement: (id: string, trackId: number) => req<{ items: MovementRow[]; units: Record<string, string> }>(`/api/vision/${id}/movement?track_id=${trackId}&limit=50000`),
  distances: (id: string, a?: number, b?: number) =>
    req<DistanceSeries>(`/api/vision/${id}/distances${a != null && b != null ? `?track_a=${a}&track_b=${b}` : ""}`),
  frame: (id: string, frame: number) => req<FrameDetail>(`/api/vision/${id}/frame/${frame}`),
  overview: () => req<VisionOverview>(`/api/vision/overview`),
  settings: () => req<VisionSettings>(`/api/vision/settings`),
  saveSettings: (s: VisionSettings) => req<VisionSettings>(`/api/vision/settings`, { method: "PUT", body: JSON.stringify(s) }),
  system: () => req<VisionSystem>(`/api/vision/system`),
  exportUrl: (id: string, kind: string) => apiUrl(`/api/vision/${id}/export?kind=${kind}`),
};

export const KEYPOINTS = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
  "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
];
const K = (n: string) => KEYPOINTS.indexOf(n);
export const SKELETON: [number, number][] = [
  ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"], ["left_hip", "right_hip"], ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"], ["nose", "left_eye"], ["nose", "right_eye"],
  ["left_eye", "left_ear"], ["right_eye", "right_ear"],
].map(([a, b]) => [K(a), K(b)] as [number, number]);

export const PALETTE = ["#ef4444", "#38bdf8", "#facc15", "#4ade80", "#c084fc", "#fb923c", "#f472b6", "#2dd4bf", "#a3e635", "#818cf8"];
export const trackColor = (id: number) => PALETTE[(Math.max(id, 1) - 1) % PALETTE.length];

export const STAGE_LABELS: Record<string, string> = {
  starting: "Starting",
  loading_model: "Loading pose model",
  analyzing: "Detecting, tracking & estimating pose",
  encoding: "Encoding annotated video",
  tracking_summary: "Summarising tracks",
  movement: "Computing movement",
  export: "Writing export bundle",
  finalizing: "Finalising",
};
