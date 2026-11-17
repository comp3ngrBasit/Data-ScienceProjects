import { ApiError, apiUrl } from "./client";

export type ActionGroup = "offensive" | "defensive" | "movement" | "unknown";
export interface Category {
  key: string;
  label: string;
  group: ActionGroup;
  family: string;
  coarse: boolean;
  trained: boolean;
}

export interface ActionJob {
  id: string;
  video_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  current_stage: string | null;
  model_version: string;
  attempt: number;
  stats: Record<string, unknown> & {
    actions?: number;
    by_category?: Record<string, number>;
    by_group?: Record<string, number>;
    model_classes?: string[];
    repeated_combinations?: Record<string, { combination: string[]; keys: string[]; count: number }[]>;
    pose_rate_fps?: number;
    windows_total?: number;
    windows_classified?: number;
    windows_idle?: number;
    windows_unknown?: number;
    duplicates_removed?: number;
    window_seconds?: number;
  } | null;
  warnings: string[] | null;
  error_message: string | null;
  settings: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface RecognizedAction {
  id: string;
  video_id: string;
  track_id: number;
  fighter_id: string | null;
  fighter_name: string | null;
  category: string;
  label: string;
  group: ActionGroup;
  model_category: string;
  start_time: number;
  end_time: number;
  duration: number;
  start_frame: number;
  end_frame: number;
  confidence: number;
  runner_up: string | null;
  runner_up_label: string | null;
  runner_up_confidence: number | null;
  validated_precision: number | null;
  validated_support: number | null;
  low_confidence: boolean;
  limb: string | null;
  windows: number;
  model_version: string;
  review_status: "unreviewed" | "confirmed" | "corrected" | "rejected";
  threshold: number;
}

export interface Annotation {
  id: string;
  video_id: string;
  fighter_id: string | null;
  fighter_name: string | null;
  track_id: number | null;
  category: string;
  label: string;
  group: ActionGroup;
  start_time: number;
  end_time: number;
  duration: number;
  confidence: number;
  opponent_reaction: string | null;
  status: "draft" | "verified" | "needs_review" | "rejected";
  source: "manual" | "import" | "correction";
  notes: string | null;
}

export interface ActionStatus {
  video_id: string;
  video_title: string;
  can_analyze: boolean;
  reason: string | null;
  active_job: ActionJob | null;
  current: ActionJob | null;
  history: ActionJob[];
  active_model: string | null;
  vision: { job_id: string; stride: number; pose_fps: number; annotated_video: boolean } | null;
}

export interface Lane {
  key: string;
  fighter_id: string | null;
  name: string;
  track_ids: number[];
  actions: RecognizedAction[];
  annotations: Annotation[];
}

export interface SequenceAction {
  action_id: string;
  category: string;
  label: string;
  group: ActionGroup;
  start: number;
  end: number;
  confidence: number;
  source: string;
}
export interface Sequence {
  id?: string;
  fighter_key: string;
  fighter_id: string | null;
  fighter_name: string;
  track_ids: number[];
  sequence_index: number;
  start_time: number;
  end_time: number;
  actions: SequenceAction[];
  intervals: number[];
  n_actions: number;
  summary: string;
}

export interface ModelSummary {
  version: string;
  algorithm: string;
  status: string;
  is_active: boolean;
  dataset_version: string;
  classes: string[];
  class_labels: string[];
  created_at: string;
  accuracy: number | null;
  precision_macro: number | null;
  recall_macro: number | null;
  f1_macro: number | null;
  error: string | null;
  evaluation_strategy: string | null;
  n_evaluated: number | null;
  training_job_id: string | null;
}

export interface ClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  confident_predictions: number;
  precision_at_threshold: number | null;
}

export interface ModelDetail extends ModelSummary {
  metrics: {
    accuracy?: number;
    accuracy_ci95?: [number, number] | null;
    majority_class_baseline?: number | null;
    precision_macro?: number;
    recall_macro?: number;
    f1_macro?: number;
    precision_weighted?: number;
    recall_weighted?: number;
    f1_weighted?: number;
    per_class?: Record<string, ClassMetrics>;
    confusion_matrix?: { labels: string[]; display_labels: string[]; matrix: number[][] };
    n_evaluated?: number;
    threshold?: number;
    evaluation?: { strategy: string; folds?: number };
    error?: string;
  };
  training_info: Record<string, unknown> & { dataset?: DatasetSummary; dropped_classes?: Record<string, string>; train_seconds?: number; device?: string };
  config: Record<string, unknown>;
}

export interface TrainingJob {
  id: string;
  status: ActionJob["status"];
  progress: number;
  current_stage: string | null;
  config: Record<string, unknown>;
  dataset_version: string | null;
  dataset_summary: DatasetSummary | null;
  results: { comparison: Record<string, { version: string; accuracy: number | null; f1_macro: number | null; precision_macro: number | null; recall_macro: number | null; strategy: string; error: string | null }>; best: string | null; classes: string[]; dropped: Record<string, string> } | null;
  warnings: string[] | null;
  error_message: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface DatasetSummary {
  version: string;
  samples: number;
  class_counts: Record<string, number>;
  groups_per_class: Record<string, number>;
  groups: number;
  videos: number;
  annotations_used: number;
  annotations_total: number;
  skipped: Record<string, number>;
}

export interface DatasetPreview extends DatasetSummary {
  warnings: string[];
  class_table: { key: string; label: string; examples: number; groups: number; trainable: boolean }[];
  requirements: { min_examples: number; min_groups: number };
}

export interface ActionSettings {
  confidence_threshold: number;
  motion_gate: number;
  unknown_motion: number;
  hop_fraction: number;
  window_seconds: number | null;
  nms_iou: number;
  min_visible_fraction: number;
  sequence_gap: number;
  fighters_only: boolean;
  algorithms: string[];
  min_examples: number;
  min_groups: number;
  augment: number;
  negatives_per_positive: number;
  seq_len: number;
  context: number;
  include_draft_annotations: boolean;
  epochs: number;
  device: string;
}

export interface ActionOverview {
  active_model: ModelSummary | null;
  annotations: { total: number; verified: number; by_category: Record<string, number> };
  videos_analysed: number;
  actions_detected: number;
  by_category: { key: string; label: string; group: ActionGroup; count: number }[];
  by_group: Record<string, number>;
  job_counts: Record<string, number>;
  active_jobs: (ActionJob & { video_title: string })[];
  training: TrainingJob | null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, headers: init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Is the backend running?");
  }
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => undefined);
  if (!res.ok) {
    const d = (body as { detail?: unknown })?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((e: { msg?: string }) => e.msg).join("; ") : `Request failed (${res.status})`;
    throw new ApiError(res.status, msg, body);
  }
  return body as T;
}

export const actionsApi = {
  taxonomy: () => req<{ categories: Category[]; active_model: string | null }>(`/api/actions/taxonomy`),
  status: (id: string) => req<ActionStatus>(`/api/actions/${id}/status`),
  analyze: (id: string, version?: string) => req<ActionJob>(`/api/actions/${id}/analyze${version ? `?model_version=${encodeURIComponent(version)}` : ""}`, { method: "POST" }),
  deleteResults: (id: string) => req<void>(`/api/actions/${id}/results`, { method: "DELETE" }),
  results: (id: string) => req<{ items: RecognizedAction[]; confidence_note: string; model_version: string }>(`/api/actions/${id}/results`),
  timeline: (id: string) => req<{ duration: number | null; fps: number | null; job_id: string | null; model_version: string | null; lanes: Lane[] }>(`/api/actions/${id}/timeline`),
  sequences: (id: string, source: "model" | "annotations" = "model") =>
    req<{ sequences: Sequence[]; fighters: { fighter_key: string; fighter_name: string; sequences: number; actions: SequenceAction[]; intervals: number[] }[]; format_note: string }>(`/api/actions/${id}/sequences?source=${source}`),
  review: (actionId: string, decision: "confirm" | "correct" | "reject", category?: string) =>
    req<RecognizedAction>(`/api/actions/actions/${actionId}/review`, { method: "POST", body: JSON.stringify({ decision, category }) }),
  annotations: (id: string) => req<{ items: Annotation[]; counts: Record<string, number>; meta: { fully_annotated: boolean; split: string; split_group: string | null } }>(`/api/actions/${id}/annotations`),
  createAnnotation: (id: string, a: Partial<Annotation>) => req<Annotation>(`/api/actions/${id}/annotations`, { method: "POST", body: JSON.stringify(a) }),
  updateAnnotation: (annId: string, a: Partial<Annotation>) => req<Annotation>(`/api/actions/annotations/${annId}`, { method: "PUT", body: JSON.stringify(a) }),
  deleteAnnotation: (annId: string) => req<void>(`/api/actions/annotations/${annId}`, { method: "DELETE" }),
  setMeta: (id: string, m: { fully_annotated?: boolean; split?: string; split_group?: string | null }) => req(`/api/actions/${id}/dataset-meta`, { method: "PUT", body: JSON.stringify(m) }),
  importFile: (id: string, fd: FormData) => req<{ created: number; errors: string[]; error_count: number }>(`/api/actions/${id}/import`, { method: "POST", body: fd }),
  modelsStatus: () => req<{ active_model: ModelSummary | null; models: ModelSummary[]; training_jobs: TrainingJob[]; training_active: boolean }>(`/api/actions/models/status`),
  train: (body: { algorithms?: string[]; activate_best?: boolean }) => req<TrainingJob>(`/api/actions/models/train`, { method: "POST", body: JSON.stringify(body) }),
  evaluation: (version?: string) => req<ModelDetail>(`/api/actions/models/evaluation${version ? `?version=${encodeURIComponent(version)}` : ""}`),
  activate: (version: string) => req<ModelSummary>(`/api/actions/models/${encodeURIComponent(version)}/activate`, { method: "POST" }),
  deleteModel: (version: string) => req<void>(`/api/actions/models/${encodeURIComponent(version)}`, { method: "DELETE" }),
  dataset: () => req<DatasetPreview>(`/api/actions/dataset`),
  settings: () => req<ActionSettings>(`/api/actions/settings`),
  saveSettings: (s: ActionSettings) => req<ActionSettings>(`/api/actions/settings`, { method: "PUT", body: JSON.stringify(s) }),
  overview: () => req<ActionOverview>(`/api/actions/overview`),
  exportUrl: (id: string, kind: "actions" | "sequences", format: "json" | "csv") => apiUrl(`/api/actions/${id}/export?kind=${kind}&format=${format}`),
};

export const GROUP_COLORS: Record<ActionGroup, string> = {
  offensive: "#ef4444",
  defensive: "#38bdf8",
  movement: "#a3a3ad",
  unknown: "#facc15",
};
export const GROUP_LABELS: Record<ActionGroup, string> = {
  offensive: "Offensive",
  defensive: "Defensive",
  movement: "Movement",
  unknown: "Unknown",
};
export const ALGO_LABELS: Record<string, string> = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  svm: "SVM (RBF)",
  lstm: "Bi-LSTM",
  tcn: "TCN",
};
