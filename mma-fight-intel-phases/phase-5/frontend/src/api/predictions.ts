import { ApiError, apiUrl } from "./client";

export type Level = "high" | "medium" | "low" | "insufficient";
export type CI = [number, number] | null;

export interface DistItem { action: string; label: string; probability: number }
export interface Reliability {
  level: Level;
  experimental_model: boolean;
  predicted_probability: number;
  uncalibrated_model_confidence: number;
  historical_accuracy_at_this_confidence: { bin: [number, number]; n: number; observed_accuracy: number } | null;
  uncertainty_normalised_entropy: number;
  supporting_observations: Record<string, number>;
  training_examples: number | null;
  training_fights: number | null;
  calibration: { method: string; temperature: number; cv_ece: number | null; cv_brier: number | null; cv_log_loss: number | null };
  cv_top1_accuracy: number | null;
  cv_top3_accuracy: number | null;
  warnings: string[];
  explanation: string;
}
export interface Evidence {
  supporting_combinations: { sequence: string[]; scope: string; count: number; of: number; fights: number }[];
  fighter_tendencies: {
    fighter_training_examples: number;
    most_frequent_actions: { action: string; share: number }[];
    usual_follow_ups_after_last_action: { action: string; count: number; of: number }[];
  };
  context_notes: string[];
}
export interface Prediction {
  top3: DistItem[];
  distribution: DistItem[];
  predicted_action: string;
  predicted_probability: number;
  probability_sum: number;
  model: { id: string; architecture: string; version: string };
  in_training_data: boolean;
  evidence: Evidence;
  reliability: Reliability;
  without_context?: DistItem[];
  context_shift?: { action: string; label: string; change: number }[];
  prediction_id?: string;
  sequence: { category: string; label: string }[];
}
export interface Metrics {
  n: number;
  top1_accuracy: number;
  top1_ci95: CI;
  top3_accuracy: number;
  top3_ci95: CI;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  log_loss: number;
  brier: number;
  ece: number;
  reliability_diagram: { bin: [number, number]; n: number; mean_confidence: number; accuracy: number }[];
  confusion_matrix?: { labels: string[]; matrix: number[][] };
  coverage?: number;
  breakdown?: Record<string, Omit<Metrics, "breakdown"> | null>;
  vs_markov_top1?: { difference: number; ci95: [number, number]; resampled_fights: number; significant: boolean };
  status?: string;
}
export interface Report {
  status: "ok" | "insufficient_data";
  reason?: string;
  data: { examples: number; fights: number; fighters: number; classes: string[]; class_labels: string[]; target_counts: Record<string, number>; sources: Record<string, number>; rare_actions: string[]; dataset_version?: string; device?: string };
  architectures?: string[];
  skipped: Record<string, string>;
  protocols: Record<string, { description: string; models: Record<string, Metrics> }>;
  selected?: string;
  selection_rule?: string;
  experimental?: boolean;
  seconds?: number;
}
export interface Job {
  id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  current_stage?: string | null;
  error_message: string | null;
  created_at: string | null;
  finished_at: string | null;
  dataset_info?: Report["data"] & { dataset_version: string; device: string };
  version?: number | null;
  video_id?: string;
  stats?: { predictions: number; evaluated: number; top1_correct: number; top3_correct: number; model: string; in_training_data: boolean } | null;
}
export interface ModelRow {
  id: string; version: number; architecture: string; name: string; is_active: boolean; selected: boolean; experimental: boolean;
  dataset_version: string; dataset_info: Job["dataset_info"]; temperature: number; created_at: string;
  summary: Partial<Metrics>;
}
export interface Overview {
  active_model: ModelRow | null;
  training_jobs: Job[];
  predictions: number;
  evaluated: number;
  held_out: { n: number; top1: number; top3: number };
  external_datasets: { dataset: string; actions: number; fights: number }[];
  settings: PredictionSettings;
}
export interface PredictionSettings {
  history_length: number; min_class_count: number; folds: number; architectures: ("gru" | "lstm" | "transformer")[];
  min_examples_deep: number; min_examples_transformer: number; min_examples_reliable: number; min_fights_reliable: number;
  include_external: boolean; device: "auto" | "cpu" | "cuda"; beam_width: number; min_support: number;
}
export interface FighterRow { fighter_key: string; name: string; source: "app" | "external"; actions: number }
export interface StoredPrediction {
  id: string; fighter_key: string; video_id: string | null; mode: string; at_time: number | null;
  sequence: { category: string; label: string; start: number | null; end: number | null }[];
  context: Record<string, unknown> | null; top3: DistItem[]; predicted_action: string; predicted_label: string;
  predicted_probability: number; reliability: Reliability; model_version: string; in_training_data: boolean; created_at: string;
  evaluation: { status: string; actual_action: string | null; actual_label: string | null; correct_top1: boolean | null; correct_top3: boolean | null; probability_of_actual: number | null; source: string; evaluated_at: string } | null;
}
export interface Agg { n: number; top1: number; top3: number; top1_rate: number | null; top1_ci95: CI; top3_rate: number | null; top3_ci95: CI }
export interface Performance { held_out: Agg; in_training_data: Agg; by_model: Record<string, Agg>; over_time: { i: number; top1_rate: number; top3_rate: number; model: string }[]; note: string }
export interface ModelStatus { ready: boolean; reason?: string; model?: ModelRow; fighter_training_examples?: number; known_fighter?: boolean; classes?: { action: string; label: string }[]; history_length?: number; note?: string | null }
export interface NextSeq { sequences: { sequence: string[]; labels: string[]; probability: number; assumed_timing: { duration: number; gap: number } }[]; note: string; model_version: string }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, headers: init?.body && typeof init.body === "string" ? { "Content-Type": "application/json" } : undefined });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Is the backend running?");
  }
  const body = await res.json().catch(() => undefined);
  if (!res.ok) {
    const d = (body as { detail?: unknown })?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((e: { msg?: string }) => e.msg).join("; ") : `Request failed (${res.status})`;
    throw new ApiError(res.status, msg, body);
  }
  return body as T;
}
const k = encodeURIComponent;

export const predictionsApi = {
  overview: () => req<Overview>("/api/predictions"),
  fighters: () => req<FighterRow[]>("/api/predictions/fighters"),
  settings: () => req<PredictionSettings>("/api/predictions/settings"),
  saveSettings: (s: PredictionSettings) => req<PredictionSettings>("/api/predictions/settings", { method: "PUT", body: JSON.stringify(s) }),
  train: () => req<Job>("/api/predictions/models/train", { method: "POST" }),
  jobs: () => req<Job[]>("/api/predictions/models/jobs"),
  versions: () => req<ModelRow[]>("/api/predictions/models/versions"),
  evaluation: () => req<{ job: Job; report: Report }>("/api/predictions/models/evaluation"),
  activate: (id: string) => req<ModelRow>(`/api/predictions/models/${k(id)}/activate`, { method: "POST" }),
  predict: (fighter: string, body: { history: { category: string; start?: number | null; end?: number | null }[] | string[]; context?: Record<string, unknown>; mode?: string }) =>
    req<Prediction>(`/api/predictions/${k(fighter)}/predict`, { method: "POST", body: JSON.stringify(body) }),
  nextSequence: (fighter: string, history: string[], depth = 3, extra: { opponent_action?: string; distance?: number } = {}) => {
    const p = new URLSearchParams();
    history.forEach((h) => p.append("history", h));
    p.set("depth", String(depth));
    if (extra.opponent_action) p.set("opponent_action", extra.opponent_action);
    if (extra.distance != null) p.set("distance", String(extra.distance));
    return req<NextSeq>(`/api/predictions/${k(fighter)}/next-sequence?${p}`);
  },
  evaluate: (fighter: string, prediction_id: string, actual_action: string) =>
    req<StoredPrediction>(`/api/predictions/${k(fighter)}/evaluate`, { method: "POST", body: JSON.stringify({ prediction_id, actual_action }) }),
  history: (fighter: string) => req<StoredPrediction[]>(`/api/predictions/${k(fighter)}/history`),
  performance: (fighter: string) => req<Performance>(`/api/predictions/${k(fighter)}/performance`),
  modelStatus: (fighter: string) => req<ModelStatus>(`/api/predictions/${k(fighter)}/model-status`),
  videoPredict: (video: string, fighter?: string) => req<Job>(`/api/predictions/video/${k(video)}/predict${fighter ? `?fighter_key=${k(fighter)}` : ""}`, { method: "POST" }),
  timeline: (video: string) => req<{ job: Job | null; predictions: StoredPrediction[] }>(`/api/predictions/video/${k(video)}/timeline`),
  importDataset: (form: FormData) => req<{ imported: number; error_count: number; errors: string[] }>("/api/predictions/datasets/import", { method: "POST", body: form }),
};

export const LEVEL: Record<Level, { label: string; cls: string }> = {
  high: { label: "High reliability", cls: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30" },
  medium: { label: "Medium reliability", cls: "bg-sky-500/15 text-sky-300 ring-sky-500/30" },
  low: { label: "Low reliability", cls: "bg-amber-500/15 text-amber-300 ring-amber-500/30" },
  insufficient: { label: "Insufficient data", cls: "bg-ink-700 text-ink-300 ring-ink-600" },
};
export const ARCH_LABEL: Record<string, string> = {
  markov: "Markov baseline", xgboost: "XGBoost", gru: "GRU", lstm: "LSTM", transformer: "Transformer", class_frequency: "Class frequency (floor)",
};
