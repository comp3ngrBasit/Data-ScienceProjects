import { ApiError, apiUrl } from "./client";
import type { Markov, Combination, Tendencies, OpponentRow, ChiSquare } from "./patterns";
import type { Prediction, Report } from "./predictions";

export interface Overview {
  totals: {
    videos: number; videos_processed: number; videos_analysed: number; fights_analysed: number; fighters_analysed: number;
    fighter_profiles: number; actions_detected: number; verified_annotations: number; actions_in_analysis: number;
    combinations_identified: number; predictions_generated: number; predictions_evaluated_held_out: number; held_out_top1: number; held_out_top3: number;
  };
  models: {
    action_recognition: { version: string; algorithm: string; f1_macro: number | null; accuracy: number | null; n_evaluated: number | null } | null;
    next_move: { id: string; name: string; experimental: boolean; top1: number | null; top3: number | null; log_loss: number | null; n: number | null } | null;
  };
  status: { action_recognition: string; next_move: string };
  pipeline_runs: { id: string; video_id: string; status: string; current_stage: string | null; progress: number }[];
  definitions: Record<string, string>;
}
export interface FighterRow {
  fighter_key: string; name: string; is_profile: boolean; stance: string | null; weight_class: string | null; actions: number; videos: number;
  fights: number; observed_seconds: number; top_action: string | null; offensive_share: number | null; combinations: number; predictions: number;
  pattern_analysis: boolean;
}
export interface Analytics {
  fighter: string | null; sequences: number; actions: number;
  action_distribution: { action: string; label: string; group: string; count: number }[];
  top_combinations: (Combination & { fighter_key: string; fighter: string })[];
  markov: Markov | null;
  sequence_length_distribution: { length: number; count: number }[];
  group_transitions: { from: string; to: string; count: number; share: number }[];
  opponent_patterns: Record<string, { fighter: string; opponents: { opponent: string; actions: number; sufficient: boolean; counter_rate: number | null; offensive_share: number | null }[]; difference_test: ChiSquare }>;
}
export interface Intelligence {
  fighter: { fighter_key: string; name: string; is_profile: boolean; nickname: string | null; stance: string | null; weight_class: string | null; height_cm: number | null; reach_cm: number | null };
  videos: { video_id: string; title: string; duration: number | null; fight_key: string | null; fight_date: string | null; source: string | null; actions: number; attacks_per_minute: number | null; distribution: { action: string; label: string; count: number }[]; js_distance_from_overall: number | null }[];
  observed: {
    label: string; analysed_seconds: number; footage_seconds: number; actions: number; fights: number;
    strike_frequency: { action: string; label: string; count: number; share: number; ci95: [number, number] | null }[];
    group_distribution: Record<string, number>; mean_time_between_actions: number | null; median_time_between_actions: number | null; interval_samples: number;
    opening_actions: { action: string; label: string; count: number; share: number }[];
    counterattacks: { after: string; counter: string; count: number }[];
    timeline: { video_id: string; fight_key: string; start: number; end: number; category: string; label: string; group: string; confidence: number; source: string; opponent_action: string | null }[];
    movement: { samples: number; state_share?: Record<string, number>; mean_center_speed?: number | null; mean_wrist_speed?: { left: number | null; right: number | null }; mean_ankle_speed?: { left: number | null; right: number | null }; screen_direction_when_moving?: Record<string, number>; units?: string };
  };
  patterns: { label: string; combinations: Combination[]; tendencies: Tendencies | null; opponents: { opponents: OpponentRow[]; difference_test: ChiSquare; findings: string[]; note: string } | null; stored_analysis_at: string | null };
  warnings: string[];
}
export interface FighterPredictions {
  fighter_key: string; available: boolean; reason?: string; video_id?: string | null; at?: number;
  context?: { opponent_action: string | null; opponent_label: string | null; opponent_lag: number | null; distance: number | null; stance: string | null; movement: Record<string, number | null> };
  previous_actions?: { category: string; label: string; start: number | null; end: number | null }[];
  prediction?: Prediction;
  next_sequences?: { sequence: string[]; labels: string[]; probability: number }[];
  timeline?: { kind: "detected" | "predicted" | "alternative"; time: number | null; end: number | null; category: string; label: string; probability?: number; time_basis?: string }[];
  performance: { held_out_evaluated: number; top1: number; top3: number };
  disclaimer?: string;
}
export interface Stage { name: string; label: string; status: string; job_id?: string; message?: string; job_progress?: number }
export interface PipelineRun { id: string; video_id: string; status: string; progress: number; current_stage: string | null; stages: Stage[]; error_message: string | null; created_at: string | null; finished_at: string | null; created?: boolean }
export interface CompleteAnalysis {
  video: { id: string; title: string; stream_url: string; duration_seconds: number | null; fps: number | null; frame_count: number | null; status: string };
  stages: { processing: string; vision: string | null; actions: string | null; action_model: string | null; predictions: string | null };
  annotated_video_url: string | null; fps: number | null;
  tracks: { track_id: number; fighter_id: string | null; fighter: string | null; role: string; first: number; last: number; coverage: number; interruptions: number }[];
  actions: { id: string; category: string; label: string; model_category: string; start: number; end: number; confidence: number; track_id: number | null; fighter_id: string | null; fighter: string | null; review_status: string; low_confidence: boolean }[];
  combinations: { fighter_key: string; fighter: string; sequence: string[]; labels: string[]; start: number; end: number; occurrences_total: number; fights: number; classification: string }[];
  predictions: { id: string; fighter_key: string; at_time: number; top3: { action: string; label: string; probability: number }[]; sequence: { category: string; label?: string }[]; reliability: { level: string; warnings: string[]; supporting_observations: Record<string, number> }; model_version: string; in_training_data: boolean; evaluation: { status: string; actual: string | null; actual_label: string | null; top1: boolean | null; top3: boolean | null } | null }[];
  pipeline: PipelineRun | null;
  fighters: string[];
}
export interface Check { check: string; passed: boolean; detail: string }
export interface Validation { passed: boolean; checks: Check[] }
export interface ModelVersion {
  id: string; kind: "action" | "prediction"; name: string; architecture: string; is_active: boolean; dataset_version: string; created_at: string;
  metrics: Record<string, number | null>; validation: Validation | null; deployment_status: string; path: string;
  status?: string; evaluation_strategy?: string | null; error?: string | null; classes?: string[]; config?: Record<string, unknown>;
  version?: number; selected?: boolean; experimental?: boolean; dataset_info?: Record<string, unknown>; params?: Record<string, unknown>; temperature?: number;
}
export interface Deployment { id: string; kind: string; model_id: string; previous_model_id: string | null; reason: string; forced: boolean; validation: Validation | null; created_at: string }
export interface PatternEval {
  combination_detection: { status: string; videos: number; reason?: string; n2?: PatternEvalN; n3?: PatternEvalN };
  transition_reliability: { status: string; reason?: string | null; fights_half_a?: number; fights_half_b?: number; probabilities_compared?: number; mean_absolute_difference?: number | null; share_within_ci95?: number | null; min_observations_per_half?: number };
  coverage: { status: string; reason?: string; held_out_transitions?: number; previous_action_seen?: number | null; exact_transition_seen?: number | null; method?: string };
  data: { fights: number; sequences: number; actions: number };
}
export interface PatternEvalN { detected: number; ground_truth: number; matched: number; precision: number | null; precision_ci95: [number, number] | null; recall: number | null; recall_ci95: [number, number] | null; f1: number | null; frequency_mae: number | null; combinations_compared: number }
export interface Evaluation {
  action_recognition: null | {
    model: ModelVersion;
    metrics: { accuracy: number; accuracy_ci95?: [number, number]; precision_macro: number; recall_macro: number; f1_macro: number; n_evaluated: number;
      per_class: Record<string, { precision: number; recall: number; f1: number; support: number }> | { class: string; precision: number; recall: number; f1: number; support: number }[];
      confusion_matrix: { labels: string[]; matrix: number[][] }; evaluation: { strategy: string } };
    dataset: Record<string, unknown> | null; samples: { final_train_samples: number | null; evaluated: number | null }; insufficient: boolean; note: string;
  };
  pattern_recognition: PatternEval;
  next_move: null | { report: Report; samples: Record<string, { folds: number; test_examples: number; mean_train_examples: number; mean_validation_examples: number }>; insufficient: boolean; finished_at: string | null };
  rules: { splits: string; never: string };
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, headers: init?.body ? { "Content-Type": "application/json" } : undefined });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Is the backend running?");
  }
  const body = await res.json().catch(() => undefined);
  if (!res.ok) {
    const d = (body as { detail?: unknown })?.detail;
    const msg = typeof d === "string" ? d : d && typeof d === "object" && "message" in (d as object) ? String((d as { message: string }).message)
      : Array.isArray(d) ? d.map((e: { msg?: string }) => e.msg).join("; ") : `Request failed (${res.status})`;
    throw new ApiError(res.status, msg, body);
  }
  return body as T;
}
const k = encodeURIComponent;

export const platformApi = {
  overview: () => req<Overview>("/api/dashboard/overview"),
  fighters: () => req<FighterRow[]>("/api/dashboard/fighters"),
  analytics: (fighter?: string) => req<Analytics>(`/api/dashboard/analytics${fighter ? `?fighter=${k(fighter)}` : ""}`),
  intelligence: (id: string, videos?: string[]) => req<Intelligence>(`/api/fighters/${k(id)}/intelligence${videos?.length ? "?" + videos.map((v) => `video=${k(v)}`).join("&") : ""}`),
  predictions: (id: string, opts: { video_id?: string; at?: number; history?: number } = {}) => {
    const p = new URLSearchParams();
    if (opts.video_id) p.set("video_id", opts.video_id);
    if (opts.at != null) p.set("at", String(opts.at));
    if (opts.history) p.set("history", String(opts.history));
    return req<FighterPredictions>(`/api/fighters/${k(id)}/predictions${p.toString() ? `?${p}` : ""}`);
  },
  complete: (videoId: string) => req<CompleteAnalysis>(`/api/videos/${k(videoId)}/complete-analysis`),
  runPipeline: (videoId: string, force = false) => req<PipelineRun>(`/api/pipeline/videos/${k(videoId)}/run${force ? "?force=true" : ""}`, { method: "POST" }),
  pipelineRuns: (videoId: string) => req<PipelineRun[]>(`/api/pipeline/videos/${k(videoId)}`),
  evaluation: () => req<Evaluation>("/api/models/evaluation"),
  versions: () => req<{ action: ModelVersion[]; prediction: ModelVersion[]; deployments: Deployment[] }>("/api/models/versions"),
  train: (kind: "action" | "prediction") => req<{ kind: string; job_id: string; status: string }>("/api/models/train", { method: "POST", body: JSON.stringify({ kind }) }),
  activate: (kind: "action" | "prediction", model_id: string, opts: { force?: boolean; reason?: "manual" | "rollback" } = {}) =>
    req<{ active: string; previous: string | null; validation: Validation }>("/api/models/activate", { method: "POST", body: JSON.stringify({ kind, model_id, ...opts }) }),
  datasets: () => req<{ prediction: { examples: number; fights: number; fighters: number; sources: Record<string, number>; targets: Record<string, number> }; action: { note: string } }>("/api/models/datasets"),
};
