import { ApiError, apiUrl } from "./client";

export type Reliability = "high" | "medium" | "low" | "single_fight" | "insufficient";
export type CI = [number, number] | null;
export type Definition = "next" | "next_offensive" | "window";

export interface PatternFilters {
  fight?: string[];
  date_from?: string;
  date_to?: string;
  min_confidence?: number;
}

export interface FighterIndexRow {
  fighter_key: string;
  name: string;
  is_track: boolean;
  actions: number;
  fights: number;
  videos: number;
  fight_keys: string[];
  dates: string[];
  analysis: "up_to_date" | "stale" | "not_analysed";
  analysed_at: string | null;
  stance: string | null;
}

export interface PatternJob {
  id: string;
  fighter_key: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  current_stage: string | null;
  dataset_version: string | null;
  stats: { analysed?: { fighter_key: string; actions: number; fights: number }[]; unchanged_skipped?: string[]; seconds?: number; video_cache_hits?: number; video_cache_misses?: number } | null;
  warnings: string[] | null;
  error_message: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface PatternVideo {
  video_id: string;
  title: string;
  source: "detections" | "annotations";
  actions: number;
  fight_key: string;
  fight_date: string | null;
  event: string | null;
  duration: number | null;
  fighters: string[];
}

export interface PatternOverview {
  fighters: FighterIndexRow[];
  profiles_without_actions: { fighter_key: string; name: string }[];
  videos: PatternVideo[];
  extraction: Record<string, number>;
  settings: PatternSettings;
  jobs: PatternJob[];
  totals: { actions: number; fighters: number; videos: number; fights: number };
}

export interface PatternSettings {
  source: "best" | "detections" | "annotations";
  min_confidence: number;
  include_low_confidence: boolean;
  sequence_gap: number;
  window: number;
  overlap_iou: number;
  min_occurrences: number;
  min_fights: number;
  min_observations: number;
  max_n: number;
  smoothing_alpha: number;
  max_order: number;
  min_context_count: number;
  significance: number;
  opponent_reaction_window: number;
  min_actions_per_opponent: number;
}

export interface Meta {
  mode: "snapshot" | "live";
  snapshot_at: string | null;
  stale: boolean;
  dataset_version: string;
  filters: Record<string, unknown>;
  available_fights: string[];
}

export interface Summary {
  actions: number;
  sequences: number;
  fights: number;
  videos: number;
  fight_keys: string[];
  period: { from: string; to: string } | null;
  minutes_observed: number;
  sources: Record<string, number>;
  multi_action_sequences: number;
  mean_sequence_length: number | null;
  preparation: Record<string, number>;
  reliability_counts_first_order: Record<string, number>;
  probabilities_meeting_thresholds: number;
  probabilities_total: number;
  single_fight_patterns: number;
  cross_fight_patterns: number;
}

export interface Combination {
  sequence: string[];
  labels: string[];
  length: number;
  occurrences: number;
  distinct_fights: number;
  distinct_videos: number;
  frequency_per_fight: number;
  avg_interval: number | null;
  avg_duration: number;
  avg_confidence: number;
  followed_by_action_pct: number;
  confidence: number | null;
  confidence_n: number;
  confidence_ci95: CI;
  lift: number | null;
  p_value: number | null;
  classification: "repeated_pattern" | "single_fight" | "not_distinguishable_from_chance";
}

export interface SeqPattern {
  pattern: string[];
  labels: string[];
  length: number;
  support: number;
  sequences_containing: number;
  distinct_fights: number;
  confidence: number | null;
  avg_time_between: number | null;
  contiguous_only: boolean;
  single_fight: boolean;
}

export interface Probability {
  definition: Definition;
  order: number;
  context: string[];
  context_labels: string[];
  next: string;
  next_label: string;
  count: number;
  context_count: number;
  probability: number;
  ci95: CI;
  fights: number;
  context_fights: number;
  coverage: number;
  reliability: Reliability;
  window_seconds: number | null;
}

export interface MarkovCell { next: string; count: number; raw: number; smoothed: number; ci95: CI }
export interface Markov {
  order: number;
  states: string[];
  columns: string[];
  column_labels: string[];
  smoothing: { method: string; alpha: number };
  rows: { context: string[]; context_labels: string[]; n: number; fights: number; most_likely_next: string | null; cells: MarkovCell[]; reliability: Reliability }[];
  start_distribution: { action: string; label: string; count: number; probability: number }[];
  most_common: { from: string[]; to: string; count: number; probability: number }[];
  rare: { from: string[]; to: string; count: number; probability: number }[];
  observations: number;
  entropy_rate_bits: number;
  insufficient_contexts: number;
}

export interface Dist { action: string; label: string; count: number; share: number; ci95: CI; per_minute: number | null; fights: number }
export interface Reaction {
  opponent_attack: string;
  label: string;
  n: number;
  fights: number;
  responses: { action: string; label: string; count: number; probability: number; ci95: CI }[];
  defensive_responses: number;
  offensive_responses: number;
  reliability: Reliability;
}
export interface Diversity { distinct_actions: number; action_entropy_bits: number; distinct_transitions: number; transition_entropy_bits: number; distinct_transition_ratio: number | null }
export interface Tendencies {
  actions: number;
  fights: number;
  minutes_observed: number;
  single_fight_only: boolean;
  group_counts: Record<string, number>;
  offensive: {
    most_used: Dist[];
    attack_frequency_per_minute: number | null;
    share_of_actions: number | null;
    average_combination_length: number | null;
    top_combinations: Combination[];
    preferred_attacking_sequences: Combination[];
  };
  defensive: {
    most_used: Dist[];
    share_of_actions: number | null;
    evasions: number;
    evasion_rate_per_opponent_attack: number | null;
    counterattacks_labelled: number;
    counterattack_rate: number | null;
    counterattack_ci95: CI;
    reactions_to_opponent_attacks: { opponent_attacks: number; window_seconds: number; by_attack: Reaction[] };
  };
  behavioral: {
    repeated_sequences: Combination[];
    typical_interval: { median: number; p25: number; p75: number; n: number } | null;
    changes_across_fights: { fight_key: string; actions: number; js_distance_from_overall: number | null; top: string[] }[];
    homogeneity_test: ChiSquare;
    sequence_diversity: Diversity;
  };
}
export interface ChiSquare { valid: boolean; chi2?: number; p_value?: number; dof?: number; reason?: string | null }

export interface SeqAction {
  action_id: string; category: string; label: string; group: string; start: number; end: number; duration: number;
  interval_prev: number | null; confidence: number; source: string; opponent_action: string | null; opponent_action_label: string | null;
  opponent_action_lag: number | null; distance: number | null; stance: string | null; track_id: number | null;
}
export interface Sequence {
  id: string; video_id: string; video_title: string | null; fight_key: string; fight_date: string | null;
  start: number; end: number; n_actions: number; categories: string[]; labels: string[]; actions: SeqAction[];
}
export interface ExplorerItem { action: string; label: string; count: number; probability: number; ci95: CI }
export interface Explorer {
  action: string; label: string; occurrences: number; fights: number; reliability: Reliability;
  before: ExplorerItem[]; after: ExplorerItem[];
  three_action: { sequence: string[]; labels: string[]; count: number; fights: number }[];
  five_action: { sequence: string[]; labels: string[]; count: number; fights: number }[];
}

export interface OpponentRow {
  opponent_key: string; opponent: string; fights: number; actions: number;
  opponent_style: string; opponent_style_vector: Record<string, number>;
  action_distribution: { action: string; label: string; count: number; share: number }[];
  offensive_share: number | null; combinations: Combination[];
  after_opponent_attacks: { opponent_attacks: number; counter_rate: number | null; counter_ci95: CI; defensive_rate: number | null; defensive_ci95: CI; detail: Reaction[] };
  sufficient: boolean;
}

interface Base { fighter_key: string; name: string; meta: Meta; warnings: string[] }
export interface FighterSummary extends Base {
  is_track: boolean; summary: Summary; config: PatternSettings; states: string[]; state_labels: Record<string, string>;
  top_combinations: Combination[]; most_likely_next: Markov["rows"];
}
export interface CompareResult {
  fighters: Record<string, {
    actions: number; fights: number; minutes_observed: number; attack_frequency_per_minute: number | null;
    offensive_share: number | null; defensive_share: number | null; most_used: Dist[]; defensive: Dist[];
    counterattack_rate: number | null; top_combinations: Combination[]; diversity: Diversity;
  }>;
  states: string[];
  state_labels: Record<string, string>;
  markov: Record<string, Markov>;
  action_distribution_js_distance: number | null;
  difference_test: ChiSquare;
  names: Record<string, string>;
  warnings: string[];
}
export interface FightRow { video_id: string; title: string; duration: number | null; fight_key: string | null; fight_date: string | null; event: string | null; notes: string | null }
export interface AnalysisStatus {
  fighter_key: string; job: PatternJob | null;
  snapshot: { id: string; created_at: string; dataset_version: string } | null;
  current_dataset_version: string; state: "processing" | "up_to_date" | "stale" | "not_analysed";
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
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((e: { msg?: string }) => e.msg).join("; ") : `Request failed (${res.status})`;
    throw new ApiError(res.status, msg, body);
  }
  return body as T;
}

function q(f: PatternFilters = {}, extra: Record<string, string | number | undefined | string[]> = {}) {
  const p = new URLSearchParams();
  (f.fight ?? []).forEach((x) => p.append("fight", x));
  if (f.date_from) p.set("date_from", f.date_from);
  if (f.date_to) p.set("date_to", f.date_to);
  if (f.min_confidence != null) p.set("min_confidence", String(f.min_confidence));
  Object.entries(extra).forEach(([k, v]) => {
    if (Array.isArray(v)) v.forEach((x) => p.append(k, x));
    else if (v !== undefined && v !== "") p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
}
const k = encodeURIComponent;

export const patternsApi = {
  overview: () => req<PatternOverview>(`/api/patterns`),
  settings: () => req<PatternSettings>(`/api/patterns/settings`),
  saveSettings: (s: PatternSettings) => req<PatternSettings>(`/api/patterns/settings`, { method: "PUT", body: JSON.stringify(s) }),
  fights: () => req<FightRow[]>(`/api/patterns/fights`),
  saveFight: (videoId: string, body: Partial<FightRow>) => req<FightRow>(`/api/patterns/fights/${k(videoId)}`, { method: "PUT", body: JSON.stringify(body) }),
  jobs: () => req<PatternJob[]>(`/api/patterns/jobs`),
  analyzeAll: (force = false) => req<PatternJob>(`/api/patterns/analyze-all${force ? "?force=true" : ""}`, { method: "POST" }),
  summary: (id: string, f?: PatternFilters) => req<FighterSummary>(`/api/patterns/${k(id)}${q(f)}`),
  combinations: (id: string, f?: PatternFilters) =>
    req<Base & { summary: Summary; combinations: Combination[]; sequential_patterns: SeqPattern[]; thresholds: Record<string, number> }>(`/api/patterns/${k(id)}/combinations${q(f)}`),
  transitions: (id: string, order: number, f?: PatternFilters) => req<Base & { markov: Markov }>(`/api/patterns/${k(id)}/transitions${q(f, { order })}`),
  probabilities: (id: string, definition: Definition, order: number, f?: PatternFilters, action?: string[]) =>
    req<Base & { definition: Definition; definition_text: string; order: number; context: string[] | null; context_count: number; probabilities: Probability[]; min_observations: number }>(
      `/api/patterns/${k(id)}/probabilities${q(f, { definition, order: action?.length ? undefined : order, action })}`),
  tendencies: (id: string, f?: PatternFilters) => req<Base & { summary: Summary; tendencies: Tendencies }>(`/api/patterns/${k(id)}/tendencies${q(f)}`),
  sequences: (id: string, f?: PatternFilters, action?: string) =>
    req<Base & { total_sequences: number; sequences: Sequence[]; explorer?: Explorer; actions_available: { action: string; label: string }[] }>(`/api/patterns/${k(id)}/sequences${q(f, { action })}`),
  opponents: (id: string, f?: PatternFilters) =>
    req<Base & { opponents: OpponentRow[]; difference_test: ChiSquare; by_opponent_style: Record<string, string[]>; findings: string[]; note: string; min_actions_per_opponent: number }>(`/api/patterns/${k(id)}/opponent-analysis${q(f)}`),
  analyze: (id: string, force = false) => req<PatternJob>(`/api/patterns/${k(id)}/analyze${force ? "?force=true" : ""}`, { method: "POST" }),
  status: (id: string) => req<AnalysisStatus>(`/api/patterns/${k(id)}/analysis-status`),
  compare: (a: string, b: string, f?: PatternFilters) => req<CompareResult>(`/api/patterns/compare${q(f, { fighter_a: a, fighter_b: b })}`),
};

export const RELIABILITY: Record<Reliability, { label: string; cls: string; hint: string }> = {
  high: { label: "High", cls: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30", hint: "≥30 observations from ≥3 fights, 95% CI narrower than 0.30" },
  medium: { label: "Medium", cls: "bg-sky-500/15 text-sky-300 ring-sky-500/30", hint: "≥10 observations from ≥2 fights, 95% CI narrower than 0.50" },
  low: { label: "Low", cls: "bg-amber-500/15 text-amber-300 ring-amber-500/30", hint: "Above the minimum sample size but wide confidence interval" },
  single_fight: { label: "Single fight", cls: "bg-violet-500/15 text-violet-300 ring-violet-500/30", hint: "All observations come from one fight — not generalisable" },
  insufficient: { label: "Insufficient", cls: "bg-ink-700 text-ink-400 ring-ink-600", hint: "Fewer observations than the configured minimum" },
};
export const DEFINITIONS: Record<Definition, string> = {
  next: "Next action in sequence",
  next_offensive: "Next offensive action",
  window: "Next action within time window",
};
