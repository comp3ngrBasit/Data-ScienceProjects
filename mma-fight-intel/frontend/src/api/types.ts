export type VideoStatus = "uploaded" | "queued" | "processing" | "completed" | "failed";
export type JobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";
export type Stance = "Orthodox" | "Southpaw" | "Unknown";

export interface FighterRef {
  id: string;
  name: string;
  nickname: string | null;
  stance: Stance;
}

export interface VideoSummary {
  id: string;
  title: string;
  original_filename: string;
  file_ext: string;
  file_size: number;
  status: VideoStatus;
  error_message: string | null;
  uploaded_at: string;
  processed_at: string | null;
  duration_seconds: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  video_codec: string | null;
  has_audio: boolean | null;
  sampled_frame_count: number;
  thumbnail_url: string | null;
  target_fighter: FighterRef | null;
  fighters: FighterRef[];
}

export interface StageLogEntry {
  stage: string;
  label: string;
  status: "running" | "completed" | "failed" | "cancelled";
  seconds?: number;
  error?: string;
}

export interface Job {
  id: string;
  video_id: string;
  video_title: string | null;
  job_type: string;
  status: JobStatus;
  attempt: number;
  progress: number;
  current_stage: string | null;
  error_message: string | null;
  stage_log: StageLogEntry[] | null;
  warnings: string[] | null;
  settings_snapshot: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
}

export interface QualityInfo {
  method: string;
  score: number;
  rating: "Good" | "Fair" | "Poor";
  issues: string[];
  resolution_class: string;
  frames_analyzed: number;
  mean_brightness: number | null;
  mean_contrast: number | null;
  mean_sharpness: number | null;
  median_sharpness: number | null;
  blurry_frame_pct: number | null;
  dark_frame_pct: number | null;
  overexposed_frame_pct: number | null;
  decode_failures: number;
  warnings?: string[];
}

export interface SamplingConfig {
  mode: "interval" | "uniform";
  interval_seconds: number | null;
  uniform_frame_count: number | null;
  max_frames: number;
  images_saved: boolean;
  frame_max_width: number;
  jpeg_quality: number;
}

export interface VideoDetail extends VideoSummary {
  content_type: string | null;
  sha256: string;
  notes: string | null;
  updated_at: string;
  frame_count: number | null;
  frame_count_source: string | null;
  rotation: number | null;
  pixel_format: string | null;
  container_format: string | null;
  bitrate_kbps: number | null;
  audio_codec: string | null;
  audio_sample_rate: number | null;
  audio_channels: number | null;
  metadata_source: string | null;
  raw_metadata: Record<string, unknown> | null;
  quality: QualityInfo | null;
  sampling_config: SamplingConfig | null;
  target_fighter_notes: string | null;
  stream_url: string;
  has_preview_proxy: boolean;
  jobs: Job[];
}

export interface Paged<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface JobList extends Paged<Job> {
  counts: Record<JobStatus, number>;
}

export interface Frame {
  id: number;
  frame_index: number;
  timestamp_seconds: number;
  width: number | null;
  height: number | null;
  brightness: number | null;
  contrast: number | null;
  sharpness: number | null;
  image_url: string | null;
}

export interface UploadItemResult {
  filename: string;
  status: "created" | "duplicate" | "rejected";
  http_status: number;
  video: VideoSummary | null;
  existing_video_id: string | null;
  error: string | null;
}

export interface UploadResponse {
  results: UploadItemResult[];
  created: number;
  duplicates: number;
  rejected: number;
}

export interface Fighter {
  id: string;
  name: string;
  nickname: string | null;
  stance: Stance;
  height_cm: number | null;
  reach_cm: number | null;
  weight_class: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
  video_count: number;
  target_video_count: number;
}

export interface FighterVideoRef {
  id: string;
  title: string;
  status: VideoStatus;
  duration_seconds: number | null;
  uploaded_at: string;
  has_thumbnail: boolean;
  is_target: boolean;
}

export interface FighterDetail extends Fighter {
  videos: FighterVideoRef[];
}

export type FighterInput = Pick<
  Fighter,
  "name" | "nickname" | "stance" | "height_cm" | "reach_cm" | "weight_class" | "notes"
>;

export interface ProcessingSettings {
  sampling_mode: "interval" | "uniform";
  sample_interval_seconds: number;
  uniform_frame_count: number;
  max_sampled_frames: number;
  save_frame_images: boolean;
  frame_max_width: number;
  frame_jpeg_quality: number;
  thumbnail_width: number;
  generate_preview_proxy: boolean;
  auto_process_on_upload: boolean;
  max_upload_size_mb: number;
}

export interface SystemInfo {
  app_name: string;
  version: string;
  ffmpeg_path: string | null;
  ffprobe_path: string | null;
  ffmpeg_source: string | null;
  opencv_version: string;
  data_dir: string;
  database_url: string;
  allowed_extensions: string[];
  worker_threads: number;
  storage_bytes: Record<string, number>;
}

export interface Overview {
  total_videos: number;
  total_fighters: number;
  videos_processed: number;
  videos_awaiting: number;
  videos_failed: number;
  status_counts: Record<VideoStatus, number>;
  total_duration_seconds: number;
  total_bytes: number;
  total_sampled_frames: number;
  storage_bytes: Record<string, number>;
  uploads_by_day: { date: string; count: number }[];
  resolution_breakdown: { label: string; count: number }[];
  recent_videos: VideoSummary[];
  recent_jobs: Job[];
  active_jobs: number;
}

export type FrameList = Paged<Frame>;
