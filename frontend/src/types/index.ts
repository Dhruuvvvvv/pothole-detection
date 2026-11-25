export interface HealthStatus {
  status: string;
  app_name: string;
  environment: string;
  database_ok: boolean;
  model_configured: boolean;
  model_status: string;
  cuda_available: boolean;
  gpu_name?: string | null;
  timestamp_utc: string;
}

export interface DetectionItem {
  id?: number;
  class_id: number;
  class_code: 'D00' | 'D10' | 'D20' | 'D40' | string;
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  box_width: number;
  box_height: number;
  box_area_px: number;
  image_area_pct: number;
  priority_score: number;
  priority_band: 'High' | 'Medium' | 'Low' | string;
  track_id?: number | null;
  frame_index?: number | null;
  score_breakdown?: {
    area_component: number;
    category_component: number;
    confidence_component: number;
  };
}

export interface AnalysisSummary {
  id: number;
  job_uid: string;
  filename: string;
  analysis_type: 'image' | 'video';
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress_pct: number;
  timestamp: string;
  processing_duration: number;
  total_detections: number;
  high_priority_count: number;
  medium_priority_count: number;
  low_priority_count: number;
  input_url?: string | null;
  output_url?: string | null;
  summary: Record<string, any>;
  detections: DetectionItem[];
}

export interface ModelRuntimeStatus {
  configured: boolean;
  status: string;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  checkpoint_size_mb?: number | null;
  device: string;
  cuda_available: boolean;
  gpu_name?: string | null;
  model_classes: Record<string, string>;
  target_classes: Record<string, string>;
  message: string;
}

export interface DashboardSummary {
  total_analyses: number;
  total_images_analysed: number;
  total_videos_analysed: number;
  total_damage_instances: number;
  d00_count: number;
  d10_count: number;
  d20_count: number;
  d40_count: number;
  high_priority_detections: number;
  medium_priority_detections: number;
  low_priority_detections: number;
  recent_analyses: AnalysisSummary[];
  model_info: ModelRuntimeStatus;
}

export interface AnalyticsDistribution {
  categories: { code: string; name: string; count: number; color: string }[];
  priorities: { band: string; count: number; color: string }[];
}

export interface HistoryPoint {
  date: string;
  analyses: number;
  detections: number;
  high_priority: number;
}

export interface OperationParamSpec {
  name: string;
  label: string;
  type: 'number' | 'select' | 'boolean';
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  default: any;
}

export interface ClassicalOperationSpec {
  id: string;
  name: string;
  category: string;
  description: string;
  formula: string;
  parameters: OperationParamSpec[];
}

export interface ImageMetrics {
  width: number;
  height: number;
  channels: number;
  mean_intensity: number;
  std_deviation: number;
  rms_contrast: number;
  michelson_contrast: number;
  shannon_entropy: number;
  laplacian_variance: number;
  edge_pixel_density: number;
  psnr_db?: number | null;
  ssim_score?: number | null;
}

export interface HistogramBin {
  intensity: number;
  luminance: number;
  red: number;
  green: number;
  blue: number;
}

export interface ClassicalProcessResponse {
  filename: string;
  original_url: string;
  processed_url: string;
  difference_url: string;
  operation: string;
  parameters: Record<string, any>;
  processing_time_ms: number;
  original_metrics: ImageMetrics;
  processed_metrics: ImageMetrics;
  original_histogram: HistogramBin[];
  processed_histogram: HistogramBin[];
}

export interface FilterComparisonResponse {
  filename: string;
  kernel_size: number;
  original_url: string;
  comparisons: {
    operation: string;
    label: string;
    processed_url: string;
    difference_url: string;
    processing_time_ms: number;
    metrics: ImageMetrics;
  }[];
}

export interface ModelExperiment {
  id: number;
  experiment_name: string;
  model_name: string;
  dataset_version: string;
  epochs: number;
  batch_size: number;
  image_size: number;
  precision: number;
  recall: number;
  f1_score: number;
  map50: number;
  map50_95: number;
  inference_ms: number;
  fps: number;
  per_class: {
    class_id: number;
    class_code: string;
    class_name: string;
    precision: number;
    recall: number;
    f1_score: number;
    map50: number;
    map50_95: number;
  }[];
  artifacts: Record<string, string | null>;
  training_date: string;
  checkpoint_path: string;
}

export interface ErrorTaxonomyEntry {
  id: string;
  name: string;
  error_mode: string;
  mechanism: string;
  mitigation: string;
}

export interface ModelInfoResponse {
  runtime: ModelRuntimeStatus;
  experiments: ModelExperiment[];
  latest_evaluation?: any | null;
  latest_training?: any | null;
  training_artifacts?: {
    results_csv_url?: string | null;
    results_png_url?: string | null;
    loss_curves_url?: string | null;
    per_class_chart_url?: string | null;
  };
  error_taxonomy: ErrorTaxonomyEntry[];
}

export interface DatasetExplorerResponse {
  prepared: boolean;
  dataset_dir: string;
  target_classes: {
    class_id: number;
    code: string;
    name: string;
    description: string;
    color: string;
  }[];
  kaggle_info: {
    primary_dataset_slug: string;
    kaggle_url: string;
    official_repo_url: string;
    kaggle_credentials_configured: boolean;
    kaggle_auth_method: string;
    important_schema_finding: string;
  };
  summary_metrics: Record<string, any>;
  class_distribution: {
    class_id: number;
    class_code: string;
    class_name: string;
    total_instances: number;
    image_count: number;
    instance_share_pct: number;
    mean_box_area_norm: number;
    train_instances: number;
    val_instances: number;
    test_instances: number;
  }[];
  annotation_issues: Record<string, string>[];
  sample_images: { filename: string; url: string }[];
  config_yaml?: any;
}

export interface RuntimeSettings {
  low_max_threshold: number;
  high_min_threshold: number;
  area_weight: number;
  category_weight: number;
  confidence_weight: number;
  reference_saturation_area_pct: number;
  class_weights: Record<string, number>;
  default_conf_threshold: number;
  default_iou_threshold: number;
  checkpoint_path?: string | null;
}
