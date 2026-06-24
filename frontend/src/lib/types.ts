// Shared types mirroring the FastAPI backend (api.py) responses.

export interface SetupFile {
  item: string;
  required: boolean;
  exists: boolean;
  path: string;
}

export interface HealthResponse {
  ready: boolean;
  loading: boolean;
  load_error: string | null;
  required_ok: boolean;
  files: SetupFile[];
}

export interface MethodInfo {
  name: string;
  available: boolean;
  description: string;
  setup: string;
}

export interface MethodsResponse {
  methods: MethodInfo[];
}

export interface MetricRow {
  method: string;
  accuracy: number;
  macro_f1: number;
  top3_accuracy: number;
  avg_inference_time_sec: number;
}

export interface MetricsResponse {
  available: boolean;
  rows: MetricRow[];
}

export interface ConfusionMethod {
  method: string;
  matrix: number[][]; // counts, rows = true category, cols = predicted
  normalized: number[][]; // row-normalized (0..1)
  support: number[]; // total true samples per category (row sums)
}

export interface ConfusionResponse {
  available: boolean;
  categories: string[];
  methods: ConfusionMethod[];
}

export interface RankItem {
  method: string;
  rank: number;
  label: string;
  score: number; // 0..1
  raw_score: number;
  distance: number | null;
  category: string;
}

// results maps method name -> ranked list (top-N)
export type MethodResults = Record<string, RankItem[]>;

export interface ClassifyResponse {
  filename: string;
  size_mb: number;
  results: MethodResults;
}
