import {
  AnalysisSummary,
  AnalyticsDistribution,
  ClassicalOperationSpec,
  ClassicalProcessResponse,
  DashboardSummary,
  DatasetExplorerResponse,
  FilterComparisonResponse,
  HealthStatus,
  HistoryPoint,
  ModelInfoResponse,
  ModelRuntimeStatus,
  RuntimeSettings,
} from '../types';

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    throw new ApiError(
      'Unable to connect to RoadGuard AI backend API (http://127.0.0.1:8000). Verify the FastAPI server is running.',
      0,
      'NETWORK_UNAVAILABLE'
    );
  }

  if (!response.ok) {
    let message = `HTTP ${response.status}: ${response.statusText}`;
    let code: string | undefined;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        if (typeof payload.detail === 'string') {
          message = payload.detail;
        } else if (typeof payload.detail === 'object') {
          message = payload.detail.message || JSON.stringify(payload.detail);
          code = payload.detail.code;
        }
      }
    } catch {
      // Fallback to statusText
    }
    throw new ApiError(message, response.status, code);
  }

  return response.json() as Promise<T>;
}

export const api = {
  getHealth: () => request<HealthStatus>('/api/health'),

  getDashboardSummary: () => request<DashboardSummary>('/api/dashboard/summary'),

  getAnalyticsDistribution: () => request<AnalyticsDistribution>('/api/analytics/distribution'),

  getAnalyticsHistory: (limit = 30) =>
    request<{ timeline: HistoryPoint[] }>(`/api/analytics/history?limit=${limit}`),

  analyzeImage: (formData: FormData) =>
    request<AnalysisSummary>('/api/analyze/image', {
      method: 'POST',
      body: formData,
    }),

  analyzeVideo: (formData: FormData) =>
    request<AnalysisSummary>('/api/analyze/video', {
      method: 'POST',
      body: formData,
    }),

  getVideoStatus: (analysisId: number) =>
    request<AnalysisSummary>(`/api/analyze/video/${analysisId}/status`),

  getClassicalCatalog: () =>
    request<{ operations: ClassicalOperationSpec[] }>('/api/process/catalog'),

  processClassicalImage: (formData: FormData) =>
    request<ClassicalProcessResponse>('/api/process/image', {
      method: 'POST',
      body: formData,
    }),

  compareClassicalFilters: (formData: FormData) =>
    request<FilterComparisonResponse>('/api/process/compare-filters', {
      method: 'POST',
      body: formData,
    }),

  listReports: (params: {
    search?: string;
    class_code?: string;
    priority_band?: string;
    analysis_type?: string;
    limit?: number;
    offset?: number;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== '') qs.set(k, String(v));
    });
    return request<{ total: number; items: AnalysisSummary[] }>(`/api/reports?${qs.toString()}`);
  },

  getReportDetail: (id: number) => request<AnalysisSummary>(`/api/reports/${id}`),

  deleteReport: (id: number) =>
    request<{ deleted: boolean; id: number }>(`/api/reports/${id}`, {
      method: 'DELETE',
    }),

  getModelInfo: () => request<ModelInfoResponse>('/api/model/info'),

  reloadModel: (checkpointPath?: string) =>
    request<ModelRuntimeStatus>('/api/model/reload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint_path: checkpointPath || null }),
    }),

  importEvaluationJson: (formData: FormData) =>
    request<{ imported: boolean; experiment_id: number }>('/api/model/import-evaluation', {
      method: 'POST',
      body: formData,
    }),

  importTrainingCsv: (formData: FormData) =>
    request<{
      imported: boolean;
      epochs: number;
      results_csv_url: string;
      results_png_url?: string | null;
    }>('/api/model/import-training-csv', {
      method: 'POST',
      body: formData,
    }),

  getDatasetSummary: () => request<DatasetExplorerResponse>('/api/dataset/summary'),

  getSettings: () => request<RuntimeSettings>('/api/settings'),

  updateSettings: (payload: RuntimeSettings) =>
    request<RuntimeSettings>('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
};
