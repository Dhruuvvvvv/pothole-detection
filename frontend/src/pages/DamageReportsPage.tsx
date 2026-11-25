import React, { useCallback, useEffect, useState } from 'react';
import {
  Eye,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  RefreshCw,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisSummary } from '../types';
import { ClassBadge, PriorityBadge } from '../components/StatusBadges';
import { EmptyState } from '../components/EmptyState';

export const DamageReportsPage: React.FC = () => {
  const [items, setItems] = useState<AnalysisSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [classCode, setClassCode] = useState('');
  const [priorityBand, setPriorityBand] = useState('');
  const [analysisType, setAnalysisType] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedDetail, setSelectedDetail] = useState<AnalysisSummary | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listReports({
        search,
        class_code: classCode,
        priority_band: priorityBand,
        analysis_type: analysisType,
        limit: 100,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setError(err.message || 'Failed to load inspection reports.');
    } finally {
      setLoading(false);
    }
  }, [search, classCode, priorityBand, analysisType]);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  const handleOpenDetail = async (id: number) => {
    setLoadingDetail(true);
    try {
      const detail = await api.getReportDetail(id);
      setSelectedDetail(detail);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm(`Permanently delete inspection report #${id} and its stored detections?`)) {
      return;
    }
    try {
      await api.deleteReport(id);
      if (selectedDetail?.id === id) setSelectedDetail(null);
      await fetchReports();
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Damage Inspection Reports &amp; Exports</h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Search, filter, inspect, and export persistent image and video analyses to CSV or PDF.
          </p>
        </div>
        <button onClick={fetchReports} className="rg-btn-secondary text-xs">
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh ({total})
        </button>
      </div>

      {/* Filter Bar */}
      <div className="rg-card p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search filename..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="rg-input w-full pl-9"
          />
        </div>

        <select
          value={classCode}
          onChange={(e) => setClassCode(e.target.value)}
          className="rg-input w-full"
        >
          <option value="">All Damage Categories</option>
          <option value="D00">D00 · Longitudinal Crack</option>
          <option value="D10">D10 · Transverse Crack</option>
          <option value="D20">D20 · Alligator Crack</option>
          <option value="D40">D40 · Pothole</option>
        </select>

        <select
          value={priorityBand}
          onChange={(e) => setPriorityBand(e.target.value)}
          className="rg-input w-full"
        >
          <option value="">All Priority Bands</option>
          <option value="High">Contains High Priority</option>
          <option value="Medium">Contains Medium Priority</option>
          <option value="Low">Contains Low Priority</option>
        </select>

        <select
          value={analysisType}
          onChange={(e) => setAnalysisType(e.target.value)}
          className="rg-input w-full"
        >
          <option value="">All Modalities (Image &amp; Video)</option>
          <option value="image">Image Inspections</option>
          <option value="video">Video Inspections</option>
        </select>
      </div>

      {error && (
        <div className="rg-card p-4 border-red-500/40 bg-red-950/20 text-xs text-red-200">
          {error}
        </div>
      )}

      {loading ? (
        <div className="rg-card p-10 text-center text-sm text-slate-400">
          Loading stored inspection reports...
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          icon={FolderOpen}
          title="No Matching Inspection Reports"
          description="No analyses match the selected filters, or no road damage inspections have been run yet."
        />
      ) : (
        <div className="rg-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-950/70 text-slate-400 uppercase">
                <tr>
                  <th className="px-4 py-3">Report ID</th>
                  <th className="px-4 py-3">Filename</th>
                  <th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Detections</th>
                  <th className="px-4 py-3">Risk Breakdown</th>
                  <th className="px-4 py-3">Timestamp</th>
                  <th className="px-4 py-3 text-right">Actions &amp; Export</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {items.map((r) => (
                  <tr key={r.id} className="hover:bg-navy-800/40 transition-colors">
                    <td className="px-4 py-3 font-mono text-slate-300">#{r.id}</td>
                    <td className="px-4 py-3 font-medium text-slate-100">{r.filename}</td>
                    <td className="px-4 py-3 uppercase font-mono text-slate-400">
                      {r.analysis_type}
                    </td>
                    <td className="px-4 py-3 font-mono text-emerald-400">{r.status}</td>
                    <td className="px-4 py-3 font-mono text-slate-200">{r.total_detections}</td>
                    <td className="px-4 py-3 font-mono text-slate-300">
                      <span className="text-red-400">H:{r.high_priority_count}</span> ·{' '}
                      <span className="text-amber-400">M:{r.medium_priority_count}</span> ·{' '}
                      <span className="text-emerald-400">L:{r.low_priority_count}</span>
                    </td>
                    <td className="px-4 py-3 text-slate-400">
                      {new Date(r.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => handleOpenDetail(r.id)}
                          className="rg-btn-secondary py-1 px-2.5 text-xs"
                          title="Inspect Details"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          View
                        </button>
                        <a
                          href={`/api/reports/${r.id}/export/csv`}
                          className="rg-btn-secondary py-1 px-2.5 text-xs"
                          title="Download CSV"
                        >
                          <FileSpreadsheet className="w-3.5 h-3.5" />
                          CSV
                        </a>
                        <a
                          href={`/api/reports/${r.id}/export/pdf`}
                          className="rg-btn-secondary py-1 px-2.5 text-xs"
                          title="Export PDF Report"
                        >
                          <FileText className="w-3.5 h-3.5" />
                          PDF
                        </a>
                        <button
                          onClick={() => handleDelete(r.id)}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-950/30 transition-colors"
                          title="Delete Report"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {(selectedDetail || loadingDetail) && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="rg-card max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-5">
            {loadingDetail || !selectedDetail ? (
              <div className="py-12 text-center text-sm text-slate-400">
                Loading report details...
              </div>
            ) : (
              <>
                <div className="flex items-start justify-between gap-4 border-b border-slate-800 pb-4">
                  <div>
                    <h3 className="text-base font-bold text-slate-100">
                      Report #{selectedDetail.id} — {selectedDetail.filename}
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5 font-mono">
                      Modality: {selectedDetail.analysis_type.toUpperCase()} · Detections:{' '}
                      {selectedDetail.total_detections} · Duration:{' '}
                      {selectedDetail.processing_duration.toFixed(2)}s
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedDetail(null)}
                    className="p-1.5 rounded-lg bg-navy-800 text-slate-300 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {selectedDetail.output_url && selectedDetail.analysis_type === 'image' && (
                  <img
                    src={selectedDetail.output_url}
                    alt="Annotated report"
                    className="w-full max-h-96 object-contain rounded-xl bg-navy-950 border border-slate-800"
                  />
                )}
                {selectedDetail.output_url && selectedDetail.analysis_type === 'video' && (
                  <video
                    src={selectedDetail.output_url}
                    controls
                    className="w-full max-h-96 rounded-xl bg-navy-950 border border-slate-800"
                  />
                )}

                <div className="flex items-center gap-2">
                  <a
                    href={`/api/reports/${selectedDetail.id}/export/csv`}
                    className="rg-btn-secondary text-xs"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    Download CSV
                  </a>
                  <a
                    href={`/api/reports/${selectedDetail.id}/export/pdf`}
                    className="rg-btn-primary text-xs"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Export PDF Inspection Report
                  </a>
                </div>

                <div className="overflow-x-auto border border-slate-800 rounded-xl">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-navy-950 text-slate-400 uppercase">
                      <tr>
                        <th className="px-3 py-2.5">#</th>
                        <th className="px-3 py-2.5">Category</th>
                        <th className="px-3 py-2.5">Conf</th>
                        <th className="px-3 py-2.5">Frame / Track</th>
                        <th className="px-3 py-2.5">Area %</th>
                        <th className="px-3 py-2.5">Score</th>
                        <th className="px-3 py-2.5">Priority</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70 font-mono">
                      {selectedDetail.detections.slice(0, 100).map((d, i) => (
                        <tr key={i}>
                          <td className="px-3 py-2 text-slate-400">{i + 1}</td>
                          <td className="px-3 py-2 font-sans">
                            <ClassBadge code={d.class_code} name={d.class_name} />
                          </td>
                          <td className="px-3 py-2 text-slate-200">
                            {(d.confidence * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-2 text-slate-400">
                            {d.frame_index !== null && d.frame_index !== undefined
                              ? `F:${d.frame_index} / #${d.track_id ?? '-'}`
                              : 'Static Image'}
                          </td>
                          <td className="px-3 py-2 text-slate-300">
                            {d.image_area_pct.toFixed(2)}%
                          </td>
                          <td className="px-3 py-2 text-slate-100">{d.priority_score.toFixed(1)}</td>
                          <td className="px-3 py-2 font-sans">
                            <PriorityBadge band={d.priority_band} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
