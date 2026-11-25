import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  BarChart2,
  Download,
  FileSpreadsheet,
  FileUp,
  HelpCircle,
  RefreshCw,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import { ModelInfoResponse } from '../types';
import { EmptyState } from '../components/EmptyState';
import { ClassBadge } from '../components/StatusBadges';

export const ModelPerformancePage: React.FC = () => {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const jsonInputRef = useRef<HTMLInputElement | null>(null);
  const csvInputRef = useRef<HTMLInputElement | null>(null);

  const loadInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getModelInfo();
      setInfo(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInfo();
  }, [loadInfo]);

  const handleImportJson = async (file: File | undefined) => {
    if (!file) return;
    setImporting(true);
    setError(null);
    setMessage(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('experiment_name', file.name.replace(/\.json$/i, ''));
      await api.importEvaluationJson(fd);
      setMessage('Imported evaluation summary into SQLite model_experiments successfully.');
      await loadInfo();
    } catch (err: any) {
      setError(err.message || 'Failed to import evaluation JSON.');
    } finally {
      setImporting(false);
    }
  };

  const handleImportCsv = async (file: File | undefined) => {
    if (!file) return;
    setImporting(true);
    setError(null);
    setMessage(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('experiment_name', file.name.replace(/\.csv$/i, ''));
      const res = await api.importTrainingCsv(fd);
      setMessage(`Imported ${res.epochs}-epoch results.csv and regenerated results.png.`);
      await loadInfo();
    } catch (err: any) {
      setError(err.message || 'Failed to import YOLO results.csv.');
    } finally {
      setImporting(false);
    }
  };

  if (loading && !info) {
    return (
      <div className="rg-card p-12 flex items-center justify-center text-sm text-slate-400">
        <RefreshCw className="w-4 h-4 animate-spin mr-2 text-blue-400" />
        Loading model evaluation records...
      </div>
    );
  }

  const activeEval = info?.latest_evaluation || info?.experiments?.[0] || null;
  const perClass = activeEval?.per_class || [];
  const trainingHistory = info?.latest_training?.training_history || [];
  const confusionMatrix = info?.latest_evaluation?.confusion_matrix || null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">
            Model Performance, Evaluation &amp; Error Analysis
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Held-out RDD2022 test-set metrics loaded strictly from real experiment artifacts — never fabricated.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={csvInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => handleImportCsv(e.target.files?.[0])}
          />
          <input
            ref={jsonInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={(e) => handleImportJson(e.target.files?.[0])}
          />
          <button
            onClick={() => csvInputRef.current?.click()}
            disabled={importing}
            className="rg-btn-secondary text-xs"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
            Import results.csv
          </button>
          <button
            onClick={() => jsonInputRef.current?.click()}
            disabled={importing}
            className="rg-btn-secondary text-xs"
          >
            <FileUp className="w-3.5 h-3.5 text-blue-400" />
            {importing ? 'Importing...' : 'Import Evaluation JSON'}
          </button>
          <button onClick={loadInfo} className="rg-btn-secondary text-xs">
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </div>
      </div>

      {message && (
        <div className="rg-card p-3.5 border-emerald-500/40 bg-emerald-950/20 text-xs text-emerald-200">
          {message}
        </div>
      )}
      {error && (
        <div className="rg-card p-3.5 border-red-500/40 bg-red-950/20 text-xs text-red-200">
          {error}
        </div>
      )}

      {!activeEval ? (
        <EmptyState
          icon={BarChart2}
          title="Pending Trained Model Evaluation Artifacts"
          description="No evaluation metrics have been recorded yet. Per strict scientific integrity rules, RoadGuard AI never displays placeholder precision, recall, or mAP values. Execute notebooks/03_model_training.ipynb and notebooks/04_model_evaluation.ipynb on Google Colab GPU, then import evaluation_summary.json above."
        />
      ) : (
        <>
          {/* Metric KPI Row */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
            {[
              { label: 'Precision', val: `${(activeEval.precision * 100).toFixed(2)}%` },
              { label: 'Recall', val: `${(activeEval.recall * 100).toFixed(2)}%` },
              { label: 'F1-Score', val: `${((activeEval.f1_score || 0) * 100).toFixed(2)}%` },
              { label: 'mAP@50', val: `${(activeEval.map50 * 100).toFixed(2)}%` },
              { label: 'mAP@50-95', val: `${(activeEval.map50_95 * 100).toFixed(2)}%` },
              {
                label: 'Inference Speed',
                val: activeEval.fps ? `${activeEval.fps} FPS` : `${activeEval.inference_ms || 0} ms`,
              },
            ].map((m) => (
              <div key={m.label} className="rg-card p-4">
                <div className="text-xs text-slate-400">{m.label}</div>
                <div className="text-xl font-bold font-mono text-slate-100 mt-1.5">{m.val}</div>
              </div>
            ))}
          </div>

          {/* Per-Class Performance Chart */}
          {perClass.length > 0 && (
            <div className="rg-card p-5">
              <h3 className="text-sm font-semibold text-slate-100 mb-1">
                Per-Category Detection Performance (Held-Out Test Split)
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Precision, Recall, F1-Score, and AP@50 across D00, D10, D20, and D40
              </p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={perClass}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="class_code" stroke="#94A3B8" fontSize={12} />
                    <YAxis stroke="#94A3B8" fontSize={12} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0B101E',
                        borderColor: '#1E293B',
                        color: '#F1F5F9',
                        fontSize: '12px',
                      }}
                    />
                    <Legend />
                    <Bar dataKey="precision" name="Precision" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="recall" name="Recall" fill="#06B6D4" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="f1_score" name="F1-Score" fill="#10B981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="map50" name="mAP@50" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Training Loss Curves if available */}
          {trainingHistory.length > 0 && (
            <div className="rg-card p-5">
              <h3 className="text-sm font-semibold text-slate-100 mb-1">
                Training &amp; Validation Loss Progression
              </h3>
              <p className="text-xs text-slate-400 mb-4">Epoch-wise optimization trajectory</p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="epoch" stroke="#94A3B8" fontSize={12} />
                    <YAxis stroke="#94A3B8" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0B101E',
                        borderColor: '#1E293B',
                        color: '#F1F5F9',
                        fontSize: '12px',
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="train/box_loss"
                      name="Train Box Loss"
                      stroke="#3B82F6"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="val/box_loss"
                      name="Val Box Loss"
                      stroke="#F59E0B"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="metrics/mAP50(B)"
                      name="Val mAP@50"
                      stroke="#10B981"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Ultralytics Training Artifacts (results.png & results.csv) */}
          {(info?.training_artifacts?.results_png_url || info?.training_artifacts?.results_csv_url) && (
            <div className="rg-card p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">
                    YOLO Training Artifacts (`results.png` &amp; `results.csv`)
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {info?.latest_training?.epochs_completed
                      ? `${info.latest_training.epochs_completed}-epoch training metrics extracted from the active checkpoint`
                      : 'Standard Ultralytics training summary plots and raw epoch CSV logs'}
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {info?.training_artifacts?.results_csv_url && (
                    <a
                      href={info.training_artifacts.results_csv_url}
                      download="results.csv"
                      className="rg-btn-secondary text-xs py-1.5"
                    >
                      <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
                      Download results.csv
                    </a>
                  )}
                  {info?.training_artifacts?.results_png_url && (
                    <a
                      href={info.training_artifacts.results_png_url}
                      download="results.png"
                      className="rg-btn-primary text-xs py-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download results.png
                    </a>
                  )}
                </div>
              </div>
              {info?.training_artifacts?.results_png_url && (
                <img
                  src={info.training_artifacts.results_png_url}
                  alt="YOLO Training Results Grid (results.png)"
                  className="w-full rounded-xl border border-slate-800 bg-white p-2"
                />
              )}
            </div>
          )}

          {/* Confusion Matrix Table */}
          {confusionMatrix?.matrix?.length > 0 && (
            <div className="rg-card overflow-hidden">
              <div className="px-5 py-3.5 border-b border-slate-800">
                <h3 className="text-sm font-semibold text-slate-100">
                  Confusion Matrix (Predicted Rows × Ground-Truth Columns)
                </h3>
              </div>
              <div className="overflow-x-auto p-4">
                <table className="w-full text-center text-xs font-mono border border-slate-800">
                  <thead className="bg-navy-950 text-slate-300">
                    <tr>
                      <th className="p-2.5 border border-slate-800">Pred \ True</th>
                      {confusionMatrix.labels.map((lbl: string) => (
                        <th key={lbl} className="p-2.5 border border-slate-800">
                          {lbl}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionMatrix.matrix.map((row: number[], rIdx: number) => (
                      <tr key={rIdx}>
                        <td className="p-2.5 border border-slate-800 bg-navy-950 font-semibold text-slate-300">
                          {confusionMatrix.labels[rIdx] || rIdx}
                        </td>
                        {row.map((val: number, cIdx: number) => (
                          <td
                            key={cIdx}
                            className={`p-2.5 border border-slate-800 ${
                              rIdx === cIdx && val > 0
                                ? 'bg-blue-600/25 text-blue-200 font-bold'
                                : 'text-slate-400'
                            }`}
                          >
                            {Math.round(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Recorded Experiments Registry */}
      {info && info.experiments.length > 0 && (
        <div className="rg-card overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-800">
            <h3 className="text-sm font-semibold text-slate-100">Experiment Tracking Registry</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-950/70 text-slate-400 uppercase">
                <tr>
                  <th className="px-4 py-3">Experiment</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Epochs</th>
                  <th className="px-4 py-3">ImgSz</th>
                  <th className="px-4 py-3">Precision</th>
                  <th className="px-4 py-3">Recall</th>
                  <th className="px-4 py-3">mAP@50</th>
                  <th className="px-4 py-3">mAP@50-95</th>
                  <th className="px-4 py-3">Recorded</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono">
                {info.experiments.map((e) => (
                  <tr key={e.id}>
                    <td className="px-4 py-3 font-sans font-medium text-slate-200">
                      {e.experiment_name}
                    </td>
                    <td className="px-4 py-3 text-blue-300">{e.model_name}</td>
                    <td className="px-4 py-3 text-slate-300">{e.epochs}</td>
                    <td className="px-4 py-3 text-slate-300">{e.image_size}</td>
                    <td className="px-4 py-3 text-slate-200">{(e.precision * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 text-slate-200">{(e.recall * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 text-emerald-400 font-semibold">
                      {(e.map50 * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-cyan-300">{(e.map50_95 * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 font-sans text-slate-400">
                      {e.training_date ? new Date(e.training_date).toLocaleDateString() : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Module 5 — Explainability & Failure Mode Taxonomy */}
      <div className="rg-card p-5 space-y-4">
        <div className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            Computer Vision Failure Mode &amp; False-Positive / False-Negative Analysis
          </h3>
        </div>
        <p className="text-xs text-slate-400">
          Domain taxonomy of environmental confounders affecting 2D road-surface object detectors on RDD2022:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {(info?.error_taxonomy || []).map((errItem) => (
            <div
              key={errItem.id}
              className="p-4 rounded-xl bg-navy-950 border border-slate-800/80 space-y-1.5"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-100">{errItem.name}</span>
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              </div>
              <div className="text-[11px] font-mono text-red-300">{errItem.error_mode}</div>
              <p className="text-xs text-slate-400 leading-relaxed">{errItem.mechanism}</p>
              <p className="text-[11px] text-cyan-300/90 pt-1">
                <strong>Mitigation:</strong> {errItem.mitigation}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
