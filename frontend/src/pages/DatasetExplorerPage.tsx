import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  ExternalLink,
  RefreshCw,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import { DatasetExplorerResponse } from '../types';
import { ClassBadge } from '../components/StatusBadges';

export const DatasetExplorerPage: React.FC = () => {
  const [data, setData] = useState<DatasetExplorerResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getDatasetSummary();
      setData(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDataset();
  }, []);

  if (loading && !data) {
    return (
      <div className="rg-card p-12 flex items-center justify-center text-sm text-slate-400">
        <RefreshCw className="w-4 h-4 animate-spin mr-2 text-blue-400" />
        Inspecting RDD2022 dataset manifests...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rg-card p-6 border-red-500/40 bg-red-950/20 text-xs text-red-200">
        {error || 'Dataset explorer unavailable.'}
      </div>
    );
  }

  const sm = data.summary_metrics || {};

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">RDD2022 Dataset Explorer &amp; Quality Audit</h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Class distribution, leakage-safe 70/15/15 split verification, ground-truth samples, and annotation validation.
          </p>
        </div>
        <button onClick={loadDataset} className="rg-btn-secondary text-xs">
          <RefreshCw className="w-3.5 h-3.5" />
          Reload Manifests
        </button>
      </div>

      {/* Kaggle Source & Important Class Discrepancy Verification Card */}
      <div className="rg-card p-5 border-blue-500/30 bg-blue-950/10 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Database className="w-4 h-4 text-blue-400" />
            Dataset Provenance &amp; Class-Mapping Verification
          </div>
          <div className="flex items-center gap-3 text-xs">
            <a
              href={data.kaggle_info.kaggle_url}
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:underline inline-flex items-center gap-1"
            >
              Kaggle Mirror <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href={data.kaggle_info.official_repo_url}
              target="_blank"
              rel="noreferrer"
              className="text-cyan-400 hover:underline inline-flex items-center gap-1"
            >
              sekilab/RoadDamageDetector <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>

        <div className="p-3.5 rounded-lg bg-navy-950/90 border border-amber-500/30 flex items-start gap-3 text-xs">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1 text-slate-300 leading-relaxed">
            <div className="font-semibold text-amber-200">
              Verified Source Metadata Finding (`aliabdelmenam/rdd-2022`)
            </div>
            <p>{data.kaggle_info.important_schema_finding}</p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>Local Kaggle CLI Auth Status:</span>
          {data.kaggle_info.kaggle_credentials_configured ? (
            <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" /> Configured ({data.kaggle_info.kaggle_auth_method})
            </span>
          ) : (
            <span className="text-amber-400 font-mono">
              Not configured in ~/.kaggle/kaggle.json (Use Colab notebook or configure ~/.kaggle/kaggle.json)
            </span>
          )}
        </div>
      </div>

      {/* Target 4-Class Specification */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {data.target_classes.map((tc) => (
          <div key={tc.code} className="rg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <ClassBadge code={tc.code} name={tc.name} />
              <span className="text-xs font-mono text-slate-500">ID: {tc.class_id}</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{tc.description}</p>
          </div>
        ))}
      </div>

      {!data.prepared ? (
        <div className="rg-card p-8 space-y-3 text-center">
          <h3 className="text-base font-semibold text-slate-100">
            Local Prepared Dataset Manifests Not Yet Generated
          </h3>
          <p className="text-xs text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Because RDD2022 is ~10.6 GB compressed (~22 GB extracted), dataset preparation and GPU
            training are designed to run in Google Colab via{' '}
            <code className="font-mono text-blue-300">notebooks/01_dataset_preparation.ipynb</code>, or
            locally via:
          </p>
          <pre className="text-xs font-mono bg-navy-950 border border-slate-800 rounded-lg p-3 max-w-2xl mx-auto text-left text-cyan-300 overflow-x-auto">
            {`.venv\\Scripts\\python.exe -m ml.dataset.cli download --slug aliabdelmenam/rdd-2022\n.venv\\Scripts\\python.exe -m ml.dataset.cli inspect --source-dir ./outputs/raw_dataset/extracted\n.venv\\Scripts\\python.exe -m ml.dataset.cli prepare --source-dir ./outputs/raw_dataset/extracted --confirm-exclude-non-target`}
          </pre>
        </div>
      ) : (
        <>
          {/* Summary KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
            {[
              { label: 'Exported Images', val: sm.total_exported_images },
              { label: 'Train Split (70%)', val: `${sm.train_images} (${sm.train_pct}%)` },
              { label: 'Val Split (15%)', val: `${sm.val_images} (${sm.val_pct}%)` },
              { label: 'Test Split (15%)', val: `${sm.test_images} (${sm.test_pct}%)` },
              { label: 'Total Bounding Boxes', val: sm.total_bounding_boxes },
              { label: 'Imbalance Ratio', val: `${sm.imbalance_ratio_max_to_min}×` },
            ].map((item) => (
              <div key={item.label} className="rg-card p-4">
                <div className="text-xs text-slate-400">{item.label}</div>
                <div className="text-lg font-bold font-mono text-slate-100 mt-1">{item.val}</div>
              </div>
            ))}
          </div>

          {/* Class Distribution Split Chart */}
          {data.class_distribution.length > 0 && (
            <div className="rg-card p-5">
              <h3 className="text-sm font-semibold text-slate-100 mb-1">
                Class Distribution Across Train / Validation / Test Splits
              </h3>
              <p className="text-xs text-slate-400 mb-4">{sm.imbalance_notes}</p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.class_distribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="class_code" stroke="#94A3B8" fontSize={12} />
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
                    <Bar dataKey="train_instances" name="Train (70%)" fill="#3B82F6" stackId="a" />
                    <Bar dataKey="val_instances" name="Validation (15%)" fill="#06B6D4" stackId="a" />
                    <Bar dataKey="test_instances" name="Test (15%)" fill="#10B981" stackId="a" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Ground-Truth Sample Gallery */}
          {data.sample_images.length > 0 && (
            <div className="rg-card p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-100">
                Verified Ground-Truth Bounding-Box Samples
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                {data.sample_images.map((s) => (
                  <div
                    key={s.filename}
                    className="rounded-lg overflow-hidden border border-slate-800 bg-navy-950"
                  >
                    <img src={s.url} alt={s.filename} className="w-full h-44 object-cover" />
                    <div className="px-2.5 py-1.5 text-[11px] font-mono text-slate-400 truncate">
                      {s.filename}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Annotation Issues Table */}
          <div className="rg-card overflow-hidden">
            <div className="px-5 py-3.5 border-b border-slate-800">
              <h3 className="text-sm font-semibold text-slate-100">
                Annotation Quality Audit Log ({data.annotation_issues.length} logged items)
              </h3>
            </div>
            {data.annotation_issues.length === 0 ? (
              <div className="p-6 text-xs text-emerald-400">
                Zero corrupt images, missing labels, or invalid bounding boxes found.
              </div>
            ) : (
              <div className="overflow-x-auto max-h-72">
                <table className="w-full text-left text-xs">
                  <thead className="bg-navy-950 text-slate-400 uppercase">
                    <tr>
                      <th className="px-4 py-2.5">Issue Type</th>
                      <th className="px-4 py-2.5">Severity</th>
                      <th className="px-4 py-2.5">Image</th>
                      <th className="px-4 py-2.5">Details</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {data.annotation_issues.map((iss, idx) => (
                      <tr key={idx}>
                        <td className="px-4 py-2 font-mono text-amber-300">{iss.issue_type}</td>
                        <td className="px-4 py-2 uppercase font-mono text-slate-400">
                          {iss.severity}
                        </td>
                        <td className="px-4 py-2 font-mono text-slate-300 truncate max-w-xs">
                          {iss.image_path}
                        </td>
                        <td className="px-4 py-2 text-slate-400">{iss.details}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
