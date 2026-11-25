import React from 'react';
import { AlertTriangle, CheckCircle2, RefreshCw, Settings as SettingsIcon } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ModelRuntimeStatus } from '../types';

export const PriorityBadge: React.FC<{ band: string }> = ({ band }) => {
  const styles: Record<string, string> = {
    High: 'bg-red-500/15 text-red-400 border-red-500/30',
    Medium: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    Low: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  };
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border ${
        styles[band] || 'bg-slate-800 text-slate-300 border-slate-700'
      }`}
    >
      {band}
    </span>
  );
};

export const ClassBadge: React.FC<{ code: string; name?: string }> = ({ code, name }) => {
  const styles: Record<string, string> = {
    D00: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
    D10: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    D20: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    D40: 'bg-red-500/15 text-red-400 border-red-500/30',
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-mono font-medium border ${
        styles[code] || 'bg-slate-800 text-slate-300 border-slate-700'
      }`}
    >
      <span>{code}</span>
      {name && <span className="font-sans font-normal opacity-85">· {name}</span>}
    </span>
  );
};

export const ModelReadinessBanner: React.FC<{
  model: ModelRuntimeStatus | null;
  onReload?: () => void;
  reloading?: boolean;
}> = ({ model, onReload, reloading }) => {
  if (!model) return null;

  if (model.configured) {
    return (
      <div className="rg-card px-4 py-3 flex flex-wrap items-center justify-between gap-3 border-emerald-500/25 bg-emerald-950/10">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
          <div className="text-sm">
            <span className="font-medium text-slate-100">Trained RDD2022 Detector Active</span>
            <span className="text-slate-400 ml-2 font-mono text-xs">
              ({model.device.toUpperCase()}{model.gpu_name ? ` · ${model.gpu_name}` : ''})
            </span>
          </div>
        </div>
        {onReload && (
          <button onClick={onReload} disabled={reloading} className="rg-btn-secondary py-1.5 text-xs">
            <RefreshCw className={`w-3.5 h-3.5 ${reloading ? 'animate-spin' : ''}`} />
            Reload Weights
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="rg-card p-4 border-amber-500/35 bg-amber-950/15">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold text-amber-200">
              Deep Learning Checkpoint Not Yet Configured
            </h4>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed max-w-3xl">
              {model.message ||
                'No trained 4-class RDD2022 YOLO weights (models/best.pt) were found.'}{' '}
              RoadGuard AI refuses to run generic COCO weights or fabricate detections. Train on Google
              Colab using <code className="font-mono text-amber-300">notebooks/03_model_training.ipynb</code>,
              place <code className="font-mono text-amber-300">best.pt</code> into{' '}
              <code className="font-mono text-amber-300">models/best.pt</code>, and click Reload.
              Meanwhile, the <strong>Image Processing Lab</strong> is 100% operational.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {onReload && (
            <button onClick={onReload} disabled={reloading} className="rg-btn-secondary text-xs">
              <RefreshCw className={`w-3.5 h-3.5 ${reloading ? 'animate-spin' : ''}`} />
              Check & Reload
            </button>
          )}
          <Link to="/settings" className="rg-btn-secondary text-xs">
            <SettingsIcon className="w-3.5 h-3.5" />
            Configure Path
          </Link>
        </div>
      </div>
    </div>
  );
};
