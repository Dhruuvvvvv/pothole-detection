import React, { useEffect, useState } from 'react';
import { CheckCircle2, RefreshCw, Save, Sliders } from 'lucide-react';
import { api } from '../services/api';
import { ModelRuntimeStatus, RuntimeSettings } from '../types';

export const SettingsPage: React.FC = () => {
  const [form, setForm] = useState<RuntimeSettings>({
    low_max_threshold: 35,
    high_min_threshold: 65,
    area_weight: 0.4,
    category_weight: 0.35,
    confidence_weight: 0.25,
    reference_saturation_area_pct: 12,
    class_weights: { D00: 0.55, D10: 0.62, D20: 0.82, D40: 1.0 },
    default_conf_threshold: 0.25,
    default_iou_threshold: 0.45,
    checkpoint_path: '',
  });
  const [modelStatus, setModelStatus] = useState<ModelRuntimeStatus | null>(null);
  const [saving, setSaving] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getSettings(), api.getModelInfo()])
      .then(([s, m]) => {
        setForm(s);
        setModelStatus(m.runtime);
      })
      .catch((err) => setError(err.message));
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setSavedMsg(null);
    try {
      const updated = await api.updateSettings(form);
      setForm(updated);
      const m = await api.getModelInfo();
      setModelStatus(m.runtime);
      setSavedMsg('Configuration saved and applied across image & video assessment pipelines.');
    } catch (err: any) {
      setError(err.message || 'Failed to save configuration.');
    } finally {
      setSaving(false);
    }
  };

  const handleReloadWeights = async () => {
    setReloading(true);
    setError(null);
    try {
      const st = await api.reloadModel(form.checkpoint_path || undefined);
      setModelStatus(st);
      setSavedMsg(`Model reload executed: ${st.status}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setReloading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-xl font-bold text-slate-100">System &amp; Heuristic Configuration</h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Configure transparent Visual Priority Heuristic thresholds, category structural weights, and YOLO checkpoint paths.
        </p>
      </div>

      {savedMsg && (
        <div className="rg-card p-3.5 border-emerald-500/40 bg-emerald-950/20 flex items-center gap-2 text-xs text-emerald-200">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{savedMsg}</span>
        </div>
      )}

      {error && (
        <div className="rg-card p-3.5 border-red-500/40 bg-red-950/20 text-xs text-red-200">
          {error}
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Checkpoint Path */}
        <div className="rg-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">Model Checkpoint Management</h3>
            <span
              className={`text-xs font-mono ${
                modelStatus?.configured ? 'text-emerald-400' : 'text-amber-400'
              }`}
            >
              Status: {modelStatus?.status || 'unknown'}
            </span>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              value={form.checkpoint_path || ''}
              onChange={(e) => setForm({ ...form, checkpoint_path: e.target.value })}
              placeholder="./models/best.pt"
              className="rg-input flex-1 font-mono text-xs"
            />
            <button
              type="button"
              onClick={handleReloadWeights}
              disabled={reloading}
              className="rg-btn-secondary shrink-0"
            >
              <RefreshCw className={`w-4 h-4 ${reloading ? 'animate-spin' : ''}`} />
              Hot-Reload Checkpoint
            </button>
          </div>
          <p className="text-xs text-slate-400">
            Drop your Colab-trained <code className="font-mono text-slate-200">best.pt</code> into{' '}
            <code className="font-mono text-slate-200">models/best.pt</code> and click Hot-Reload — no backend restart required.
          </p>
        </div>

        {/* Priority Band Thresholds */}
        <div className="rg-card p-5 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Sliders className="w-4 h-4 text-blue-400" />
            Estimated Visual Priority Heuristic (0–100 Scale)
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Low → Medium Threshold
              </label>
              <input
                type="number"
                step="0.5"
                min="5"
                max="85"
                value={form.low_max_threshold}
                onChange={(e) => setForm({ ...form, low_max_threshold: Number(e.target.value) })}
                className="rg-input w-full font-mono"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Medium → High Threshold
              </label>
              <input
                type="number"
                step="0.5"
                min="15"
                max="98"
                value={form.high_min_threshold}
                onChange={(e) => setForm({ ...form, high_min_threshold: Number(e.target.value) })}
                className="rg-input w-full font-mono"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Area Saturation (% Frame)
              </label>
              <input
                type="number"
                step="0.5"
                min="1"
                max="50"
                value={form.reference_saturation_area_pct}
                onChange={(e) =>
                  setForm({ ...form, reference_saturation_area_pct: Number(e.target.value) })
                }
                className="rg-input w-full font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Bounding-Box Area Weight
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={form.area_weight}
                onChange={(e) => setForm({ ...form, area_weight: Number(e.target.value) })}
                className="rg-input w-full font-mono"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Category Structural Weight
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={form.category_weight}
                onChange={(e) => setForm({ ...form, category_weight: Number(e.target.value) })}
                className="rg-input w-full font-mono"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Detection Confidence Weight
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={form.confidence_weight}
                onChange={(e) => setForm({ ...form, confidence_weight: Number(e.target.value) })}
                className="rg-input w-full font-mono"
              />
            </div>
          </div>

          <div className="pt-2">
            <div className="text-xs font-medium text-slate-300 mb-2">
              Category Relative Risk Factors (0.0 – 1.0)
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {(['D00', 'D10', 'D20', 'D40'] as const).map((code) => (
                <div key={code}>
                  <label className="block text-xs font-mono text-slate-400 mb-1">{code}</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.05"
                    max="1.0"
                    value={form.class_weights[code] ?? 0.5}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        class_weights: {
                          ...form.class_weights,
                          [code]: Number(e.target.value),
                        },
                      })
                    }
                    className="rg-input w-full font-mono"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Default Inference Thresholds */}
        <div className="rg-card p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-100">Default Detector Thresholds</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-300 mb-1">
                Default Confidence Threshold
              </label>
              <input
                type="number"
                step="0.01"
                min="0.05"
                max="0.95"
                value={form.default_conf_threshold}
                onChange={(e) =>
                  setForm({ ...form, default_conf_threshold: Number(e.target.value) })
                }
                className="rg-input w-full font-mono"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Default NMS IoU Threshold</label>
              <input
                type="number"
                step="0.01"
                min="0.1"
                max="0.95"
                value={form.default_iou_threshold}
                onChange={(e) =>
                  setForm({ ...form, default_iou_threshold: Number(e.target.value) })
                }
                className="rg-input w-full font-mono"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button type="submit" disabled={saving} className="rg-btn-primary px-6 py-2.5">
            <Save className="w-4 h-4" />
            {saving ? 'Saving Settings...' : 'Save Configuration'}
          </button>
        </div>
      </form>
    </div>
  );
};
