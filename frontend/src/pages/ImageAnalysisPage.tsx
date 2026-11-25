import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  Download,
  FileSpreadsheet,
  FileText,
  Info,
  Play,
  Sparkles,
  UploadCloud,
} from 'lucide-react';
import { api, ApiError } from '../services/api';
import { AnalysisSummary, ModelRuntimeStatus } from '../types';
import { ClassBadge, ModelReadinessBanner, PriorityBadge } from '../components/StatusBadges';
import { SplitImageCompare } from '../components/SplitImageCompare';

export const ImageAnalysisPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [imageSize, setImageSize] = useState(640);
  const [includeEigenCam, setIncludeEigenCam] = useState(false);

  const [modelStatus, setModelStatus] = useState<ModelRuntimeStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisSummary | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .getModelInfo()
      .then((info) => setModelStatus(info.runtime))
      .catch(() => {});
    api
      .getSettings()
      .then((s) => {
        setConfThreshold(s.default_conf_threshold);
        setIouThreshold(s.default_iou_threshold);
      })
      .catch(() => {});
  }, []);

  const handleSelectFile = (file: File | undefined) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleRunDetection = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      fd.append('conf_threshold', String(confThreshold));
      fd.append('iou_threshold', String(iouThreshold));
      fd.append('image_size', String(imageSize));
      fd.append('include_eigencam', String(includeEigenCam));
      const res = await api.analyzeImage(fd);
      setResult(res);
    } catch (err: any) {
      if (err instanceof ApiError && err.code === 'MODEL_NOT_CONFIGURED') {
        setError(err.message);
      } else {
        setError(err.message || 'Image inference failed.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-100">Road Surface Image Analysis</h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Multi-instance YOLO road damage detection with transparent 2D bounding-box visual priority scoring.
        </p>
      </div>

      <ModelReadinessBanner
        model={modelStatus}
        onReload={async () => {
          const st = await api.reloadModel();
          setModelStatus(st);
        }}
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls Column */}
        <div className="lg:col-span-4 space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleSelectFile(e.dataTransfer.files?.[0]);
            }}
            onClick={() => fileInputRef.current?.click()}
            className="rg-card p-6 border-dashed border-2 border-slate-700 hover:border-blue-500/70 cursor-pointer transition-colors text-center"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.webp"
              className="hidden"
              onChange={(e) => handleSelectFile(e.target.files?.[0])}
            />
            <UploadCloud className="w-9 h-9 text-blue-400 mx-auto mb-2.5" />
            <div className="text-sm font-medium text-slate-200">
              {selectedFile ? selectedFile.name : 'Drop road image here or click to browse'}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Supports JPG, PNG, BMP, WEBP (up to 25 MB)
            </p>
          </div>

          <div className="rg-card p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-100">Inference Parameters</h3>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300">Confidence Threshold</span>
                <span className="font-mono text-blue-400">{confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.9}
                step={0.05}
                value={confThreshold}
                onChange={(e) => setConfThreshold(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300">NMS IoU Threshold</span>
                <span className="font-mono text-blue-400">{iouThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.15}
                max={0.85}
                step={0.05}
                value={iouThreshold}
                onChange={(e) => setIouThreshold(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1.5">Inference Resolution</label>
              <select
                value={imageSize}
                onChange={(e) => setImageSize(Number(e.target.value))}
                className="rg-input w-full"
              >
                <option value={512}>512 × 512 px (Fast)</option>
                <option value={640}>640 × 640 px (Standard RDD2022)</option>
                <option value={960}>960 × 960 px (Small Far-Field Cracks)</option>
              </select>
            </div>

            <label className="flex items-start gap-2.5 text-xs text-slate-300 cursor-pointer pt-1">
              <input
                type="checkbox"
                checked={includeEigenCam}
                onChange={(e) => setIncludeEigenCam(e.target.checked)}
                className="mt-0.5 accent-blue-500"
              />
              <span>
                Generate <strong>EigenCAM</strong> neck feature activation overlay (class-agnostic SVD spatial saliency)
              </span>
            </label>

            <button
              onClick={handleRunDetection}
              disabled={!selectedFile || loading}
              className="rg-btn-primary w-full py-2.5"
            >
              <Play className="w-4 h-4" />
              {loading ? 'Running Road Damage Detection...' : 'Run Damage Detection'}
            </button>
          </div>

          <div className="rg-card p-4 bg-navy-950/60 text-xs text-slate-400 space-y-1.5">
            <div className="flex items-center gap-1.5 text-slate-200 font-medium">
              <Info className="w-3.5 h-3.5 text-blue-400" />
              Calibration &amp; Severity Note
            </div>
            <p className="leading-relaxed">
              Priority bands (Low / Medium / High) are transparent 2D visual estimates computed from
              normalized bounding-box area, category weight, and confidence. Physical crack depth or
              millimeter width requires calibrated stereo/LiDAR sensors.
            </p>
          </div>
        </div>

        {/* Visualization & Detections Column */}
        <div className="lg:col-span-8 space-y-6">
          {error && (
            <div className="rg-card p-4 border-red-500/40 bg-red-950/20 flex items-start gap-3 text-xs text-red-200">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold text-red-100">Detection Could Not Run</div>
                <p className="mt-0.5 leading-relaxed">{error}</p>
              </div>
            </div>
          )}

          {result && result.input_url && result.output_url ? (
            <div className="rg-card p-5 space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">
                    Inspection Output — {result.filename}
                  </h3>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">
                    {result.total_detections} detections ·{' '}
                    {result.summary?.inference_time_ms ?? 0} ms inference (
                    {result.summary?.fps ?? 0} FPS)
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <a
                    href={result.output_url}
                    download={`annotated_${result.filename}`}
                    className="rg-btn-secondary text-xs py-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Annotated PNG
                  </a>
                  <a
                    href={`/api/reports/${result.id}/export/csv`}
                    className="rg-btn-secondary text-xs py-1.5"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    CSV Report
                  </a>
                  <a
                    href={`/api/reports/${result.id}/export/pdf`}
                    className="rg-btn-secondary text-xs py-1.5"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    PDF Report
                  </a>
                </div>
              </div>

              <SplitImageCompare
                originalUrl={result.input_url}
                processedUrl={result.output_url}
                originalLabel="Original Road Frame"
                processedLabel="YOLO Detections"
              />

              {result.summary?.eigencam_url && (
                <div className="pt-3 border-t border-slate-800">
                  <div className="flex items-center gap-2 text-xs font-medium text-blue-300 mb-2">
                    <Sparkles className="w-3.5 h-3.5" />
                    EigenCAM Neck Activation Overlay
                  </div>
                  <img
                    src={result.summary.eigencam_url}
                    alt="EigenCAM activation map"
                    className="max-h-72 rounded-lg border border-slate-800 mx-auto"
                  />
                  <p className="text-[11px] text-slate-400 mt-1.5">{result.summary.eigencam_note}</p>
                </div>
              )}
            </div>
          ) : previewUrl ? (
            <div className="rg-card p-5">
              <div className="text-xs font-medium text-slate-400 mb-2">Selected Image Preview</div>
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full max-h-[460px] object-contain rounded-lg bg-navy-950 border border-slate-800"
              />
            </div>
          ) : (
            <div className="rg-card p-14 text-center text-sm text-slate-400">
              Upload a road surface photograph on the left to inspect damage instances and priority scores.
            </div>
          )}

          {result && (
            <div className="rg-card overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-100">
                  Detected Damage Instances ({result.detections.length})
                </h3>
                <span className="text-xs text-slate-400 font-mono">
                  High: {result.high_priority_count} · Med: {result.medium_priority_count} · Low:{' '}
                  {result.low_priority_count}
                </span>
              </div>
              {result.detections.length === 0 ? (
                <div className="p-8 text-center text-xs text-slate-400">
                  No road damage instances detected above confidence threshold {confThreshold.toFixed(2)}.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-navy-950/70 text-slate-400 uppercase">
                      <tr>
                        <th className="px-4 py-3">#</th>
                        <th className="px-4 py-3">Category</th>
                        <th className="px-4 py-3">Confidence</th>
                        <th className="px-4 py-3">Bounding Box (px)</th>
                        <th className="px-4 py-3">Dimensions</th>
                        <th className="px-4 py-3">Frame Area %</th>
                        <th className="px-4 py-3">Priority Score</th>
                        <th className="px-4 py-3">Band</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70">
                      {result.detections.map((d, idx) => (
                        <tr key={idx} className="hover:bg-navy-800/40">
                          <td className="px-4 py-3 font-mono text-slate-400">{idx + 1}</td>
                          <td className="px-4 py-3">
                            <ClassBadge code={d.class_code} name={d.class_name} />
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-200">
                            {(d.confidence * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-400">
                            [{d.x1.toFixed(0)}, {d.y1.toFixed(0)}, {d.x2.toFixed(0)}, {d.y2.toFixed(0)}]
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-300">
                            {d.box_width.toFixed(0)} × {d.box_height.toFixed(0)} px
                          </td>
                          <td className="px-4 py-3 font-mono text-slate-300">
                            {d.image_area_pct.toFixed(2)}%
                          </td>
                          <td className="px-4 py-3 font-mono font-semibold text-slate-100">
                            {d.priority_score.toFixed(1)} / 100
                          </td>
                          <td className="px-4 py-3">
                            <PriorityBadge band={d.priority_band} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
