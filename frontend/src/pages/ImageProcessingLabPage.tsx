import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Download,
  FlaskConical,
  Grid,
  Play,
  Sparkles,
  UploadCloud,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import {
  ClassicalOperationSpec,
  ClassicalProcessResponse,
  FilterComparisonResponse,
} from '../types';
import { SplitImageCompare } from '../components/SplitImageCompare';

/**
 * Generates an explicitly-labeled synthetic asphalt + crack calibration test pattern
 * on an offscreen HTML5 canvas so students/evaluators can test OpenCV filters immediately
 * even before selecting a road photograph from disk.
 */
async function createSyntheticCalibrationImageFile(): Promise<File> {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 420;
  const ctx = canvas.getContext('2d')!;

  // Asphalt base with realistic aggregate speckle noise
  const imgData = ctx.createImageData(canvas.width, canvas.height);
  let seed = 1337;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  for (let i = 0; i < imgData.data.length; i += 4) {
    const base = 92 + Math.floor((rand() - 0.5) * 48);
    // Occasional salt-and-pepper aggregate grain to showcase Median vs Gaussian filtering
    const impulse = rand() < 0.012 ? (rand() < 0.5 ? 25 : 225) : base;
    imgData.data[i] = impulse;
    imgData.data[i + 1] = impulse;
    imgData.data[i + 2] = Math.min(255, impulse + 4);
    imgData.data[i + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);

  // Simulated cast shadow band across right side (tests CLAHE / Gamma)
  const grad = ctx.createLinearGradient(340, 0, 640, 0);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,0.52)');
  ctx.fillStyle = grad;
  ctx.fillRect(340, 0, 300, 420);

  // Simulated longitudinal & branched alligator cracks
  ctx.strokeStyle = '#181A20';
  ctx.lineWidth = 3.2;
  ctx.beginPath();
  ctx.moveTo(150, 20);
  ctx.lineTo(168, 145);
  ctx.lineTo(142, 270);
  ctx.lineTo(164, 400);
  ctx.stroke();

  ctx.lineWidth = 2.0;
  ctx.beginPath();
  ctx.moveTo(168, 145);
  ctx.lineTo(240, 175);
  ctx.lineTo(295, 160);
  ctx.stroke();

  // Simulated dark pothole depression
  ctx.fillStyle = '#1B1E26';
  ctx.beginPath();
  ctx.ellipse(455, 255, 58, 38, 0.15, 0, Math.PI * 2);
  ctx.fill();

  const blob: Blob = await new Promise((resolve) =>
    canvas.toBlob((b) => resolve(b!), 'image/png')
  );
  return new File([blob], 'synthetic_asphalt_calibration_card.png', { type: 'image/png' });
}

export const ImageProcessingLabPage: React.FC = () => {
  const [catalog, setCatalog] = useState<ClassicalOperationSpec[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('Enhancement');
  const [selectedOpId, setSelectedOpId] = useState<string>('clahe');
  const [paramValues, setParamValues] = useState<Record<string, any>>({});

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [comparingFilters, setComparingFilters] = useState(false);
  const [filterKernel, setFilterKernel] = useState(5);

  const [result, setResult] = useState<ClassicalProcessResponse | null>(null);
  const [filterComparison, setFilterComparison] = useState<FilterComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .getClassicalCatalog()
      .then((res) => {
        setCatalog(res.operations);
      })
      .catch((err) => setError(err.message));
  }, []);

  const categories = useMemo(
    () => Array.from(new Set(catalog.map((op) => op.category))),
    [catalog]
  );

  const filteredOps = useMemo(
    () => catalog.filter((op) => op.category === selectedCategory),
    [catalog, selectedCategory]
  );

  const activeSpec = useMemo(
    () => catalog.find((op) => op.id === selectedOpId) || null,
    [catalog, selectedOpId]
  );

  // Initialize default parameter values whenever selected operation changes
  useEffect(() => {
    if (!activeSpec) return;
    const defaults: Record<string, any> = {};
    activeSpec.parameters.forEach((p) => {
      defaults[p.name] = p.default;
    });
    setParamValues(defaults);
  }, [activeSpec]);

  const handleChooseFile = (f: File | undefined) => {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setFilterComparison(null);
    setError(null);
  };

  const handleLoadCalibrationPattern = async () => {
    const synth = await createSyntheticCalibrationImageFile();
    handleChooseFile(synth);
  };

  const handleExecuteOperation = async () => {
    if (!file || !activeSpec) return;
    setLoading(true);
    setError(null);
    setFilterComparison(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('operation', activeSpec.id);
      fd.append('parameters_json', JSON.stringify(paramValues));
      const res = await api.processClassicalImage(fd);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Classical OpenCV operation failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleCompareAllFilters = async () => {
    if (!file) return;
    setComparingFilters(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('kernel_size', String(filterKernel));
      const res = await api.compareClassicalFilters(fd);
      setFilterComparison(res);
    } catch (err: any) {
      setError(err.message || 'Filter comparison failed.');
    } finally {
      setComparingFilters(false);
    }
  };

  const mergedHistogram = useMemo(() => {
    if (!result) return [];
    return result.original_histogram.map((origBin, idx) => ({
      intensity: origBin.intensity,
      origLuminance: origBin.luminance,
      procLuminance: result.processed_histogram[idx]?.luminance ?? 0,
    }));
  }, [result]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">
            Classical Digital Image Processing Laboratory
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Independent OpenCV laboratory for spatial filtering, contrast enhancement, gradient edge operators, morphological analysis &amp; PSNR/SSIM evaluation.
          </p>
        </div>
        <button onClick={handleLoadCalibrationPattern} className="rg-btn-secondary text-xs">
          <Sparkles className="w-3.5 h-3.5 text-blue-400" />
          Load Synthetic Calibration Pattern
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Operator Workbench */}
        <div className="lg:col-span-4 space-y-4">
          {/* Upload Box */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleChooseFile(e.dataTransfer.files?.[0]);
            }}
            onClick={() => fileRef.current?.click()}
            className="rg-card p-5 border-dashed border-2 border-slate-700 hover:border-blue-500/70 cursor-pointer text-center"
          >
            <input
              ref={fileRef}
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.webp"
              className="hidden"
              onChange={(e) => handleChooseFile(e.target.files?.[0])}
            />
            <UploadCloud className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <div className="text-xs font-medium text-slate-200">
              {file ? file.name : 'Upload road surface image for OpenCV processing'}
            </div>
          </div>

          {/* Category & Operation Picker */}
          <div className="rg-card p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <FlaskConical className="w-4 h-4 text-blue-400" />
              Operator Configuration
            </div>

            <div className="flex flex-wrap gap-1.5">
              {categories.map((cat) => (
                <button
                  key={cat}
                  type="button"
                  onClick={() => {
                    setSelectedCategory(cat);
                    const first = catalog.find((o) => o.category === cat);
                    if (first) setSelectedOpId(first.id);
                  }}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                    selectedCategory === cat
                      ? 'bg-blue-600 text-white'
                      : 'bg-navy-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1.5">Select Technique</label>
              <select
                value={selectedOpId}
                onChange={(e) => setSelectedOpId(e.target.value)}
                className="rg-input w-full"
              >
                {filteredOps.map((op) => (
                  <option key={op.id} value={op.id}>
                    {op.name}
                  </option>
                ))}
              </select>
            </div>

            {activeSpec && (
              <div className="p-3 rounded-lg bg-navy-950 border border-slate-800/90 space-y-1.5">
                <p className="text-xs text-slate-300 leading-relaxed">{activeSpec.description}</p>
                <div className="text-[11px] font-mono text-cyan-400 pt-1 border-t border-slate-800/80">
                  {activeSpec.formula}
                </div>
              </div>
            )}

            {/* Dynamic Parameters */}
            {activeSpec && activeSpec.parameters.length > 0 && (
              <div className="space-y-3 pt-1">
                {activeSpec.parameters.map((param) => (
                  <div key={param.name}>
                    {param.type === 'number' && (
                      <>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-300">{param.label}</span>
                          <span className="font-mono text-blue-400">
                            {paramValues[param.name] ?? param.default}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          value={paramValues[param.name] ?? param.default}
                          onChange={(e) =>
                            setParamValues((prev) => ({
                              ...prev,
                              [param.name]: Number(e.target.value),
                            }))
                          }
                          className="w-full accent-blue-500"
                        />
                      </>
                    )}

                    {param.type === 'select' && (
                      <>
                        <label className="block text-xs text-slate-300 mb-1">{param.label}</label>
                        <select
                          value={paramValues[param.name] ?? param.default}
                          onChange={(e) =>
                            setParamValues((prev) => ({
                              ...prev,
                              [param.name]: e.target.value,
                            }))
                          }
                          className="rg-input w-full"
                        >
                          {(param.options || []).map((opt) => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      </>
                    )}

                    {param.type === 'boolean' && (
                      <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={Boolean(paramValues[param.name] ?? param.default)}
                          onChange={(e) =>
                            setParamValues((prev) => ({
                              ...prev,
                              [param.name]: e.target.checked,
                            }))
                          }
                          className="accent-blue-500"
                        />
                        <span>{param.label}</span>
                      </label>
                    )}
                  </div>
                ))}
              </div>
            )}

            <button
              onClick={handleExecuteOperation}
              disabled={!file || loading}
              className="rg-btn-primary w-full py-2.5"
            >
              <Play className="w-4 h-4" />
              {loading ? 'Executing OpenCV Pipeline...' : 'Apply Operation'}
            </button>
          </div>

          {/* 4-Filter Side-by-Side Benchmark Card */}
          <div className="rg-card p-5 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <Grid className="w-4 h-4 text-cyan-400" />
              Spatial Filter Benchmark
            </div>
            <p className="text-xs text-slate-400">
              Compare Gaussian, Median, Bilateral, and Average filtering simultaneously at identical aperture size.
            </p>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">Filter Kernel Size (odd)</span>
                <span className="font-mono text-cyan-400">{filterKernel}×{filterKernel}</span>
              </div>
              <input
                type="range"
                min={3}
                max={21}
                step={2}
                value={filterKernel}
                onChange={(e) => setFilterKernel(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>
            <button
              onClick={handleCompareAllFilters}
              disabled={!file || comparingFilters}
              className="rg-btn-secondary w-full"
            >
              {comparingFilters ? 'Running 4-Filter Matrix...' : 'Compare All 4 Spatial Filters'}
            </button>
          </div>
        </div>

        {/* Right Output & Quantitative Telemetry */}
        <div className="lg:col-span-8 space-y-6">
          {error && (
            <div className="rg-card p-4 border-red-500/40 bg-red-950/20 flex items-start gap-2.5 text-xs text-red-200">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {filterComparison ? (
            <div className="rg-card p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">
                    Multi-Filter Benchmark Matrix (k = {filterComparison.kernel_size})
                  </h3>
                  <p className="text-xs text-slate-400">
                    Quantitative trade-off between noise attenuation (PSNR/SSIM) and edge preservation (Laplacian Variance)
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filterComparison.comparisons.map((c) => (
                  <div
                    key={c.operation}
                    className="rounded-xl border border-slate-800 bg-navy-950 p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-100">{c.label}</span>
                      <span className="font-mono text-slate-400">{c.processing_time_ms} ms</span>
                    </div>
                    <img
                      src={c.processed_url}
                      alt={c.label}
                      className="w-full max-h-56 object-contain rounded-lg bg-navy-900"
                    />
                    <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono text-slate-300">
                      <div>PSNR: {c.metrics.psnr_db ?? 'N/A'} dB</div>
                      <div>SSIM: {c.metrics.ssim_score ?? 'N/A'}</div>
                      <div>LapVar: {c.metrics.laplacian_variance.toFixed(0)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : result ? (
            <>
              <div className="rg-card p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      {activeSpec?.name || result.operation} — Interactive Comparison
                    </h3>
                    <p className="text-xs text-slate-400 font-mono mt-0.5">
                      Execution Time: {result.processing_time_ms} ms · Resolution:{' '}
                      {result.processed_metrics.width}×{result.processed_metrics.height}
                    </p>
                  </div>
                  <a
                    href={result.processed_url}
                    download={`processed_${result.operation}_${result.filename}`}
                    className="rg-btn-primary text-xs py-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download Processed Image
                  </a>
                </div>

                <SplitImageCompare
                  originalUrl={result.original_url}
                  processedUrl={result.processed_url}
                  differenceUrl={result.difference_url}
                  originalLabel="Original Input"
                  processedLabel={activeSpec?.name || result.operation}
                />
              </div>

              {/* Quantitative Metrics Table */}
              <div className="rg-card overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-800">
                  <h3 className="text-sm font-semibold text-slate-100">
                    Quantitative Signal &amp; Texture Analysis
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-navy-950/70 text-slate-400 uppercase">
                      <tr>
                        <th className="px-5 py-2.5">Metric</th>
                        <th className="px-5 py-2.5">Original Image</th>
                        <th className="px-5 py-2.5">Processed Image</th>
                        <th className="px-5 py-2.5">Delta / Reference</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70 font-mono">
                      {[
                        {
                          label: 'Mean Luminance Intensity',
                          o: result.original_metrics.mean_intensity,
                          p: result.processed_metrics.mean_intensity,
                        },
                        {
                          label: 'Standard Deviation (RMS Contrast)',
                          o: result.original_metrics.std_deviation,
                          p: result.processed_metrics.std_deviation,
                        },
                        {
                          label: 'Michelson Contrast',
                          o: result.original_metrics.michelson_contrast,
                          p: result.processed_metrics.michelson_contrast,
                        },
                        {
                          label: 'Shannon Entropy (bits)',
                          o: result.original_metrics.shannon_entropy,
                          p: result.processed_metrics.shannon_entropy,
                        },
                        {
                          label: 'Laplacian Variance (Sharpness)',
                          o: result.original_metrics.laplacian_variance,
                          p: result.processed_metrics.laplacian_variance,
                        },
                        {
                          label: 'Canny Edge Pixel Density',
                          o: result.original_metrics.edge_pixel_density,
                          p: result.processed_metrics.edge_pixel_density,
                        },
                      ].map((row) => {
                        const delta = row.p - row.o;
                        return (
                          <tr key={row.label}>
                            <td className="px-5 py-2.5 font-sans text-slate-300">{row.label}</td>
                            <td className="px-5 py-2.5 text-slate-300">{row.o}</td>
                            <td className="px-5 py-2.5 text-blue-300 font-semibold">{row.p}</td>
                            <td
                              className={`px-5 py-2.5 ${
                                delta >= 0 ? 'text-emerald-400' : 'text-amber-400'
                              }`}
                            >
                              {delta >= 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}
                            </td>
                          </tr>
                        );
                      })}
                      <tr className="bg-navy-950/40">
                        <td className="px-5 py-2.5 font-sans text-slate-200 font-medium">
                          PSNR / SSIM (vs. Original Reference)
                        </td>
                        <td className="px-5 py-2.5 text-slate-500">Reference</td>
                        <td className="px-5 py-2.5 text-cyan-300 font-semibold" colSpan={2}>
                          PSNR:{' '}
                          {result.processed_metrics.psnr_db !== null &&
                          result.processed_metrics.psnr_db !== undefined
                            ? `${result.processed_metrics.psnr_db} dB`
                            : 'N/A (Resized)'}{' '}
                          &nbsp;|&nbsp; SSIM:{' '}
                          {result.processed_metrics.ssim_score !== null &&
                          result.processed_metrics.ssim_score !== undefined
                            ? result.processed_metrics.ssim_score
                            : 'N/A'}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Luminance Histogram Comparison */}
              <div className="rg-card p-5">
                <h3 className="text-sm font-semibold text-slate-100 mb-1">
                  Luminance Intensity Histogram Distribution (0–255)
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Pixel density (% of frame) illustrating dynamic range compression, stretching, or equalization
                </p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={mergedHistogram}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="intensity" stroke="#94A3B8" fontSize={11} />
                      <YAxis stroke="#94A3B8" fontSize={11} unit="%" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0B101E',
                          borderColor: '#1E293B',
                          color: '#F1F5F9',
                          fontSize: '12px',
                        }}
                      />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="origLuminance"
                        name="Original Luminance %"
                        stroke="#94A3B8"
                        fill="#94A3B8"
                        fillOpacity={0.18}
                      />
                      <Area
                        type="monotone"
                        dataKey="procLuminance"
                        name="Processed Luminance %"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.28}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : previewUrl ? (
            <div className="rg-card p-5">
              <div className="text-xs font-medium text-slate-400 mb-2">
                Loaded Input Image — Choose an OpenCV operator on the left and click Apply Operation
              </div>
              <img
                src={previewUrl}
                alt="Input preview"
                className="w-full max-h-[460px] object-contain rounded-lg bg-navy-950 border border-slate-800"
              />
            </div>
          ) : (
            <div className="rg-card p-14 text-center space-y-3">
              <p className="text-sm text-slate-300 font-medium">
                Ready for Digital Image Processing Experiments
              </p>
              <p className="text-xs text-slate-400 max-w-lg mx-auto">
                Upload any road photograph on the left, or click{' '}
                <strong>Load Synthetic Calibration Pattern</strong> above to evaluate filtering, CLAHE,
                gamma correction, Canny/Sobel edge operators, and morphological operations immediately.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
