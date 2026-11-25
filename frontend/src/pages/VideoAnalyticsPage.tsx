import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  Download,
  FileSpreadsheet,
  FileText,
  Info,
  Loader2,
  Play,
  UploadCloud,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import { AnalysisSummary, ModelRuntimeStatus } from '../types';
import { ClassBadge, ModelReadinessBanner } from '../components/StatusBadges';

export const VideoAnalyticsPage: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [trackerType, setTrackerType] = useState<'bytetrack' | 'sort'>('bytetrack');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [frameStride, setFrameStride] = useState(1);

  const [modelStatus, setModelStatus] = useState<ModelRuntimeStatus | null>(null);
  const [job, setJob] = useState<AnalysisSummary | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .getModelInfo()
      .then((info) => setModelStatus(info.runtime))
      .catch(() => {});
  }, []);

  // Poll active video background job until completion or failure
  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'processing')) return;
    const timer = setInterval(async () => {
      try {
        const updated = await api.getVideoStatus(job.id);
        setJob(updated);
        if (updated.status === 'failed') {
          setError(updated.summary?.error || 'Video processing job failed.');
        }
      } catch (err: any) {
        setError(err.message);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [job]);

  const handleSelectVideo = (f: File | undefined) => {
    if (!f) return;
    setVideoFile(f);
    setJob(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const handleStartVideoAnalysis = async () => {
    if (!videoFile) return;
    setSubmitting(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', videoFile);
      fd.append('tracker_type', trackerType);
      fd.append('conf_threshold', String(confThreshold));
      fd.append('frame_stride', String(frameStride));
      const queued = await api.analyzeVideo(fd);
      setJob(queued);
    } catch (err: any) {
      setError(err.message || 'Failed to launch video analysis job.');
    } finally {
      setSubmitting(false);
    }
  };

  const isRunning = job && (job.status === 'queued' || job.status === 'processing');
  const vSummary = job?.summary || {};
  const timeline = vSummary.frame_timeline || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-100">Video Road Damage &amp; Tracking Analytics</h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Frame-by-frame YOLO inspection with ByteTrack two-stage and SORT multi-object trajectory tracking.
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
        {/* Left Controls */}
        <div className="lg:col-span-4 space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleSelectVideo(e.dataTransfer.files?.[0]);
            }}
            onClick={() => inputRef.current?.click()}
            className="rg-card p-6 border-dashed border-2 border-slate-700 hover:border-blue-500/70 cursor-pointer text-center"
          >
            <input
              ref={inputRef}
              type="file"
              accept=".mp4,.avi,.mov"
              className="hidden"
              onChange={(e) => handleSelectVideo(e.target.files?.[0])}
            />
            <UploadCloud className="w-9 h-9 text-blue-400 mx-auto mb-2.5" />
            <div className="text-sm font-medium text-slate-200">
              {videoFile ? videoFile.name : 'Upload dashcam footage (.MP4, .AVI, .MOV)'}
            </div>
            <p className="text-xs text-slate-500 mt-1">Asynchronous background processing up to 250 MB</p>
          </div>

          <div className="rg-card p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-100">Tracking &amp; Frame Controls</h3>

            <div>
              <label className="block text-xs text-slate-300 mb-1.5">Multi-Object Tracker</label>
              <select
                value={trackerType}
                onChange={(e) => setTrackerType(e.target.value as 'bytetrack' | 'sort')}
                className="rg-input w-full"
              >
                <option value="bytetrack">ByteTrack (Two-Stage High/Low Conf Association)</option>
                <option value="sort">SORT (Single-Stage Velocity + IoU Association)</option>
              </select>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300">Primary Confidence Threshold</span>
                <span className="font-mono text-blue-400">{confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={0.85}
                step={0.05}
                value={confThreshold}
                onChange={(e) => setConfThreshold(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1.5">Frame Sampling Stride</label>
              <select
                value={frameStride}
                onChange={(e) => setFrameStride(Number(e.target.value))}
                className="rg-input w-full"
              >
                <option value={1}>Every Frame (Stride = 1, Maximum Temporal Fidelity)</option>
                <option value={2}>Every 2nd Frame (Stride = 2, 2× Faster)</option>
                <option value={3}>Every 3rd Frame (Stride = 3, Rapid Corridor Scan)</option>
              </select>
            </div>

            <button
              onClick={handleStartVideoAnalysis}
              disabled={!videoFile || submitting || Boolean(isRunning)}
              className="rg-btn-primary w-full py-2.5"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing ({job?.progress_pct.toFixed(0)}%)...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Video Inspection
                </>
              )}
            </button>
          </div>

          <div className="rg-card p-4 bg-navy-950/60 text-xs text-slate-400 space-y-1.5">
            <div className="flex items-center gap-1.5 text-slate-200 font-medium">
              <Info className="w-3.5 h-3.5 text-amber-400" />
              Frame Detections vs. Local Track IDs
            </div>
            <p className="leading-relaxed">
              <strong>Frame-level detections</strong> sum every bounding box across all processed frames.
              <strong> Local Track IDs</strong> maintain short-term temporal continuity, not globally
              unique physical pothole counts (which require centimetre-grade GNSS/odometry registration).
            </p>
          </div>
        </div>

        {/* Right Video Output & Analytics */}
        <div className="lg:col-span-8 space-y-6">
          {error && (
            <div className="rg-card p-4 border-red-500/40 bg-red-950/20 flex items-start gap-3 text-xs text-red-200">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">Video Analytics Error</div>
                <p className="mt-0.5">{error}</p>
              </div>
            </div>
          )}

          {isRunning && job && (
            <div className="rg-card p-5 space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-200 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                  Background Frame Detection &amp; {trackerType.toUpperCase()} Association Running...
                </span>
                <span className="font-mono text-blue-400">{job.progress_pct.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 bg-navy-950 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${Math.max(4, job.progress_pct)}%` }}
                />
              </div>
            </div>
          )}

          {job?.status === 'completed' ? (
            <>
              {/* Video Summary KPIs */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {[
                  { label: 'Processed Frames', val: vSummary.processed_frames ?? 0 },
                  { label: 'Frame-Level Dets', val: job.total_detections },
                  { label: 'Local Track IDs', val: vSummary.unique_track_ids_observed ?? 0 },
                  { label: 'Avg Processing FPS', val: `${vSummary.average_processing_fps ?? 0}` },
                  { label: 'Wall Duration', val: `${job.processing_duration.toFixed(2)}s` },
                ].map((m) => (
                  <div key={m.label} className="rg-card p-3.5">
                    <div className="text-[11px] text-slate-400">{m.label}</div>
                    <div className="text-lg font-bold font-mono text-slate-100 mt-1">{m.val}</div>
                  </div>
                ))}
              </div>

              {/* Annotated Video Player & Downloads */}
              <div className="rg-card p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Annotated Tracking Output — {job.filename}
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Tracker: {(vSummary.tracker_used || trackerType).toUpperCase()} · Resolution:{' '}
                      {vSummary.video_width}×{vSummary.video_height}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {job.output_url && (
                      <a href={job.output_url} download className="rg-btn-primary text-xs py-1.5">
                        <Download className="w-3.5 h-3.5" />
                        Download Output Video
                      </a>
                    )}
                    <a
                      href={`/api/reports/${job.id}/export/csv`}
                      className="rg-btn-secondary text-xs py-1.5"
                    >
                      <FileSpreadsheet className="w-3.5 h-3.5" />
                      CSV Report
                    </a>
                    <a
                      href={`/api/reports/${job.id}/export/pdf`}
                      className="rg-btn-secondary text-xs py-1.5"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      PDF Summary
                    </a>
                  </div>
                </div>

                {job.output_url && (
                  <video
                    src={job.output_url}
                    controls
                    className="w-full max-h-[460px] rounded-xl bg-navy-950 border border-slate-800"
                  />
                )}

                {/* Class Frame-Level Distribution */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  {(['D00', 'D10', 'D20', 'D40'] as const).map((code) => (
                    <div
                      key={code}
                      className="p-3 rounded-lg bg-navy-950 border border-slate-800 flex items-center justify-between"
                    >
                      <ClassBadge code={code} />
                      <span className="font-mono text-sm font-semibold text-slate-100">
                        {vSummary.class_frame_detection_distribution?.[code] ?? 0}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Frame-Level Detection Timeline */}
              {timeline.length > 0 && (
                <div className="rg-card p-5">
                  <h3 className="text-sm font-semibold text-slate-100 mb-1">
                    Frame-Level Detection Density Over Time
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Active bounding boxes and high-priority instances per processed video frame
                  </p>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={timeline}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                        <XAxis dataKey="frame_index" stroke="#94A3B8" fontSize={11} />
                        <YAxis stroke="#94A3B8" fontSize={11} allowDecimals={false} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#0B101E',
                            borderColor: '#1E293B',
                            color: '#F1F5F9',
                            fontSize: '12px',
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="detection_count"
                          name="Frame Detections"
                          stroke="#3B82F6"
                          fill="#3B82F6"
                          fillOpacity={0.22}
                        />
                        <Area
                          type="monotone"
                          dataKey="high_priority_count"
                          name="High Priority"
                          stroke="#EF4444"
                          fill="#EF4444"
                          fillOpacity={0.2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          ) : previewUrl ? (
            <div className="rg-card p-5">
              <div className="text-xs font-medium text-slate-400 mb-2">Source Video Preview</div>
              <video
                src={previewUrl}
                controls
                className="w-full max-h-[440px] rounded-xl bg-navy-950 border border-slate-800"
              />
            </div>
          ) : (
            <div className="rg-card p-14 text-center text-sm text-slate-400">
              Select a road inspection video (.MP4, .AVI, or .MOV) to run multi-frame detection and object tracking.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
