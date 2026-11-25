import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AlertOctagon,
  ArrowRight,
  Camera,
  CheckCircle,
  FileSearch,
  Layers,
  RefreshCw,
  Sliders,
  Video,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  AreaChart,
  Area,
} from 'recharts';
import { api } from '../services/api';
import { AnalyticsDistribution, DashboardSummary, HistoryPoint } from '../types';
import { ClassBadge, ModelReadinessBanner, PriorityBadge } from '../components/StatusBadges';
import { EmptyState } from '../components/EmptyState';

export const OverviewPage: React.FC = () => {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [dist, setDist] = useState<AnalyticsDistribution | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [reloadingModel, setReloadingModel] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, d, h] = await Promise.all([
        api.getDashboardSummary(),
        api.getAnalyticsDistribution(),
        api.getAnalyticsHistory(30),
      ]);
      setSummary(s);
      setDist(d);
      setHistory(h.timeline);
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard telemetry.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  const handleReloadModel = async () => {
    setReloadingModel(true);
    try {
      await api.reloadModel();
      await loadDashboard();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setReloadingModel(false);
    }
  };

  if (loading && !summary) {
    return (
      <div className="rg-card p-12 flex items-center justify-center text-sm text-slate-400">
        <RefreshCw className="w-4 h-4 animate-spin mr-2 text-blue-400" />
        Loading infrastructure monitoring telemetry...
      </div>
    );
  }

  if (error && !summary) {
    return (
      <div className="rg-card p-6 border-red-500/40 bg-red-950/20 text-sm text-red-200">
        <div className="font-semibold mb-1">Telemetry Unavailable</div>
        <p className="text-xs text-red-300">{error}</p>
        <button onClick={loadDashboard} className="rg-btn-secondary mt-4 text-xs">
          Retry Connection
        </button>
      </div>
    );
  }

  if (!summary) return null;

  const kpis = [
    {
      label: 'Images Analysed',
      value: summary.total_images_analysed,
      sub: `${summary.total_videos_analysed} video streams`,
      icon: Camera,
      accent: 'text-blue-400',
    },
    {
      label: 'Total Detections',
      value: summary.total_damage_instances,
      sub: `Across ${summary.total_analyses} completed runs`,
      icon: Layers,
      accent: 'text-cyan-400',
    },
    {
      label: 'High-Priority Defects',
      value: summary.high_priority_detections,
      sub: `${summary.medium_priority_detections} medium · ${summary.low_priority_detections} low`,
      icon: AlertOctagon,
      accent: 'text-red-400',
    },
    {
      label: 'D00 · Longitudinal',
      value: summary.d00_count,
      sub: 'Wheel-path linear cracks',
      icon: CheckCircle,
      accent: 'text-blue-400',
    },
    {
      label: 'D10 · Transverse',
      value: summary.d10_count,
      sub: 'Perpendicular thermal cracks',
      icon: CheckCircle,
      accent: 'text-cyan-400',
    },
    {
      label: 'D20 · Alligator',
      value: summary.d20_count,
      sub: 'Fatigue network cracking',
      icon: CheckCircle,
      accent: 'text-amber-400',
    },
    {
      label: 'D40 · Potholes',
      value: summary.d40_count,
      sub: 'Localized surface cavities',
      icon: AlertOctagon,
      accent: 'text-red-400',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Infrastructure Monitoring Overview</h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Live SQLite/PostgreSQL database aggregation — zero hardcoded or synthetic sample figures.
          </p>
        </div>
        <button onClick={loadDashboard} className="rg-btn-secondary text-xs">
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh Metrics
        </button>
      </div>

      <ModelReadinessBanner
        model={summary.model_info}
        onReload={handleReloadModel}
        reloading={reloadingModel}
      />

      {/* KPI Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-3.5">
        {kpis.map((k) => {
          const Icon = k.icon;
          return (
            <div key={k.label} className="rg-card p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="truncate">{k.label}</span>
                <Icon className={`w-4 h-4 shrink-0 ${k.accent}`} />
              </div>
              <div className="mt-3">
                <div className="text-2xl font-bold font-mono text-slate-100">{k.value}</div>
                <div className="text-[11px] text-slate-500 mt-0.5 truncate">{k.sub}</div>
              </div>
            </div>
          );
        })}
      </div>

      {summary.total_analyses === 0 ? (
        <EmptyState
          icon={FileSearch}
          title="No Inspection History Recorded Yet"
          description="Dashboard analytics populate strictly from real image and video inspections executed by RoadGuard AI. Explore classical OpenCV operations in the Image Processing Lab right away, or run road damage detection once your RDD2022 checkpoint is loaded."
          action={
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Link to="/image-processing-lab" className="rg-btn-primary">
                <Sliders className="w-4 h-4" />
                Open Image Processing Lab
              </Link>
              <Link to="/image-analysis" className="rg-btn-secondary">
                <Camera className="w-4 h-4" />
                Image Damage Analysis
              </Link>
              <Link to="/video-analytics" className="rg-btn-secondary">
                <Video className="w-4 h-4" />
                Video Stream Analytics
              </Link>
            </div>
          }
        />
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Damage Category Distribution */}
            <div className="rg-card p-5">
              <h3 className="text-sm font-semibold text-slate-100 mb-1">
                Damage Category Distribution
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Cumulative detected instances across RDD2022 defect classes
              </p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dist?.categories || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="code" stroke="#94A3B8" fontSize={12} />
                    <YAxis stroke="#94A3B8" fontSize={12} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0B101E',
                        borderColor: '#1E293B',
                        color: '#F1F5F9',
                        fontSize: '12px',
                      }}
                    />
                    <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                      {(dist?.categories || []).map((entry) => (
                        <Cell key={entry.code} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Analysis Activity Chart */}
            <div className="rg-card p-5">
              <h3 className="text-sm font-semibold text-slate-100 mb-1">
                Inspection Activity &amp; Detections Timeline
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Daily completed inspections and detected surface anomalies
              </p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="date" stroke="#94A3B8" fontSize={11} />
                    <YAxis stroke="#94A3B8" fontSize={12} allowDecimals={false} />
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
                      dataKey="detections"
                      name="Detections"
                      stroke="#3B82F6"
                      fill="#3B82F6"
                      fillOpacity={0.2}
                    />
                    <Area
                      type="monotone"
                      dataKey="high_priority"
                      name="High Priority"
                      stroke="#EF4444"
                      fill="#EF4444"
                      fillOpacity={0.18}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Recent Analyses Table */}
          <div className="rg-card overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">Recent Inspection Runs</h3>
                <p className="text-xs text-slate-400">Latest persisted image and video analyses</p>
              </div>
              <Link
                to="/damage-reports"
                className="text-xs text-blue-400 hover:text-blue-300 inline-flex items-center gap-1"
              >
                View All Reports <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-navy-950/70 text-slate-400 uppercase tracking-wider">
                  <tr>
                    <th className="px-5 py-3">ID</th>
                    <th className="px-5 py-3">Source File</th>
                    <th className="px-5 py-3">Modality</th>
                    <th className="px-5 py-3">Detections</th>
                    <th className="px-5 py-3">Highest Risk</th>
                    <th className="px-5 py-3">Duration</th>
                    <th className="px-5 py-3">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {summary.recent_analyses.map((item) => {
                    const topBand =
                      item.high_priority_count > 0
                        ? 'High'
                        : item.medium_priority_count > 0
                        ? 'Medium'
                        : 'Low';
                    return (
                      <tr key={item.id} className="hover:bg-navy-800/40 transition-colors">
                        <td className="px-5 py-3 font-mono text-slate-300">#{item.id}</td>
                        <td className="px-5 py-3 font-medium text-slate-100">{item.filename}</td>
                        <td className="px-5 py-3 uppercase font-mono text-slate-400">
                          {item.analysis_type}
                        </td>
                        <td className="px-5 py-3 font-mono text-slate-200">
                          {item.total_detections}
                        </td>
                        <td className="px-5 py-3">
                          {item.total_detections > 0 ? (
                            <PriorityBadge band={topBand} />
                          ) : (
                            <span className="text-slate-500">None</span>
                          )}
                        </td>
                        <td className="px-5 py-3 font-mono text-slate-400">
                          {item.processing_duration.toFixed(2)}s
                        </td>
                        <td className="px-5 py-3 text-slate-400">
                          {new Date(item.timestamp).toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Active Model Runtime Card */}
      <div className="rg-card p-5">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Detector Runtime &amp; Target Class Specification
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            { code: 'D00', name: 'Longitudinal Crack', desc: 'Wheel-path cracks parallel to travel' },
            { code: 'D10', name: 'Transverse Crack', desc: 'Thermal cracks perpendicular to lane' },
            { code: 'D20', name: 'Alligator Crack', desc: 'Fatigue structural mesh cracking' },
            { code: 'D40', name: 'Pothole', desc: 'Bowl-shaped localized surface cavity' },
          ].map((c) => (
            <div key={c.code} className="p-3 rounded-lg bg-navy-950 border border-slate-800/80">
              <ClassBadge code={c.code} name={c.name} />
              <p className="text-xs text-slate-400 mt-2">{c.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
