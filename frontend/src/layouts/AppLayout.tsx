import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  Cpu,
  Database,
  FileText,
  Image as ImageIcon,
  LayoutDashboard,
  Menu,
  Settings,
  ShieldCheck,
  Sliders,
  Video,
  WifiOff,
  X,
} from 'lucide-react';
import { api } from '../services/api';
import { HealthStatus } from '../types';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/image-analysis', label: 'Image Analysis', icon: ImageIcon },
  { to: '/video-analytics', label: 'Video Analytics', icon: Video },
  { to: '/image-processing-lab', label: 'Image Processing Lab', icon: Sliders },
  { to: '/model-performance', label: 'Model Performance', icon: BarChart3 },
  { to: '/damage-reports', label: 'Damage Reports', icon: FileText },
  { to: '/dataset-explorer', label: 'Dataset Explorer', icon: Database },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export const AppLayout: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [apiOffline, setApiOffline] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    let active = true;
    const pollHealth = async () => {
      try {
        const res = await api.getHealth();
        if (active) {
          setHealth(res);
          setApiOffline(false);
        }
      } catch {
        if (active) setApiOffline(true);
      }
    };
    pollHealth();
    const timer = setInterval(pollHealth, 15000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-navy-900 border-r border-slate-800/80 flex flex-col transition-transform duration-200 lg:static lg:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="h-16 px-5 flex items-center justify-between border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="text-sm font-bold tracking-tight text-slate-100">RoadGuard AI</div>
              <div className="text-[11px] text-slate-400 font-mono">Infrastructure CV v1.0</div>
            </div>
          </div>
          <button
            className="lg:hidden text-slate-400 hover:text-slate-200"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-navy-800/70 border border-transparent'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800/80 space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5" /> API Telemetry
            </span>
            {apiOffline ? (
              <span className="text-red-400 font-medium">Offline</span>
            ) : (
              <span className="text-emerald-400 font-medium">Connected</span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" /> Detector
            </span>
            <span
              className={`font-mono ${
                health?.model_configured ? 'text-emerald-400' : 'text-amber-400'
              }`}
            >
              {health?.model_configured ? 'RDD2022 Ready' : 'Unconfigured'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 bg-navy-900/75 backdrop-blur border-b border-slate-800/80 px-6 flex items-center justify-between sticky top-0 z-30">
          <div className="flex items-center gap-3">
            <button
              className="lg:hidden p-1.5 rounded-lg bg-navy-800 border border-slate-700 text-slate-300"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-sm font-semibold text-slate-100">
                Intelligent Road Damage Detection, Assessment &amp; Monitoring
              </h1>
              <p className="text-xs text-slate-400 hidden sm:block">
                RDD2022 · Classical OpenCV Pipeline · YOLO Object Detection · ByteTrack/SORT Analytics
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span className="hidden md:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-navy-800 border border-slate-700/80 text-slate-300 font-mono">
              {health?.cuda_available ? `GPU: ${health.gpu_name || 'CUDA'}` : 'Compute: CPU Fallback'}
            </span>
          </div>
        </header>

        {apiOffline && (
          <div className="bg-red-950/60 border-b border-red-500/40 px-6 py-2.5 flex items-center gap-2.5 text-xs text-red-200">
            <WifiOff className="w-4 h-4 text-red-400 shrink-0" />
            <span>
              Backend API unreachable at <code className="font-mono">http://127.0.0.1:8000</code>.
              Start the FastAPI backend using <code className="font-mono">scripts\start_backend.cmd</code>.
            </span>
          </div>
        )}

        <main className="flex-1 p-6 max-w-[1440px] w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
