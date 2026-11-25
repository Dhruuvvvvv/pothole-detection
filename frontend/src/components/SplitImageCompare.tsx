import React, { useState } from 'react';
import { Columns, Eye, Flame } from 'lucide-react';

interface SplitImageCompareProps {
  originalUrl: string;
  processedUrl: string;
  differenceUrl?: string | null;
  originalLabel?: string;
  processedLabel?: string;
}

export const SplitImageCompare: React.FC<SplitImageCompareProps> = ({
  originalUrl,
  processedUrl,
  differenceUrl,
  originalLabel = 'Original',
  processedLabel = 'Processed',
}) => {
  const [mode, setMode] = useState<'split' | 'side' | 'diff'>('split');
  const [sliderPct, setSliderPct] = useState<number>(50);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-lg bg-navy-950 p-1 border border-slate-800">
          <button
            type="button"
            onClick={() => setMode('split')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              mode === 'split' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            Interactive Curtain
          </button>
          <button
            type="button"
            onClick={() => setMode('side')}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              mode === 'side' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            Side-by-Side
          </button>
          {differenceUrl && (
            <button
              type="button"
              onClick={() => setMode('diff')}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                mode === 'diff' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Flame className="w-3.5 h-3.5" />
              Difference Heatmap
            </button>
          )}
        </div>

        {mode === 'split' && (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>Split Position:</span>
            <input
              type="range"
              min={0}
              max={100}
              value={sliderPct}
              onChange={(e) => setSliderPct(Number(e.target.value))}
              className="w-32 accent-blue-500 cursor-pointer"
            />
            <span className="font-mono w-9 text-right text-slate-300">{sliderPct}%</span>
          </div>
        )}
      </div>

      {mode === 'split' && (
        <div className="relative w-full overflow-hidden rounded-xl border border-slate-800 bg-navy-950 select-none">
          <img
            src={processedUrl}
            alt={processedLabel}
            className="w-full max-h-[520px] object-contain mx-auto block"
          />
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - sliderPct}% 0 0)` }}
          >
            <img
              src={originalUrl}
              alt={originalLabel}
              className="w-full max-h-[520px] object-contain mx-auto block"
            />
          </div>
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.9)] pointer-events-none"
            style={{ left: `${sliderPct}%` }}
          />
          <span className="absolute top-3 left-3 px-2.5 py-1 rounded bg-navy-950/85 border border-slate-700 text-xs font-medium text-slate-200">
            {originalLabel}
          </span>
          <span className="absolute top-3 right-3 px-2.5 py-1 rounded bg-blue-950/85 border border-blue-500/40 text-xs font-medium text-blue-200">
            {processedLabel}
          </span>
        </div>
      )}

      {mode === 'side' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-slate-800 bg-navy-950 p-2">
            <div className="text-xs font-medium text-slate-400 mb-2 px-1">{originalLabel}</div>
            <img src={originalUrl} alt={originalLabel} className="w-full max-h-[420px] object-contain mx-auto rounded-lg" />
          </div>
          <div className="rounded-xl border border-slate-800 bg-navy-950 p-2">
            <div className="text-xs font-medium text-blue-400 mb-2 px-1">{processedLabel}</div>
            <img src={processedUrl} alt={processedLabel} className="w-full max-h-[420px] object-contain mx-auto rounded-lg" />
          </div>
        </div>
      )}

      {mode === 'diff' && differenceUrl && (
        <div className="rounded-xl border border-slate-800 bg-navy-950 p-3">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
            <span>Amplified Absolute Pixel Difference (Turbo Perceptual Colormap)</span>
            <span className="font-mono">2.5x |I_orig - I_proc|</span>
          </div>
          <img
            src={differenceUrl}
            alt="Difference heatmap"
            className="w-full max-h-[480px] object-contain mx-auto rounded-lg"
          />
        </div>
      )}
    </div>
  );
};
