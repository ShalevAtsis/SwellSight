import type { ScoreBreakdown as Breakdown } from "@/lib/types";

const LABELS: Record<keyof Breakdown, { label: string; hint: string }> = {
  wave_quality: {
    label: "Wave quality",
    hint: "Breaking type and shape — cleaner walls score higher.",
  },
  size_factor: {
    label: "Size",
    hint: "Height vs ideal surf range (~1.5 m). Too flat or too huge lowers score.",
  },
  confidence_factor: {
    label: "Confidence",
    hint: "Model certainty on height, direction, and breaking.",
  },
  safety_penalty: {
    label: "Safety",
    hint: "Penalty for extreme or hazardous conditions.",
  },
};

export function ScoreBreakdownPanel({ breakdown }: { breakdown: Breakdown }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {(Object.keys(LABELS) as (keyof Breakdown)[]).map((key) => {
        const meta = LABELS[key];
        const value = breakdown[key];
        const pct = Math.round(value * 100);
        return (
          <div
            key={key}
            className="rounded-xl border border-ocean-800 bg-ocean-900/50 p-4"
            title={meta.hint}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-foam-200">{meta.label}</span>
              <span className="text-sm tabular-nums text-swell-300">{pct}%</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-ocean-800">
              <div
                className="h-full rounded-full bg-swell-500 transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-foam-500">{meta.hint}</p>
          </div>
        );
      })}
    </div>
  );
}
