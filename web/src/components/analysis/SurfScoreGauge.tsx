export function SurfScoreGauge({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(100, Math.round(score)));

  let label = "Fair";
  if (clamped >= 80) label = "Epic";
  else if (clamped >= 65) label = "Good";
  else if (clamped >= 45) label = "Fair";
  else label = "Poor";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-40 w-40">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke="currentColor"
            strokeWidth="10"
            className="text-ocean-800"
          />
          <circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke="currentColor"
            strokeWidth="10"
            strokeDasharray={`${(clamped / 100) * 327} 327`}
            strokeLinecap="round"
            className="text-swell-400 transition-all duration-700"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-4xl font-bold text-foam-50">{clamped}</span>
          <span className="text-xs uppercase tracking-wider text-foam-400">{label}</span>
        </div>
      </div>
      <p className="text-sm text-foam-400">Surf score (0–100)</p>
    </div>
  );
}
