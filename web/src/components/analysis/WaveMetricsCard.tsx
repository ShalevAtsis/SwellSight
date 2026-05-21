import type { WaveResultJson } from "@/lib/types";

export function WaveMetricsCard({ result }: { result: WaveResultJson }) {
  const heightM = result.wave_height_meters;
  const heightFt = result.wave_height_feet;
  const conf = result.overall_confidence;

  return (
    <div className="rounded-xl border border-ocean-800 bg-ocean-900/50 p-5 space-y-4">
      <h3 className="font-display text-lg font-semibold text-foam-100">Wave metrics</h3>
      <dl className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <dt className="text-foam-500">Height</dt>
          <dd className="text-lg font-semibold text-foam-100">
            {heightM != null ? `${heightM.toFixed(1)} m` : "—"}
            {heightFt != null && (
              <span className="ml-2 text-foam-400">({heightFt.toFixed(1)} ft)</span>
            )}
          </dd>
        </div>
        <div>
          <dt className="text-foam-500">Direction</dt>
          <dd className="text-lg font-semibold text-foam-100">
            {result.direction ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-foam-500">Breaking</dt>
          <dd className="text-lg font-semibold text-foam-100">
            {result.breaking_type ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-foam-500">Confidence</dt>
          <dd className="text-lg font-semibold text-foam-100">
            {conf != null ? `${Math.round(conf * 100)}%` : "—"}
          </dd>
        </div>
      </dl>
      {result.extreme_conditions && (
        <p className="text-sm text-amber-300" role="status">
          Extreme conditions detected — use extra caution.
        </p>
      )}
      {result.warnings && result.warnings.length > 0 && (
        <ul className="text-sm text-amber-200/90 list-disc pl-4 space-y-1">
          {result.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
