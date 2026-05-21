import type { AnalysisStatus } from "@/lib/types";

const STEPS: { status: AnalysisStatus; label: string }[] = [
  { status: "pending", label: "Queued" },
  { status: "processing", label: "Analyzing waves" },
  { status: "completed", label: "Done" },
];

export function AnalysisProgress({ status }: { status: AnalysisStatus }) {
  const order = ["pending", "processing", "completed", "failed"];
  const idx = status === "failed" ? -1 : order.indexOf(status);

  return (
    <div className="space-y-3" aria-live="polite">
      {status === "failed" ? (
        <p className="text-red-300 text-sm">Analysis failed. See error below.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {STEPS.map((step, i) => {
            const done = idx > i;
            const active = idx === i;
            return (
              <li
                key={step.status}
                className={`flex items-center gap-3 text-sm ${
                  done ? "text-swell-400" : active ? "text-foam-100" : "text-foam-600"
                }`}
              >
                <span
                  className={`flex h-6 w-6 items-center justify-center rounded-full border text-xs ${
                    done
                      ? "border-swell-500 bg-swell-500/20"
                      : active
                        ? "border-swell-400 animate-pulse"
                        : "border-ocean-700"
                  }`}
                >
                  {done ? "✓" : i + 1}
                </span>
                {step.label}
                {active && (
                  <span className="ml-auto text-xs text-foam-500">polling…</span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
