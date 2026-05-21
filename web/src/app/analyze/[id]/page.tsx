"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { AnalysisProgress } from "@/components/analysis/AnalysisProgress";
import { ScoreBreakdownPanel } from "@/components/analysis/ScoreBreakdown";
import { SurfScoreGauge } from "@/components/analysis/SurfScoreGauge";
import { WaveMetricsCard } from "@/components/analysis/WaveMetricsCard";
import { ShareResultButton } from "@/components/analysis/ShareResultButton";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { useAuth } from "@/components/providers/AuthProvider";
import { useAnalysisPoll } from "@/hooks/useAnalysisPoll";

export default function AnalysisDetailPage() {
  const params = useParams();
  const id = typeof params.id === "string" ? params.id : null;
  const { token } = useAuth();
  const { analysis, error, polling } = useAnalysisPoll(token, id);

  return (
    <RequireAuth>
      <div className="mx-auto max-w-2xl px-4 py-10 space-y-8">
        <div>
          <Link href="/history" className="text-sm text-swell-400 hover:underline">
            ← History
          </Link>
          <h1 className="mt-2 font-display text-2xl font-bold text-foam-50">
            Analysis results
          </h1>
          {id && (
            <p className="mt-1 text-xs font-mono text-foam-600 truncate">{id}</p>
          )}
        </div>

        {error && <ErrorAlert message={error} />}

        {analysis && (
          <>
            <AnalysisProgress status={analysis.status} />

            {analysis.status === "failed" && (
              <ErrorAlert
                message={
                  analysis.error_message ||
                  "Analysis failed. Try another photo or check that the worker is running."
                }
              />
            )}

            {analysis.status === "completed" && (
              <div className="space-y-8">
                {analysis.surf_score != null && (
                  <div className="flex justify-center">
                    <SurfScoreGauge score={analysis.surf_score} />
                  </div>
                )}
                {analysis.result_json && (
                  <WaveMetricsCard result={analysis.result_json} />
                )}
                {analysis.score_breakdown && (
                  <section>
                    <h2 className="mb-3 font-display text-lg font-semibold text-foam-100">
                      Score breakdown
                    </h2>
                    <ScoreBreakdownPanel breakdown={analysis.score_breakdown} />
                  </section>
                )}
                {analysis.model_version && (
                  <p className="text-xs text-foam-600">
                    Model: {analysis.model_version}
                  </p>
                )}
                {id && <ShareResultButton analysisId={id} />}
              </div>
            )}

            {!["completed", "failed"].includes(analysis.status) && polling && (
              <div className="text-sm text-foam-500 text-center space-y-2">
                <p>Checking every 2 seconds…</p>
                <p className="text-xs text-foam-600">
                  If this takes more than a few minutes, ensure the GPU worker is
                  running and a checkpoint is configured.
                </p>
              </div>
            )}
          </>
        )}

        {!analysis && !error && (
          <p className="text-foam-500 text-center">Loading analysis…</p>
        )}
      </div>
    </RequireAuth>
  );
}
