"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { UploadZone } from "@/components/analysis/UploadZone";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { useAuth } from "@/components/providers/AuthProvider";
import * as api from "@/lib/api";
import { ApiClientError } from "@/lib/api";

export default function AnalyzePage() {
  const { token } = useAuth();
  const router = useRouter();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!token) return;
      setUploading(true);
      setError(null);
      try {
        const analysis = await api.createAnalysis(token, file);
        router.push(`/analyze/${analysis.id}`);
      } catch (e) {
        const msg =
          e instanceof ApiClientError
            ? e.status === 429
              ? "Daily limit reached. Try again tomorrow."
              : e.message
            : e instanceof Error
              ? e.message
              : "Upload failed";
        setError(msg);
      } finally {
        setUploading(false);
      }
    },
    [token, router],
  );

  return (
    <RequireAuth>
      <div className="mx-auto max-w-lg px-4 py-10">
        <h1 className="font-display text-2xl font-bold text-foam-50">Analyze surf</h1>
        <p className="mt-2 text-sm text-foam-500">
          Upload a beach cam frame. We&apos;ll queue AI analysis and show results when ready.
        </p>
        <div className="mt-8 space-y-4">
          {error && <ErrorAlert message={error} />}
          <UploadZone onFile={handleFile} disabled={uploading} />
          {uploading && (
            <p className="text-center text-sm text-swell-400" aria-live="polite">
              Uploading…
            </p>
          )}
        </div>
      </div>
    </RequireAuth>
  );
}
