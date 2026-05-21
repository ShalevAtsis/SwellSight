"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { ErrorAlert } from "@/components/ui/ErrorAlert";
import { useAuth } from "@/components/providers/AuthProvider";
import * as api from "@/lib/api";
import type { Analysis } from "@/lib/types";

function statusColor(status: string) {
  switch (status) {
    case "completed":
      return "text-swell-400";
    case "failed":
      return "text-red-400";
    case "processing":
      return "text-amber-300";
    default:
      return "text-foam-500";
  }
}

export default function HistoryPage() {
  const { token } = useAuth();
  const [items, setItems] = useState<Analysis[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) return;
    api
      .listAnalyses(token)
      .then(setItems)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, [token]);

  return (
    <RequireAuth>
      <div className="mx-auto max-w-2xl px-4 py-10">
        <div className="flex items-center justify-between gap-4">
          <h1 className="font-display text-2xl font-bold text-foam-50">History</h1>
          <Link
            href="/analyze"
            className="rounded-lg bg-swell-500 px-4 py-2 text-sm font-semibold text-ocean-950 hover:bg-swell-400"
          >
            New analysis
          </Link>
        </div>

        {error && <div className="mt-4"><ErrorAlert message={error} /></div>}

        {loading ? (
          <p className="mt-8 text-foam-500">Loading…</p>
        ) : items.length === 0 ? (
          <p className="mt-8 text-foam-500">
            No analyses yet.{" "}
            <Link href="/analyze" className="text-swell-400 hover:underline">
              Upload your first photo
            </Link>
          </p>
        ) : (
          <ul className="mt-8 space-y-3">
            {items.map((a) => (
              <li key={a.id}>
                <Link
                  href={`/analyze/${a.id}`}
                  className="flex items-center justify-between rounded-xl border border-ocean-800 bg-ocean-900/40 px-4 py-3 hover:border-swell-500/40 transition"
                >
                  <div>
                    <p className="font-mono text-xs text-foam-600 truncate max-w-[200px] sm:max-w-none">
                      {a.id}
                    </p>
                    <p className="text-sm text-foam-400">
                      {a.created_at
                        ? new Date(a.created_at).toLocaleString()
                        : "—"}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className={`text-sm capitalize ${statusColor(a.status)}`}>
                      {a.status}
                    </span>
                    {a.surf_score != null && (
                      <p className="font-display text-lg font-bold text-foam-100">
                        {Math.round(a.surf_score)}
                      </p>
                    )}
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </RequireAuth>
  );
}
