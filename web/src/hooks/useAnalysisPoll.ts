"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "@/lib/api";
import type { Analysis } from "@/lib/types";

const TERMINAL = new Set(["completed", "failed"]);

export function useAnalysisPoll(token: string | null, analysisId: string | null) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPolling(false);
  }, []);

  const fetchOnce = useCallback(async () => {
    if (!token || !analysisId) return;
    try {
      const data = await api.getAnalysis(token, analysisId);
      setAnalysis(data);
      setError(null);
      if (TERMINAL.has(data.status)) stop();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load analysis");
      stop();
    }
  }, [token, analysisId, stop]);

  useEffect(() => {
    if (!token || !analysisId) return;
    setPolling(true);
    fetchOnce();
    intervalRef.current = setInterval(fetchOnce, 2000);
    return stop;
  }, [token, analysisId, fetchOnce, stop]);

  return { analysis, error, polling, refresh: fetchOnce };
}
