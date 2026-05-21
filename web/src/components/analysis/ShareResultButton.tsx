"use client";

import { useState } from "react";

export function ShareResultButton({ analysisId }: { analysisId: string }) {
  const [copied, setCopied] = useState(false);

  async function copyLink() {
    const url =
      typeof window !== "undefined"
        ? `${window.location.origin}/analyze/${analysisId}`
        : `/analyze/${analysisId}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button
      type="button"
      onClick={copyLink}
      className="rounded-lg border border-ocean-700 px-4 py-2 text-sm text-foam-200 hover:bg-ocean-900 transition"
    >
      {copied ? "Link copied" : "Copy result link"}
    </button>
  );
}
