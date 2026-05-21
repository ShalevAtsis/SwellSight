"use client";

import { useEffect, useState } from "react";
import { analysisImageUrl } from "@/lib/api";

interface AnalysisThumbnailProps {
  analysisId: string;
  token: string;
  alt?: string;
}

export function AnalysisThumbnail({
  analysisId,
  token,
  alt = "Beach cam thumbnail",
}: AnalysisThumbnailProps) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let objectUrl: string | null = null;
    fetch(analysisImageUrl(analysisId), {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => (res.ok ? res.blob() : null))
      .then((blob) => {
        if (blob) {
          objectUrl = URL.createObjectURL(blob);
          setSrc(objectUrl);
        }
      })
      .catch(() => setSrc(null));
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [analysisId, token]);

  if (!src) {
    return (
      <div
        className="h-14 w-14 rounded-lg bg-ocean-800 shrink-0"
        aria-hidden
      />
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt}
      className="h-14 w-14 rounded-lg object-cover bg-ocean-800 shrink-0"
    />
  );
}
