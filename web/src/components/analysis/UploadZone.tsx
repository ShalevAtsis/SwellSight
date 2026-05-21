"use client";

import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function UploadZone({ onFile, disabled }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      setPreview(URL.createObjectURL(file));
      onFile(file);
    },
    [onFile],
  );

  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed p-8 text-center transition ${
        dragOver
          ? "border-swell-400 bg-swell-500/10"
          : "border-ocean-700 bg-ocean-900/30"
      } ${disabled ? "opacity-50 pointer-events-none" : "cursor-pointer hover:border-swell-500/60"}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f) handleFile(f);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      aria-label="Upload beach cam photo"
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        capture="environment"
        className="sr-only"
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={preview}
          alt="Preview"
          className="mx-auto max-h-48 rounded-lg object-contain"
        />
      ) : (
        <>
          <p className="text-foam-200 font-medium">Drop a beach cam photo</p>
          <p className="mt-1 text-sm text-foam-500">or tap to choose / use camera</p>
          <p className="mt-3 text-xs text-foam-600">JPEG, PNG, WebP · max 10 MB</p>
        </>
      )}
    </div>
  );
}
