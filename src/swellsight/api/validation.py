"""
Upload validation for beach cam images (P3-T10).
"""

from __future__ import annotations

import io
from typing import Tuple

from fastapi import HTTPException

ALLOWED_MIME = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # WEBP has RIFF....WEBP
}


def detect_image_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image_upload(
    content: bytes,
    declared_mime: str | None,
    max_bytes: int,
    max_dimension: int,
) -> Tuple[str, int, int]:
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {max_bytes // (1024*1024)}MB)",
        )

    detected = detect_image_type(content)
    if not detected:
        raise HTTPException(status_code=400, detail="Unrecognized image format (JPEG/PNG/WebP only)")

    if declared_mime and declared_mime not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {declared_mime}")

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
        width, height = img.size
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file") from exc

    if width < 64 or height < 64:
        raise HTTPException(status_code=400, detail="Image too small (minimum 64x64)")

    if width > max_dimension or height > max_dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions exceed maximum {max_dimension}px",
        )

    return detected, width, height
