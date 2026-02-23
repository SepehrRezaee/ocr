from __future__ import annotations

import base64

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
}


def to_data_url(image_bytes: bytes, mime_type: str) -> str:
    normalized_mime = normalize_image_mime_type(mime_type)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{normalized_mime};base64,{encoded}"


def normalize_image_mime_type(content_type: str) -> str:
    normalized = content_type.strip().lower()
    return _MIME_ALIASES.get(normalized, normalized)
