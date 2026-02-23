from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Formats logs as compact JSON records for downstream ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "path",
            "method",
            "status_code",
            "duration_ms",
            "error_code",
            "client_ip",
            "file_name",
            "file_size",
            "model_path",
            "model_name",
            "model_repo_id",
            "model_filename",
            "model_store_dir",
            "configured_device",
            "applied_device",
            "configured_attn_impl",
            "applied_attn_impl",
            "configured_top_k",
            "configured_top_p",
            "applied_top_k",
            "applied_top_p",
            "backend",
            "backend_status_code",
            "backend_latency_ms",
            "backend_error_class",
            "backend_error_detail",
            "startup_error_detail",
            "retry_attempt",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
