from __future__ import annotations

import argparse
import logging

from app.config import get_settings
from app.logging_utils import configure_logging
from app.model_store import ensure_model_store, resolve_repo_dir

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and cache the OCR model into local model_store."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local artifacts already exist.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.effective_log_level)

    bootstrap_settings = settings.model_copy(
        update={
            "require_local_model_store": True,
            "auto_download_model_store": True,
            "model_force_download": args.force,
        }
    )
    ensure_model_store(bootstrap_settings)
    repo_dir = resolve_repo_dir(bootstrap_settings)
    logger.info(
        "Model bootstrap completed",
        extra={
            "model_name": bootstrap_settings.model_name,
            "model_repo_id": bootstrap_settings.model_repo_id,
            "model_store_dir": bootstrap_settings.model_store_dir,
        },
    )
    print(repo_dir)


if __name__ == "__main__":
    main()
