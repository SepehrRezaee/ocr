from __future__ import annotations

import inspect
import logging
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


def ensure_model_store(settings: Settings) -> str | None:
    """Ensures model artifacts exist locally and honors offline settings."""
    should_bootstrap = (
        settings.require_local_model_store
        or settings.auto_download_model_store
        or settings.model_force_download
    )
    if not should_bootstrap:
        return None

    model_store_dir = Path(settings.model_store_dir).expanduser().resolve()
    model_store_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = resolve_repo_dir(settings)
    has_expected = _has_expected_artifacts(repo_dir, settings.model_filename)
    if has_expected and not settings.model_force_download:
        _write_bootstrap_marker(model_store_dir)
        return str(model_store_dir)

    should_download = settings.model_force_download or settings.auto_download_model_store
    if not should_download:
        if settings.require_local_model_store:
            raise RuntimeError(
                f"Model artifacts are required but missing in '{repo_dir}'. "
                "Set OCR_AUTO_DOWNLOAD_MODEL_STORE=true for initial bootstrap, "
                "or download the model to OCR_MODEL_STORE_DIR before startup."
            )
        return None

    _download_model_snapshot(settings=settings, model_store_dir=model_store_dir)
    if not _has_expected_artifacts(repo_dir, settings.model_filename):
        raise RuntimeError(
            f"Model bootstrap finished but expected artifacts were not found under '{repo_dir}'."
        )
    _write_bootstrap_marker(model_store_dir)

    return str(model_store_dir)


def validate_model_store(
    model_store_dir: str | Path | Settings | None = None,
    *,
    path: str | Path | None = None,
    require_non_empty: bool | None = None,
    **legacy_kwargs: object,
) -> str:
    """
    Backward-compatible model store validator used by older startup code.

    This delegates to ensure_model_store so first-run bootstrap download
    still happens even if callers import validate_model_store.
    """
    if "settings" in legacy_kwargs and model_store_dir is None:
        model_store_dir = legacy_kwargs.pop("settings")  # type: ignore[assignment]
    if "model_store_path" in legacy_kwargs and path is None:
        maybe_path = legacy_kwargs.pop("model_store_path")
        if isinstance(maybe_path, (str, Path)):
            path = maybe_path

    allow_empty = legacy_kwargs.pop("allow_empty", None)
    if require_non_empty is None:
        require_non_empty = not bool(allow_empty) if allow_empty is not None else True

    configured_path: str | Path | None = path
    if configured_path is None and not isinstance(model_store_dir, Settings):
        configured_path = model_store_dir

    if isinstance(model_store_dir, Settings):
        settings = model_store_dir
    else:
        from app.config import get_settings

        settings = get_settings()

    if configured_path is not None:
        settings = settings.model_copy(
            update={"model_store_dir": str(Path(configured_path).expanduser())}
        )

    if legacy_kwargs:
        logger.debug(
            "Ignoring unrecognized validate_model_store kwargs",
            extra={"ignored_kwargs": sorted(legacy_kwargs.keys())},
        )

    ensured_path = ensure_model_store(settings)
    resolved = Path(settings.model_store_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)

    if ensured_path is not None:
        return ensured_path

    if require_non_empty and not _model_store_has_files(resolved):
        raise RuntimeError(
            f"Model store directory '{resolved}' is empty. "
            "Enable OCR_AUTO_DOWNLOAD_MODEL_STORE=true or place model artifacts before startup."
        )

    _write_bootstrap_marker(resolved)
    return str(resolved)


def resolve_repo_dir(settings: Settings) -> Path:
    model_store_dir = Path(settings.model_store_dir).expanduser().resolve()
    return model_store_dir / settings.model_local_dir_name


def _download_model_snapshot(settings: Settings, model_store_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required for first-run model download. "
            "Install dependencies from requirements.txt."
        ) from exc

    model_store_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = resolve_repo_dir(settings)
    repo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading model snapshot into model_store",
        extra={
            "model_name": settings.model_name,
            "model_repo_id": settings.model_repo_id,
            "model_store_dir": str(model_store_dir),
        },
    )

    kwargs: dict[str, object] = {
        "repo_id": settings.model_repo_id,
        "local_dir": str(repo_dir),
        "token": settings.hf_token,
        "force_download": settings.model_force_download,
    }
    if settings.model_filename:
        kwargs["allow_patterns"] = [settings.model_filename]

    params = inspect.signature(snapshot_download).parameters
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in params and value is not None
    }
    try:
        snapshot_download(**filtered_kwargs)
    except Exception as exc:  # pragma: no cover - network/backend errors
        raise RuntimeError(
            f"Failed to download model '{settings.model_repo_id}' into '{repo_dir}'. "
            "Check Hugging Face connectivity and OCR_HF_TOKEN/HF_TOKEN for gated models."
        ) from exc

    logger.info(
        "Model snapshot ready",
        extra={
            "model_name": settings.model_name,
            "model_repo_id": settings.model_repo_id,
            "model_store_dir": str(model_store_dir),
        },
    )


def _has_expected_artifacts(repo_dir: Path, model_filename: str | None) -> bool:
    if not repo_dir.exists() or not repo_dir.is_dir():
        return False

    if model_filename:
        return _find_model_file(repo_dir, model_filename) is not None

    return any(candidate.is_file() for candidate in repo_dir.rglob("*"))


def _find_model_file(repo_dir: Path, filename_pattern: str) -> Path | None:
    if not repo_dir.exists() or not repo_dir.is_dir():
        return None

    normalized = filename_pattern.strip()
    if not normalized:
        return None

    wildcard_chars = {"*", "?", "["}
    if not any(char in normalized for char in wildcard_chars):
        direct = (repo_dir / normalized).resolve()
        if direct.exists() and direct.is_file():
            return direct

    matches = sorted(path for path in repo_dir.rglob(normalized) if path.is_file())
    if matches:
        return matches[0]
    return None


def _model_store_has_files(model_store_dir: Path) -> bool:
    if not model_store_dir.exists() or not model_store_dir.is_dir():
        return False
    return any(candidate.is_file() for candidate in model_store_dir.rglob("*"))


def _write_bootstrap_marker(model_store_dir: Path) -> None:
    marker = model_store_dir / ".model_store_ready"
    try:
        marker.touch(exist_ok=True)
    except OSError:
        logger.warning(
            "Failed to update model_store bootstrap marker",
            extra={"model_store_dir": str(model_store_dir)},
        )
