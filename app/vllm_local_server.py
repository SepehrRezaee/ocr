from __future__ import annotations

import os
import shlex
import sys

from app.config import get_settings
from app.logging_utils import configure_logging
from app.model_store import ensure_model_store, resolve_repo_dir


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    ensure_model_store(settings)
    model_path = resolve_repo_dir(settings)
    if not model_path.exists() or not model_path.is_dir():
        raise RuntimeError(
            f"Local model directory does not exist for vLLM: '{model_path}'."
        )

    cmd = [
        sys.executable,
        "-m",
        "app.vllm_no_flash_entrypoint",
        "--host",
        settings.vllm_host,
        "--port",
        str(settings.vllm_port),
        "--model",
        str(model_path),
        "--served-model-name",
        settings.resolved_vllm_model_id,
        "--dtype",
        settings.vllm_dtype,
        "--max-model-len",
        str(settings.vllm_max_model_len),
        "--tensor-parallel-size",
        str(settings.vllm_tensor_parallel_size),
        "--gpu-memory-utilization",
        str(settings.vllm_gpu_memory_utilization),
    ]

    if settings.vllm_trust_remote_code:
        cmd.append("--trust-remote-code")
    if settings.vllm_enforce_eager:
        cmd.append("--enforce-eager")
    if settings.vllm_disable_mm_preprocessor_cache:
        cmd.append("--disable-mm-preprocessor-cache")
    if settings.vllm_additional_args:
        cmd.extend(shlex.split(settings.vllm_additional_args))

    env = os.environ.copy()
    if settings.hf_token:
        env.setdefault("HF_TOKEN", settings.hf_token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)

    os.execvpe(sys.executable, cmd, env)


if __name__ == "__main__":
    main()
