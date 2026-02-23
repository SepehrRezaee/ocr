from __future__ import annotations

import builtins
import runpy
from typing import Any


def _is_blocked_module(module_name: str) -> bool:
    return module_name == "flash_attn" or module_name.startswith("flash_attn.")


def _install_flash_attn_block() -> None:
    """
    Block flash_attn imports during vLLM startup.

    This prevents startup crashes when a broken flash-attn wheel is present
    but incompatible with the local torch ABI.
    """
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[Any, ...] | list[Any] = (),
        level: int = 0,
    ) -> Any:
        if _is_blocked_module(name):
            raise ModuleNotFoundError(f"Blocked optional dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def main() -> None:
    _install_flash_attn_block()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
