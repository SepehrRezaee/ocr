from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_STORE_DIR = PROJECT_ROOT / "model_store"
DISPLAY_MODEL_NAME = "sharifsetup-OCR"

GEMMA_3_4B_OCR_PROMPT = """<system_directive>
You are an expert, zero-shot multimodal extraction engine. Your task is to convert the provided physical documents, handwritten notes, digital desktop interfaces, and physical signage into structured Markdown. 

⚠️ **CRITICAL constraints:** - You process all languages natively.
- You must preserve inherent scripts, layouts, and formatting.
- You do NOT chat, explain, or interact. Output ONLY the extracted text.
</system_directive>

<extraction_rules>
1. **TRANSCRIPTION & FIDELITY**
   - Extract text exactly as written. Do not translate, interpret, or correct grammar/typos.
   - **Handwriting & Marginalia:** Process printed text and handwritten text (including highly stylized cursive, diagonal annotations, and margin notes) with equal precision. 
   - Preserve exact numeral systems, punctuation, and special characters.

2. **LAYOUT, ALIGNMENT & TABLES**
   - Use Markdown strictly to replicate structural hierarchy.
   - **Key-Value & Tabular Alignment:** Use Markdown tables (`|...|`) for explicit grid tables AND visually implied tabular data (e.g., menus, receipts, aligned forms) to strictly preserve horizontal relationships.
   - **Disjointed Elements:** Group separate physical signs, discrete UI panels, or distinct physical documents using a double line break (`\n\n`).

3. **SPATIAL HIERARCHY & MULTI-DOC BOUNDARIES**
   - **Native Directionality:** Respect the primary reading direction of the document. For multi-column layouts, process in the natural reading order of the primary language.
   - **Bidirectional Integrity:** When contrasting text directions (RTL and LTR) share a line, maintain the internal sequence of each string without scrambling word order.
   - **Overlapping UI & Z-Axis:** For desktop screenshots, transcribe the active/foreground window completely before moving to background windows or system chrome (taskbars, docks). Maintain UI nesting (App -> Tab -> Content).
   - **Physical Boundaries:** If multiple physical pages/documents appear side-by-side, transcribe them sequentially as entirely separate logical blocks. Do not read horizontally across physical gaps.

4. **NON-TEXT ELEMENTS, UI & SYMBOLS**
   - **Digital Interfaces:** Represent interactive elements using standard Markdown: `- [ ]` (empty checkbox), `- [x]` (checked checkbox). Place them immediately adjacent to their corresponding text labels. Use `[Button: Name]` or `[Tab: Name]` for UI chrome.
   - **Navigational Symbols:** Transcribe meaningful navigational markers (e.g., highway arrows) using appropriate Unicode (e.g., ⬆️, ⬅️, ↗️). Ignore decorative UI arrows (e.g., browser refresh buttons).
   - **Entity Tagging:** Output exact bracketed tags for visual markers: `[signature]`, `[stamp]`, `[logo]`, `[CAPTCHA]`.

5. **NOISE & DEGRADATION THRESHOLDS**
   - **Noise Rejection:** Silently ignore decorative backgrounds, faint overlays, and watermarks that do not convey primary information.
   - **Censorship:** Output `[redacted]` for intentionally obscured, blocked, or pixelated text. 
   - **Illegibility Threshold:** Output `[unreadable]` ONLY if physically degraded, cut-off, or heavily stylized cursive text is completely indecipherable. Do not hallucinate guesses.
</extraction_rules>

<output_formatting>
⚠️ **CRITICAL:** Provide the raw markdown text ONLY. 
- Do NOT wrap the output in markdown code blocks (e.g., ```markdown). 
- Do NOT include greetings, explanations, or conversational filler. 
- Begin immediately with the first extracted character.
</output_formatting>"""

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="OCR_",
        extra="ignore",
    )

    app_name: str = DISPLAY_MODEL_NAME
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    model_name: str = DISPLAY_MODEL_NAME
    model_store_dir: str = str(DEFAULT_MODEL_STORE_DIR)
    require_local_model_store: bool = True
    auto_download_model_store: bool = True
    model_force_download: bool = False
    model_repo_id: str = "allenai/olmOCR-2-7B-1025-FP8"
    model_filename: str | None = None
    hf_token: str | None = None

    startup_compat_check: bool = True
    vllm_base_url: str = "http://127.0.0.1:8001"
    vllm_api_key: str = "EMPTY"
    vllm_model_id: str | None = None
    vllm_timeout_seconds: int = 120
    vllm_startup_timeout_seconds: int = 600

    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8001
    vllm_dtype: str = "bfloat16"
    vllm_max_model_len: int = 8192
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.90
    vllm_trust_remote_code: bool = False
    vllm_enforce_eager: bool = False
    vllm_disable_mm_preprocessor_cache: bool = False
    vllm_additional_args: str | None = None

    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    max_tokens: int = 4096
    inference_timeout_seconds: int = 90

    max_upload_megabytes: int = 15

    ocr_prompt: str = Field(default=GEMMA_3_4B_OCR_PROMPT)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        level = value.upper().strip()
        valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return level

    @field_validator("app_name", "model_name")
    @classmethod
    def enforce_display_name(cls, value: str) -> str:
        if value != DISPLAY_MODEL_NAME:
            raise ValueError(f"Display name must be '{DISPLAY_MODEL_NAME}'.")
        return value

    @field_validator("model_store_dir")
    @classmethod
    def expand_model_store_dir(cls, value: str) -> str:
        return str(Path(value).expanduser())

    @field_validator("model_filename")
    @classmethod
    def normalize_model_filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized

    @field_validator("vllm_base_url")
    @classmethod
    def normalize_vllm_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("vllm base_url must not be empty")
        return normalized

    @field_validator("vllm_model_id")
    @classmethod
    def normalize_vllm_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("vllm_port")
    @classmethod
    def validate_vllm_port(cls, value: int) -> int:
        if value < 1 or value > 65535:
            raise ValueError("vllm_port must be between 1 and 65535")
        return value

    @field_validator("vllm_dtype")
    @classmethod
    def validate_vllm_dtype(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"auto", "half", "float16", "bfloat16", "float", "float32"}
        if normalized not in valid:
            raise ValueError(f"vllm_dtype must be one of: {', '.join(sorted(valid))}")
        return normalized

    @field_validator("vllm_tensor_parallel_size")
    @classmethod
    def validate_vllm_tensor_parallel_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("vllm_tensor_parallel_size must be >= 1")
        return value

    @field_validator("vllm_gpu_memory_utilization")
    @classmethod
    def validate_vllm_gpu_memory_utilization(cls, value: float) -> float:
        if value <= 0.0 or value > 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be > 0 and <= 1")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k must be >= 1")
        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float) -> float:
        if value <= 0.0 or value > 1.0:
            raise ValueError("top_p must be > 0 and <= 1")
        return value

    @field_validator("hf_token")
    @classmethod
    def normalize_hf_token(cls, value: str | None) -> str | None:
        if value and value.strip():
            return value.strip()
        fallback = os.getenv("HF_TOKEN")
        if fallback and fallback.strip():
            return fallback.strip()
        return None

    @property
    def resolved_vllm_model_id(self) -> str:
        return self.vllm_model_id or self.model_repo_id

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_megabytes * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
