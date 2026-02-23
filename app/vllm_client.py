from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

TWO_BY_TWO_TRANSPARENT_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAC0lEQVR42mNgQAcAABIAAeRVjecAAAAASUVORK5CYII="
)


class VLLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        detail: str | None = None,
        backend_status_code: int | None = None,
        backend_error_class: str = "vllm_error",
        backend_latency_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.detail = detail
        self.backend_status_code = backend_status_code
        self.backend_error_class = backend_error_class
        self.backend_latency_ms = backend_latency_ms


class VLLMTimeoutError(VLLMError):
    pass


class VLLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        headers = {"Content-Type": "application/json"}
        if settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"

        self._client = httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            timeout=settings.vllm_timeout_seconds,
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def startup_check(self, check_vision: bool) -> None:
        await self._wait_until_ready()
        if check_vision:
            await self._verify_vision_path()

    async def run_ocr(self, image_data_url: str) -> str:
        payload = self._build_payload(
            prompt=self._settings.ocr_prompt,
            image_data_url=image_data_url,
            max_tokens=self._settings.max_tokens,
        )
        response = await self._chat_completion(payload)
        return _extract_message_content(response)

    async def _verify_vision_path(self) -> None:
        payload = self._build_payload(
            prompt="Reply with exactly OK.",
            image_data_url=TWO_BY_TWO_TRANSPARENT_PNG,
            max_tokens=8,
        )
        await self._chat_completion(payload)

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._settings.vllm_startup_timeout_seconds
        last_error: str | None = None

        while time.monotonic() < deadline:
            try:
                response = await self._client.get("/v1/models")
                if response.status_code == 200:
                    return
                last_error = (
                    f"vLLM readiness probe failed with HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            except Exception as exc:
                last_error = f"vLLM readiness probe error: {exc.__class__.__name__}: {exc}"

            await asyncio.sleep(1)

        raise VLLMTimeoutError(
            "Timed out waiting for local vLLM server readiness.",
            detail=last_error,
            backend_error_class="vllm_startup_timeout",
        )

    def _build_payload(self, *, prompt: str, image_data_url: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": self._settings.resolved_vllm_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": max_tokens,
            "top_k": self._settings.top_k,
        }

    async def _chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            raise VLLMTimeoutError(
                "vLLM request timed out.",
                detail=str(exc),
                backend_error_class=exc.__class__.__name__,
                backend_latency_ms=latency_ms,
            ) from exc
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            raise VLLMError(
                "vLLM request failed.",
                detail=str(exc),
                backend_error_class=exc.__class__.__name__,
                backend_latency_ms=latency_ms,
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            detail = _extract_error_message(response)
            raise VLLMError(
                "vLLM returned an inference error.",
                detail=detail,
                backend_status_code=response.status_code,
                backend_error_class="vllm_http_error",
                backend_latency_ms=latency_ms,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise VLLMError(
                "vLLM returned invalid JSON.",
                detail=str(exc),
                backend_status_code=response.status_code,
                backend_error_class=exc.__class__.__name__,
                backend_latency_ms=latency_ms,
            ) from exc

        return data


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return response.text[:500]


def _extract_message_content(result: dict[str, Any]) -> str:
    choices = result.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise VLLMError("vLLM response did not include any completion choices.")

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        combined = "\n".join(parts).strip()
        if combined:
            return combined

    text = first_choice.get("text") if isinstance(first_choice, dict) else None
    if isinstance(text, str):
        return text.strip()

    raise VLLMError("vLLM returned an unsupported completion content format.")
