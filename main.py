from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.errors import APIError
from app.image_processing import ALLOWED_CONTENT_TYPES, normalize_image_mime_type, to_data_url
from app.logging_utils import configure_logging
from app.model_store import ensure_model_store
from app.schemas import ErrorResponse, OCRResponse
from app.vllm_client import VLLMClient, VLLMError, VLLMTimeoutError

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.effective_log_level)
    model_store_path = ensure_model_store(settings)
    logger.info(
        "Starting OCR API",
        extra={
            "model_name": settings.model_name,
            "model_repo_id": settings.model_repo_id,
            "model_store_dir": model_store_path,
            "model_filename": settings.model_filename,
        },
    )

    vllm_client = VLLMClient(settings)
    try:
        await vllm_client.startup_check(check_vision=settings.startup_compat_check)
    except Exception as exc:
        await vllm_client.close()
        startup_error_detail = exc.detail if isinstance(exc, VLLMError) else None
        logger.exception(
            "Startup validation failed",
            extra={
                "model_name": settings.model_name,
                "backend": "vllm",
                "model_repo_id": settings.model_repo_id,
                "model_filename": settings.model_filename,
                "startup_error_detail": startup_error_detail,
            },
        )
        raise RuntimeError(
            "vLLM backend startup check failed. "
            "Ensure local vLLM server is running and model assets are available in OCR_MODEL_STORE_DIR."
        ) from exc

    app.state.settings = settings
    app.state.vllm_client = vllm_client
    logger.info(
        "vLLM backend ready",
        extra={
            "model_name": settings.model_name,
            "backend": "vllm",
            "model_repo_id": settings.model_repo_id,
            "model_filename": settings.model_filename,
            "model_store_dir": model_store_path,
        },
    )

    try:
        yield
    finally:
        logger.info("Stopping OCR API", extra={"model_name": settings.model_name})
        await app.state.vllm_client.close()


app = FastAPI(
    title=settings.model_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    model_name = getattr(getattr(request.app.state, "settings", None), "model_name", settings.model_name)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "Request failed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
                "model_name": model_name,
            },
        )
        raise

    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["x-request-id"] = request_id
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else None,
            "model_name": model_name,
        },
    )
    return response


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/ocr",
    response_model=OCRResponse,
    responses={
        400: {"model": ErrorResponse},
        408: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ocr(request: Request, file: UploadFile = File(...)) -> OCRResponse:
    runtime_settings: Settings = request.app.state.settings
    vllm_client: VLLMClient = request.app.state.vllm_client
    request_id: str = request.state.request_id
    started = time.perf_counter()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise APIError(
            status_code=415,
            error_code="unsupported_media_type",
            message=f"Unsupported file type: {file.content_type}",
        )

    payload = await file.read()
    await file.close()

    if not payload:
        raise APIError(status_code=400, error_code="empty_file", message="Uploaded file is empty.")

    if len(payload) > runtime_settings.max_upload_bytes:
        raise APIError(
            status_code=413,
            error_code="file_too_large",
            message=f"File exceeds {runtime_settings.max_upload_megabytes}MB limit.",
        )

    logger.info(
        "OCR request accepted",
        extra={
            "request_id": request_id,
            "file_name": file.filename,
            "file_size": len(payload),
            "path": request.url.path,
            "method": request.method,
            "model_name": runtime_settings.model_name,
            "model_repo_id": runtime_settings.model_repo_id,
        },
    )

    image_data_url = to_data_url(
        image_bytes=payload,
        mime_type=normalize_image_mime_type(file.content_type or "image/jpeg"),
    )

    try:
        markdown = await asyncio.wait_for(
            vllm_client.run_ocr(image_data_url),
            timeout=runtime_settings.inference_timeout_seconds,
        )
    except (asyncio.TimeoutError, VLLMTimeoutError) as exc:
        logger.exception(
            "vLLM inference timeout",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "backend": "vllm",
                "backend_error_class": exc.__class__.__name__,
                "model_name": runtime_settings.model_name,
                "model_repo_id": runtime_settings.model_repo_id,
            },
        )
        raise APIError(
            status_code=408,
            error_code="inference_timeout",
            message="Inference timed out.",
        ) from exc
    except VLLMError as exc:
        logger.exception(
            "vLLM inference failure",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "backend": "vllm",
                "backend_status_code": exc.backend_status_code,
                "backend_latency_ms": exc.backend_latency_ms,
                "backend_error_class": exc.backend_error_class,
                "backend_error_detail": exc.detail,
                "model_name": runtime_settings.model_name,
                "model_repo_id": runtime_settings.model_repo_id,
            },
        )
        raise APIError(
            status_code=503,
            error_code="inference_failure",
            message="Model inference failed.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected inference failure",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "backend": "vllm",
                "backend_error_class": exc.__class__.__name__,
                "model_name": runtime_settings.model_name,
                "model_repo_id": runtime_settings.model_repo_id,
            },
        )
        raise APIError(
            status_code=503,
            error_code="inference_failure",
            message="Model inference failed.",
        ) from exc

    markdown_output = markdown.strip()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return OCRResponse(
        request_id=request_id,
        model=runtime_settings.model_name,
        markdown=markdown_output,
        processing_ms=elapsed_ms,
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    model_name = getattr(getattr(request.app.state, "settings", None), "model_name", settings.model_name)
    logger.error(
        "API error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "model_name": model_name,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            request_id=request_id,
            error_code=exc.error_code,
            message=exc.message,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    model_name = getattr(getattr(request.app.state, "settings", None), "model_name", settings.model_name)
    logger.error(
        "Validation error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": 422,
            "error_code": "validation_error",
            "model_name": model_name,
        },
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            request_id=request_id,
            error_code="validation_error",
            message=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    model_name = getattr(getattr(request.app.state, "settings", None), "model_name", settings.model_name)
    logger.exception(
        "Unhandled server error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": 500,
            "error_code": "internal_server_error",
            "model_name": model_name,
        },
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            request_id=request_id,
            error_code="internal_server_error",
            message="Unexpected server error.",
        ).model_dump(),
    )
