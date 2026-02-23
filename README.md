# sharifsetup-OCR

Production OCR API for Persian, Hebrew, Arabic, and English using FastAPI + local vLLM inference.
Swagger/OpenAPI title and runtime model display name are fixed to `sharifsetup-OCR`.

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Configure

```bash
cp .env.example .env
```

Set at least:
- `OCR_MODEL_STORE_DIR`: local model directory (default `./model_store`)
- `OCR_MODEL_REPO_ID`: Hugging Face model repo id (default `allenai/olmOCR-2-7B-1025-FP8`)

## 3) Runtime Behavior

- On first run, app startup downloads model artifacts into `OCR_MODEL_STORE_DIR`.
- OCR inference is sent to local vLLM at `OCR_VLLM_BASE_URL` using `/v1/chat/completions`.
- Uploaded image bytes are passed to the model as-is (no preprocessing).

## 4) Run (Two Processes)

Terminal A:

```bash
PYTHONPATH=./ python3 -m app.vllm_local_server
```

Terminal B:

```bash
PYTHONPATH=./ uvicorn main:app --host 0.0.0.0 --port 8000
```

## 5) OCR Request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ocr" \
  -H "accept: application/json" \
  -F "file=@/absolute/path/to/image.png"
```

Success response:

```json
{
  "request_id": "4d483dc5-79db-46c1-9d59-8f8668a75f15",
  "model": "sharifsetup-OCR",
  "markdown": "extracted markdown text",
  "processing_ms": 1423
}
```

## 6) Docker (Single Dockerfile)

The single container starts local vLLM and FastAPI together.
Model artifacts under local `./model_store` are baked into the image at build time.

```bash
docker build -t sharifsetup-ocr .
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 \
  sharifsetup-ocr
```

Important:
- Do not mount an empty volume to `/srv/model_store` when using baked model artifacts, because it hides the bundled model files.

## 7) Troubleshooting

- Use the correct env var name: `PYTHONPATH` (not `PTYHONPATH`).
- If startup fails on model download, verify `OCR_HF_TOKEN` or `HF_TOKEN`.
- If FastAPI starts but readiness fails, check local vLLM endpoint: `http://127.0.0.1:8001/v1/models`.
