# Sharifsetup OCR

Production OCR API using FastAPI + local vLLM with offline-ready Docker packaging.
The display name is fixed everywhere to `Sharifsetup-OCR`.

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

## 3) One-Time Model Download (Project Local)

Download `allenai/olmOCR-2-7B-1025-FP8` into `./model_store`:

```bash
PYTHONPATH=./ python3 -m app.bootstrap_model_store
```

After download, model files are stored under:
`./model_store/sharifsetup-ocr`

## 4) Run Locally

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
  "model": "Sharifsetup-OCR",
  "markdown": "extracted markdown text",
  "processing_ms": 1423
}
```

## 6) Docker Offline Build/Run

The Docker image copies local `./model_store` into `/srv/model_store`, so the container can run offline.

```bash
docker build -t sharifsetup-ocr .
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 \
  sharifsetup-ocr
```

Or with Compose:

```bash
docker compose up --build
```

Important:
- Run step 3 before building the image.
- Do not mount an empty volume to `/srv/model_store` because it hides bundled model files.

## 7) Logging Flag

- `OCR_VERBOSE_LOGS=true` -> runtime log level is `DEBUG` (all levels visible).
- `OCR_VERBOSE_LOGS=false` -> runtime log level is `ERROR` only.
