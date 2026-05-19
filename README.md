# GPU-Parallelized Prompt Sensitivity Mapper

Local MVP for mapping how small prompt changes shift LLM outputs.

## What it does

- Generates rule-based prompt perturbations across lexical, syntactic, persona, politeness, specificity, negation, and position categories.
- Runs batched inference with Gemini when `GEMINI_API_KEY` is available.
- Falls back to deterministic synthetic outputs so the demo works without API access.
- Uses deterministic hashed embeddings by default so demos are fully offline and fast.
- Can embed outputs with `all-MiniLM-L6-v2` on CUDA when `ALLOW_EMBEDDING_MODEL_DOWNLOAD=true`.
- Projects responses to 2D, clusters them, and serves an interactive robustness map.
- Uses fast PCA/KMeans by default; set `ENABLE_UMAP_HDBSCAN=true` to use UMAP and HDBSCAN.
- Trains a small PyTorch risk predictor from saved runs to classify high-shift prompt variations.

## Quick start

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run build
npm run preview -- --host 127.0.0.1 --port 5173
```

Open http://127.0.0.1:5173.

## Gemini live runs

Set `GEMINI_API_KEY` in `.env` or your shell. The default model is:

```text
gemini-2.5-flash
```

Use the UI mode selector to choose demo, mock, or live. Demo and mock modes do not call external APIs.

Example `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

For full MiniLM embeddings, allow the model load/download:

```powershell
$env:ALLOW_EMBEDDING_MODEL_DOWNLOAD="true"
$env:ENABLE_UMAP_HDBSCAN="true"
```

## API

- `POST /api/runs` starts a run.
- `GET /api/runs/{run_id}` returns status and results.
- `GET /api/runs/{run_id}/export` downloads the run JSON.
- `POST /api/training/risk-model` trains the prompt risk predictor.
- `GET /api/training/risk-model` returns training status, GPU/device, loss, and accuracy.
- `GET /api/health` reports CUDA and dependency status.

## Notes

- KL divergence is intentionally marked unavailable in v1 because this app does not receive token logits from the live API.
- GPU acceleration is shown through embedding device/throughput and PyTorch risk-model training when CUDA is available.
- Results are stored under `backend/data/runs`.
- Trained model artifacts are stored under `backend/data/models`.
