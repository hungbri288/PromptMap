import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.analysis import analyze_outputs, dependency_status
from backend.app.config import get_settings
from backend.app.inference import run_inference
from backend.app.perturbations import generate_variants
from backend.app.schemas import RunRecord, RunRequest, TrainingRecord
from backend.app.storage import RunStore
from backend.app.training import idle_training_record, train_risk_predictor

app = FastAPI(title="Prompt Sensitivity Mapper", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_state: TrainingRecord | None = None


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "default_model": settings.gemini_model,
        "live_provider": "gemini",
        "live_ready": bool(settings.gemini_api_key),
        "dependencies": dependency_status(),
    }


@app.post("/api/runs", response_model=RunRecord)
async def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunRecord:
    settings = get_settings()
    store = RunStore(settings.run_storage_dir)
    run_id = uuid4().hex[:12]
    record = RunRecord(
        id=run_id,
        status="queued",
        created_at=datetime.now(timezone.utc),
        request=request,
    )
    store.save(record)
    if request.mode == "live" and not settings.gemini_api_key:
        record.status = "failed"
        record.error = "Gemini live mode requires GEMINI_API_KEY."
        record.completed_at = datetime.now(timezone.utc)
        store.save(record)
        return store.load(run_id)
    if request.mode == "demo":
        await _execute_run(run_id)
    else:
        background_tasks.add_task(_execute_run, run_id)
    return store.load(run_id)


@app.get("/api/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    settings = get_settings()
    store = RunStore(settings.run_storage_dir)
    try:
        return store.load(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str) -> JSONResponse:
    record = get_run(run_id)
    return JSONResponse(
        content=record.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="prompt-sensitivity-{run_id}.json"'},
    )


@app.get("/api/training/risk-model", response_model=TrainingRecord)
def get_training_status() -> TrainingRecord:
    global training_state
    settings = get_settings()
    if training_state is not None:
        return training_state
    training_state = idle_training_record(settings.model_storage_dir)
    return training_state


@app.post("/api/training/risk-model", response_model=TrainingRecord)
async def start_training() -> TrainingRecord:
    global training_state
    settings = get_settings()
    training_state = TrainingRecord(status="running")
    training_state = await asyncio.to_thread(
        train_risk_predictor,
        settings.run_storage_dir,
        settings.model_storage_dir,
    )
    return training_state


async def _execute_run(run_id: str) -> None:
    settings = get_settings()
    store = RunStore(settings.run_storage_dir)
    record = store.load(run_id)
    record.status = "running"
    store.save(record)
    try:
        variants = generate_variants(
            base_prompt=record.request.base_prompt,
            categories=record.request.categories,
            sample_count=record.request.sample_count,
            seed=record.request.seed,
        )
        responses = await run_inference(record.request, variants, settings)
        points, metrics = await asyncio.to_thread(
            analyze_outputs,
            variants,
            responses,
            settings.allow_embedding_model_download,
            settings.enable_umap_hdbscan,
        )
        record.variants = variants
        record.responses = responses
        record.points = points
        record.metrics = metrics
        record.status = "completed"
        record.completed_at = datetime.now(timezone.utc)
        store.save(record)
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        record.completed_at = datetime.now(timezone.utc)
        store.save(record)
