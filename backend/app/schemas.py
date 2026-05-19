from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


PerturbationCategory = Literal[
    "lexical",
    "syntactic",
    "persona",
    "politeness",
    "specificity",
    "negation",
    "position",
]

RunMode = Literal["demo", "mock", "live", "local"]
RunStatus = Literal["queued", "running", "completed", "failed"]
TrainingStatus = Literal["idle", "running", "completed", "failed"]


class RunRequest(BaseModel):
    base_prompt: str = Field(..., min_length=3)
    categories: list[PerturbationCategory] = Field(
        default_factory=lambda: [
            "lexical",
            "syntactic",
            "persona",
            "politeness",
            "specificity",
            "negation",
            "position",
        ]
    )
    sample_count: int = Field(default=64, ge=5, le=250)
    model: str | None = None
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)
    seed: int = Field(default=42)
    mode: RunMode = "demo"


class PromptVariant(BaseModel):
    id: str
    prompt: str
    category: PerturbationCategory | Literal["base"]
    transform: str
    system_prompt: str | None = None


class ResponseResult(BaseModel):
    variant_id: str
    output: str
    latency_ms: int
    error: str | None = None


class MapPoint(BaseModel):
    variant_id: str
    x: float
    y: float
    cluster: int
    semantic_distance: float
    output_length: int
    token_count: int
    entropy: float


class CategorySummary(BaseModel):
    category: str
    count: int
    avg_distance: float
    max_distance: float


class RunMetrics(BaseModel):
    embedding_device: str
    embedding_ms: int
    projection_method: str
    clustering_method: str
    api_success_count: int
    api_error_count: int
    kl_divergence_available: bool = False
    category_summaries: list[CategorySummary]


class RunRecord(BaseModel):
    id: str
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None = None
    request: RunRequest
    variants: list[PromptVariant] = Field(default_factory=list)
    responses: list[ResponseResult] = Field(default_factory=list)
    points: list[MapPoint] = Field(default_factory=list)
    base_variant_id: str = "base"
    metrics: RunMetrics | None = None
    error: str | None = None


class TrainingRecord(BaseModel):
    status: TrainingStatus = "idle"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    device: str = "unknown"
    cuda_available: bool = False
    cuda_device_name: str = "cpu"
    dataset_size: int = 0
    train_size: int = 0
    validation_size: int = 0
    epochs: int = 0
    final_loss: float | None = None
    validation_accuracy: float | None = None
    threshold: float | None = None
    training_ms: int | None = None
    model_path: str | None = None
    error: str | None = None
