import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.app.schemas import RunRecord, TrainingRecord


CATEGORIES = ["lexical", "syntactic", "persona", "politeness", "specificity", "negation", "position"]
METRICS_PATH = "risk_predictor_metrics.json"
MODEL_PATH = "risk_predictor.pt"


def idle_training_record(model_dir: Path) -> TrainingRecord:
    metrics_path = model_dir / METRICS_PATH
    if metrics_path.exists():
        return TrainingRecord.model_validate(json.loads(metrics_path.read_text(encoding="utf-8")))
    return TrainingRecord(status="idle")


def train_risk_predictor(run_dir: Path, model_dir: Path, epochs: int = 80) -> TrainingRecord:
    import torch
    from torch import nn

    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    model_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = bool(torch.cuda.is_available())
    device = torch.device("cuda" if cuda_available else "cpu")
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"

    try:
        features, distances = _load_dataset(run_dir)
        if len(features) < 8:
            raise ValueError("Need at least 8 completed prompt-variation examples before training.")

        threshold = float(np.percentile(distances, 75))
        labels = (distances >= threshold).astype(np.float32)
        if labels.min() == labels.max():
            threshold = float(np.median(distances))
            labels = (distances >= threshold).astype(np.float32)
        if labels.min() == labels.max():
            raise ValueError("Training labels have only one class. Run more varied prompt tests first.")

        rng = np.random.default_rng(42)
        order = rng.permutation(len(features))
        features = features[order]
        labels = labels[order]

        split = max(1, int(len(features) * 0.8))
        if split >= len(features):
            split = len(features) - 1

        train_x_np = features[:split]
        train_y_np = labels[:split]
        val_x_np = features[split:]
        val_y_np = labels[split:]

        mean = train_x_np.mean(axis=0, keepdims=True)
        std = train_x_np.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0

        train_x = torch.tensor((train_x_np - mean) / std, dtype=torch.float32, device=device)
        train_y = torch.tensor(train_y_np.reshape(-1, 1), dtype=torch.float32, device=device)
        val_x = torch.tensor((val_x_np - mean) / std, dtype=torch.float32, device=device)
        val_y = torch.tensor(val_y_np.reshape(-1, 1), dtype=torch.float32, device=device)

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(train_x.shape[1], 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        final_loss = 0.0
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        model.eval()
        with torch.no_grad():
            predictions = (torch.sigmoid(model(val_x)) >= 0.5).float()
            accuracy = float((predictions.eq(val_y)).float().mean().cpu().item())

        artifact_path = model_dir / MODEL_PATH
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "categories": CATEGORIES,
                "threshold": threshold,
                "input_dim": int(train_x.shape[1]),
            },
            artifact_path,
        )

        record = TrainingRecord(
            status="completed",
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            device=str(device),
            cuda_available=cuda_available,
            cuda_device_name=cuda_device_name,
            dataset_size=int(len(features)),
            train_size=int(len(train_x_np)),
            validation_size=int(len(val_x_np)),
            epochs=epochs,
            final_loss=final_loss,
            validation_accuracy=accuracy,
            threshold=threshold,
            training_ms=int((time.perf_counter() - start_time) * 1000),
            model_path=str(artifact_path),
        )
    except Exception as exc:
        record = TrainingRecord(
            status="failed",
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            device=str(device),
            cuda_available=cuda_available,
            cuda_device_name=cuda_device_name,
            training_ms=int((time.perf_counter() - start_time) * 1000),
            error=str(exc),
        )

    (model_dir / METRICS_PATH).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _load_dataset(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    distances: list[float] = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            record = RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if record.status != "completed" or not record.points:
            continue
        variant_by_id = {variant.id: variant for variant in record.variants}
        for point in record.points:
            variant = variant_by_id.get(point.variant_id)
            if not variant or variant.category == "base":
                continue
            category_vector = [1.0 if variant.category == category else 0.0 for category in CATEGORIES]
            rows.append(
                category_vector
                + [
                    float(point.output_length),
                    float(point.token_count),
                    float(point.entropy),
                    float(point.cluster),
                    float(point.x),
                    float(point.y),
                ]
            )
            distances.append(float(point.semantic_distance))

    if not rows:
        return np.empty((0, len(CATEGORIES) + 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32), np.asarray(distances, dtype=np.float32)
