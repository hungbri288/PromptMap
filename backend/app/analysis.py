import hashlib
import importlib.util
import math
import time
from collections import Counter, defaultdict

import numpy as np

from backend.app.schemas import CategorySummary, MapPoint, PromptVariant, ResponseResult, RunMetrics


def analyze_outputs(
    variants: list[PromptVariant],
    responses: list[ResponseResult],
    allow_model_download: bool = False,
    enable_umap_hdbscan: bool = False,
) -> tuple[list[MapPoint], RunMetrics]:
    response_by_id = {response.variant_id: response for response in responses}
    texts = [response_by_id[variant.id].output or response_by_id[variant.id].error or "" for variant in variants]

    embed_started = time.perf_counter()
    vectors, device = _embed(texts, allow_model_download)
    embedding_ms = int((time.perf_counter() - embed_started) * 1000)

    coords, projection_method = _project(vectors, enable_umap_hdbscan)
    clusters, clustering_method = _cluster(coords, enable_umap_hdbscan)
    base_vector = vectors[0]

    points = []
    for index, variant in enumerate(variants):
        response = response_by_id[variant.id]
        text = response.output
        points.append(
            MapPoint(
                variant_id=variant.id,
                x=float(coords[index][0]),
                y=float(coords[index][1]),
                cluster=int(clusters[index]),
                semantic_distance=float(_cosine_distance(base_vector, vectors[index])),
                output_length=len(text),
                token_count=len(text.split()),
                entropy=float(_entropy(text)),
            )
        )

    summaries = _category_summaries(variants, points)
    metrics = RunMetrics(
        embedding_device=device,
        embedding_ms=embedding_ms,
        projection_method=projection_method,
        clustering_method=clustering_method,
        api_success_count=sum(1 for response in responses if not response.error),
        api_error_count=sum(1 for response in responses if response.error),
        category_summaries=summaries,
    )
    return points, metrics


def dependency_status() -> dict[str, str | bool]:
    status: dict[str, str | bool] = {}
    try:
        import torch

        status["torch"] = "available"
        status["cuda"] = bool(torch.cuda.is_available())
        status["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        status["torch"] = f"unavailable: {exc}"
        status["cuda"] = False
        status["cuda_device"] = "cpu"
    for module in ["sentence_transformers", "umap", "hdbscan", "sklearn"]:
        status[module] = "installed" if importlib.util.find_spec(module) else "not-installed"
    return status


def _embed(texts: list[str], allow_model_download: bool) -> tuple[np.ndarray, str]:
    if not allow_model_download:
        return _hashed_embeddings(texts), "hashed-offline"
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device,
        )
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32), device
    except Exception:
        return _hashed_embeddings(texts), "hashed-fallback"


def _hashed_embeddings(texts: list[str], dimensions: int = 64) -> np.ndarray:
    rows = []
    for text in texts:
        vector = np.zeros(dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:2], "big") % dimensions
            sign = 1 if digest[2] % 2 == 0 else -1
            vector[index] += sign
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return np.vstack(rows)


def _project(vectors: np.ndarray, enable_umap: bool) -> tuple[np.ndarray, str]:
    if len(vectors) < 3:
        return np.pad(vectors[:, :1], ((0, 0), (0, 1))), "fallback"
    if enable_umap:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(vectors) - 1), min_dist=0.12)
            return reducer.fit_transform(vectors), "umap"
        except Exception:
            pass
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.pad(coords, ((0, 0), (0, 1)))
    return coords[:, :2], "pca"


def _cluster(coords: np.ndarray, enable_hdbscan: bool) -> tuple[np.ndarray, str]:
    if len(coords) < 6:
        return np.zeros(len(coords), dtype=int), "single-cluster"
    if enable_hdbscan:
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(3, len(coords) // 12), min_samples=2)
            labels = clusterer.fit_predict(coords)
            return labels.astype(int), "hdbscan"
        except Exception:
            pass
    try:
        from sklearn.cluster import KMeans

        k = min(5, max(2, int(math.sqrt(len(coords) / 2))))
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(coords)
        return labels.astype(int), "kmeans"
    except Exception:
        return np.zeros(len(coords), dtype=int), "single-cluster-fallback"


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return 1.0 - float(np.dot(left, right) / denom)


def _entropy(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _category_summaries(variants: list[PromptVariant], points: list[MapPoint]) -> list[CategorySummary]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for variant, point in zip(variants, points, strict=True):
        if variant.category != "base":
            buckets[str(variant.category)].append(point.semantic_distance)
    return [
        CategorySummary(
            category=category,
            count=len(values),
            avg_distance=float(sum(values) / len(values)),
            max_distance=float(max(values)),
        )
        for category, values in sorted(buckets.items())
        if values
    ]
