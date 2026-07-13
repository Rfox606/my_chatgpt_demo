from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import TemporalPrototypeConfig
from .data import SourceScaler, causal_sequences, stagewise_source_split
from .evaluation import source_selection_score, stage_metrics
from .model import TemporalPrototypeNet


@dataclass
class SourceBundle:
    dataset: str
    model: TemporalPrototypeNet
    scaler: SourceScaler
    prototypes: np.ndarray
    variances: np.ndarray
    distance_p95: np.ndarray
    support: np.ndarray
    tes_p99: float
    source_axis: np.ndarray
    validation_metrics: dict[str, float]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensors(sequences: np.ndarray, labels: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = torch.from_numpy(sequences)
    y = None if labels is None else torch.from_numpy(labels.astype(np.int64))
    return x, y


def _batch_loss(model: TemporalPrototypeNet, x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor, restart: torch.Tensor) -> torch.Tensor:
    out = model(x)
    threshold_targets = (y.unsqueeze(1) > torch.arange(1, 5, device=y.device)).float()
    ordinal = nn.functional.binary_cross_entropy_with_logits(out["ordinal_logits"], threshold_targets)
    targets = torch.tensor([0.0, 0.2, 0.45, 0.7, 1.0], device=y.device)[y - 1]
    health = nn.functional.smooth_l1_loss(out["health"], targets)
    embedding = out["embedding"]
    compactness = torch.zeros((), device=y.device)
    for stage in range(1, 6):
        mask = y == stage
        if mask.sum() > 1:
            compactness = compactness + ((embedding[mask] - embedding[mask].mean(dim=0)) ** 2).sum(dim=1).mean()
    delta = indices.unsqueeze(1) - indices.unsqueeze(0)
    higher = y.unsqueeze(1) > y.unsqueeze(0)
    rank_mask = (delta >= 100) & higher
    if rank_mask.any():
        hi, lo = torch.where(rank_mask)
        ranking = torch.relu(0.10 - out["health"][hi] + out["health"][lo]).mean()
    else:
        ranking = torch.zeros((), device=y.device)
    adjacent = (indices[1:] - indices[:-1] == 1) & (~restart[1:].bool())
    temporal = torch.abs(out["health"][1:] - out["health"][:-1])[adjacent].mean() if adjacent.any() else torch.zeros((), device=y.device)
    return ordinal + 0.5 * health + 0.2 * ranking + 0.1 * temporal + 0.1 * compactness


def _predict(model: TemporalPrototypeNet, sequences: np.ndarray, batch_size: int = 512) -> dict[str, np.ndarray]:
    model.eval()
    results: dict[str, list[np.ndarray]] = {"embedding": [], "stage_probs": [], "health": [], "ordinal_logits": []}
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            out = model(torch.from_numpy(sequences[start:start + batch_size]))
            for key in results:
                results[key].append(out[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in results.items()}


def _make_prototypes(embedding: np.ndarray, stage: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proto, variance, p95, support = [], [], [], []
    for value in range(1, 6):
        x = embedding[stage == value]
        if len(x) == 0:
            raise ValueError(f"Source training has no Stage {value} samples.")
        mean = x.mean(axis=0)
        dist = np.sum((x - mean) ** 2, axis=1)
        proto.append(mean)
        variance.append(np.maximum(x.var(axis=0), 1e-5))
        p95.append(np.percentile(dist, 95))
        support.append(len(x))
    return np.asarray(proto), np.asarray(variance), np.asarray(p95), np.asarray(support)


def train_source(frame: pd.DataFrame, dataset: str, config: TemporalPrototypeConfig, output_dir: Path) -> tuple[SourceBundle, pd.DataFrame]:
    source = frame[frame["dataset"] == dataset].sort_values("window_index").reset_index(drop=True).copy()
    train_mask, gap_mask, validation_mask = stagewise_source_split(source, config)
    scaler = SourceScaler.fit(source.loc[train_mask], config.input_features, config)
    values, _ = scaler.transform(source, config)
    sequences = causal_sequences(values, source["restart_mask"].to_numpy(bool), config.sequence_length)
    labels = source["stage"].to_numpy(int)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    records: list[dict] = []
    best_score, best_state, best_metrics, best_seed = -np.inf, None, {}, None
    for seed in config.seeds:
        _set_seed(seed)
        model = TemporalPrototypeNet(len(config.input_features))
        model.assert_unidirectional()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        local_best, local_state, waiting = -np.inf, None, 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            for start in range(0, len(train_indices), config.batch_size):
                batch = train_indices[start:start + config.batch_size]
                x, y = _tensors(sequences[batch], labels[batch])
                loss = _batch_loss(model, x, y, torch.from_numpy(source.loc[batch, "window_index"].to_numpy(np.int64)), torch.from_numpy(source.loc[batch, "restart_mask"].to_numpy(bool)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            prediction = _predict(model, sequences[validation_indices])
            metrics = stage_metrics(labels[validation_indices], prediction["stage_probs"])
            score = source_selection_score(metrics)
            if score > local_best + 1e-8:
                local_best, local_state, waiting = score, copy.deepcopy(model.state_dict()), 0
            else:
                waiting += 1
            if waiting >= config.patience:
                break
        assert local_state is not None
        model.load_state_dict(local_state)
        final_prediction = _predict(model, sequences[validation_indices])
        final_metrics = stage_metrics(labels[validation_indices], final_prediction["stage_probs"])
        final_score = source_selection_score(final_metrics)
        torch.save({"model_state": model.state_dict(), "seed": seed, "score": final_score}, output_dir / f"{dataset}_seed_{seed}.pt")
        records.append({"dataset": dataset, "seed": seed, "best_epoch": epoch - waiting, "selection_score": final_score, "selected": False, **final_metrics})
        if final_score > best_score:
            best_score, best_state, best_metrics, best_seed = final_score, copy.deepcopy(model.state_dict()), final_metrics, seed
    selected = TemporalPrototypeNet(len(config.input_features))
    selected.load_state_dict(best_state)
    train_prediction = _predict(selected, sequences[train_indices])
    prototypes, variances, p95, support = _make_prototypes(train_prediction["embedding"], labels[train_indices])
    # Fit the source health axis to the five source-stage prototype means.  Unlike using
    # only Stage5-Stage1, this guarantees an ordered source reference whenever the five
    # prototype means span the ordinal target vector, so valid target updates are not
    # rejected solely because an intermediate source mean is slightly out of sequence.
    stage_target = np.linspace(0.0, 1.0, 5)
    centred = prototypes - prototypes.mean(axis=0, keepdims=True)
    axis = np.linalg.pinv(centred) @ (stage_target - stage_target.mean())
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    records[int(np.argmax([r["selection_score"] for r in records]))]["selected"] = True
    tes_p99 = float(np.nanpercentile(source.loc[train_mask, "TES"], 99))
    bundle = SourceBundle(dataset, selected, scaler, prototypes, variances, p95, support, tes_p99, axis, best_metrics)
    torch.save({
        "model_state": selected.state_dict(), "dataset": dataset, "features": list(config.input_features),
        "scaler": {"median": scaler.median, "iqr": scaler.iqr, "lower": scaler.lower, "upper": scaler.upper},
        "prototypes": prototypes, "variances": variances, "distance_p95": p95, "support": support,
        "validation_metrics": best_metrics, "seed": best_seed,
    }, output_dir / f"{dataset}_source_model.pt")
    source["split"] = np.where(train_mask, "train", np.where(gap_mask, "gap", "validation"))
    return bundle, pd.DataFrame(records)
