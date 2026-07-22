from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from temporal_prototype_v2.model import TemporalPrototypeNet

from .config import OrderedRegimeConfig
from .data import causal_sequences, source_split
from .trajectory import causal_trajectory


@dataclass(frozen=True)
class FrozenScaler:
    features: tuple[str, ...]
    median: np.ndarray
    iqr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.loc[:, self.features].to_numpy(float)
        missing = np.mean(~np.isfinite(x), axis=1)
        x = np.where(np.isfinite(x), x, self.median)
        x = np.clip(x, self.lower, self.upper)
        return ((x - self.median) / self.iqr).astype(np.float32), missing.astype(np.float32)


def _network(model: TemporalPrototypeNet, sequence: np.ndarray) -> dict[str, np.ndarray]:
    output = {"embedding": [], "stage_probs": [], "health": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequence), 512):
            batch = model(torch.from_numpy(sequence[start:start + 512]).float())
            for key in output:
                output[key].append(batch[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in output.items()}


@dataclass
class SourceArtifacts:
    dataset: str
    model: TemporalPrototypeNet
    scaler: FrozenScaler
    level_proto: np.ndarray
    trajectory_proto: np.ndarray
    level_radius_p95: np.ndarray
    trajectory_radius_p95: np.ndarray
    level_distance_ref: list[np.ndarray]
    trajectory_distance_ref: list[np.ndarray]
    source_tes_p99: float
    health_axis: np.ndarray
    typical_level_radius: float
    typical_trajectory_radius: float
    source_state_dict: dict[str, torch.Tensor]

    def encode(self, frame: pd.DataFrame, config: OrderedRegimeConfig) -> dict[str, np.ndarray]:
        values, missing = self.scaler.transform(frame)
        sequence = causal_sequences(values, frame.restart_mask.to_numpy(bool), config.sequence_length)
        output = _network(self.model, sequence)
        trajectory, fields = causal_trajectory(output["embedding"])
        fields.update(output)
        fields["trajectory"] = trajectory
        fields["missing_fraction"] = missing
        fields["health_axis_projection"] = output["embedding"] @ self.health_axis
        fields["health_axis_velocity20"] = np.r_[0., np.diff(fields["health_axis_projection"])]
        fields["health_axis_velocity100"] = fields["health_axis_projection"] - np.r_[np.repeat(fields["health_axis_projection"][0], min(100, len(frame))), fields["health_axis_projection"][:-100]][:len(frame)]
        return fields

    def unchanged(self) -> bool:
        return all(torch.equal(value, self.model.state_dict()[key]) for key, value in self.source_state_dict.items())


def load_source_artifacts(frame: pd.DataFrame, dataset: str, config: OrderedRegimeConfig) -> SourceArtifacts:
    path = Path(config.v2_model_dir) / f"{dataset}_source_model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalPrototypeNet(len(config.input_features))
    model.load_state_dict(payload["model_state"]); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    stats = payload["scaler"]
    scaler = FrozenScaler(tuple(config.input_features), np.asarray(stats["median"]), np.maximum(np.asarray(stats["iqr"]), 1e-6), np.asarray(stats["lower"]), np.asarray(stats["upper"]))
    source = frame[frame.dataset == dataset].sort_values("window_index").reset_index(drop=True)
    train, _, _ = source_split(source, config)
    values, _ = scaler.transform(source)
    sequence = causal_sequences(values, source.restart_mask.to_numpy(bool), config.sequence_length)
    output = _network(model, sequence)
    trajectory, _ = causal_trajectory(output["embedding"])
    labels = source.stage.to_numpy(int)
    level_proto=[]; traj_proto=[]; lr=[]; tr=[]; lref=[]; tref=[]
    for stage in range(1, 6):
        x = output["embedding"][train & (labels == stage)]; y = trajectory[train & (labels == stage)]
        mu=x.mean(axis=0); nu=y.mean(axis=0)
        dx=np.linalg.norm(x-mu,axis=1); dy=np.linalg.norm(y-nu,axis=1)
        level_proto.append(mu); traj_proto.append(nu); lr.append(np.percentile(dx,95)); tr.append(np.percentile(dy,95)); lref.append(dx); tref.append(dy)
    level_proto=np.asarray(level_proto); traj_proto=np.asarray(traj_proto)
    target=np.linspace(0,1,5); axis=np.linalg.pinv(level_proto-level_proto.mean(axis=0)) @ (target-target.mean()); axis/=max(np.linalg.norm(axis),1e-8)
    return SourceArtifacts(dataset, model, scaler, level_proto, traj_proto, np.asarray(lr), np.asarray(tr), lref, tref,
                           float(np.percentile(source.loc[train,"TES"],99)), axis, float(np.median(lr)), float(np.median(tr)),
                           {key:value.detach().clone() for key,value in model.state_dict().items()})


def load_v12_anchor(config: OrderedRegimeConfig, direction: str) -> pd.DataFrame:
    anchor = pd.read_csv(config.v12_scores_path, usecols=["direction_id", "model", "window_index", "raw_logit"])
    return anchor[(anchor.direction_id == direction) & (anchor.model == "P1")][["window_index", "raw_logit"]].copy()
