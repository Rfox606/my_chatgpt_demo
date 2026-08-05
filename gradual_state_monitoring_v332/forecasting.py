from __future__ import annotations

import gc
import pickle
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gradual_state_monitoring_v33.data import MAIN_FEATURES, load_window_data

from .config import GradualStateMonitoringV332Config
from .online import validate_label_free_online_frame


@dataclass
class ForecastRun:
    setting: str
    source_dataset: str
    target_dataset: str
    prediction_start_index: int
    training_examples: int
    predictions: pd.DataFrame
    h20_error_at_arrival: dict[str, np.ndarray]


class CausalConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(inputs, outputs, kernel_size=kernel_size, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Left padding is the causal constraint: no convolution sees a later window.
        value = torch.nn.functional.pad(value, (self.left_padding, 0))
        return self.dropout(self.activation(self.conv(value)))


class CausalTCN(nn.Module):
    def __init__(self, feature_count: int, output_count: int, config: GradualStateMonitoringV332Config) -> None:
        super().__init__()
        channels = config.tcn_channels
        if len(channels) != len(config.tcn_dilations):
            raise ValueError("TCN channels and dilations must have equal predeclared lengths")
        blocks: list[nn.Module] = []
        previous = feature_count
        for channel, dilation in zip(channels, config.tcn_dilations):
            blocks.append(CausalConvBlock(previous, channel, config.tcn_kernel_size, dilation, config.tcn_dropout))
            previous = channel
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(previous, output_count)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(history.transpose(1, 2))
        return self.head(encoded[:, :, -1])


class SmallGRU(nn.Module):
    def __init__(self, feature_count: int, output_count: int, config: GradualStateMonitoringV332Config) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_count, config.gru_hidden_size, num_layers=1, batch_first=True, dropout=0.0)
        self.head = nn.Linear(config.gru_hidden_size, output_count)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history)
        return self.head(hidden[-1])


class _ZeroPersistence:
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.zeros((len(inputs), 18), dtype=float)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _calibration_scale(values: np.ndarray, config: GradualStateMonitoringV332Config) -> tuple[np.ndarray, np.ndarray]:
    observed = values[:config.scaler_calibration_windows]
    if len(observed) < config.scaler_calibration_windows:
        raise ValueError("A target sequence needs 128 observed no-label windows for scaler calibration")
    location = np.mean(observed, axis=0)
    scale = np.std(observed, axis=0, ddof=1)
    return location, np.maximum(scale, config.eps)


def _normalise(values: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - location) / scale


def _origins(length: int, history: int, maximum_horizon: int, end_exclusive: int) -> np.ndarray:
    upper = min(length - maximum_horizon, end_exclusive)
    return np.arange(history - 1, upper, dtype=int)


def _sample_origins(origins: np.ndarray, config: GradualStateMonitoringV332Config) -> np.ndarray:
    if len(origins) < config.min_training_examples:
        raise ValueError(f"Only {len(origins)} causal training examples; need {config.min_training_examples}")
    if len(origins) <= config.max_training_examples:
        return origins
    # Fixed uniform temporal subsampling prevents a large experiment from receiving
    # a different, label-selected training slice.
    position = np.linspace(0, len(origins) - 1, config.max_training_examples, dtype=int)
    return origins[position]


def _make_examples(normalised: np.ndarray, origins: np.ndarray, config: GradualStateMonitoringV332Config) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([normalised[origin - config.feature_history + 1: origin + 1] for origin in origins]).astype(np.float32)
    y = np.stack([np.concatenate([normalised[origin + horizon] - normalised[origin] for horizon in config.horizons]) for origin in origins]).astype(np.float32)
    return x, y


def _make_predictor(name: str, x: np.ndarray, y: np.ndarray, config: GradualStateMonitoringV332Config) -> Any:
    flattened = x.reshape(len(x), -1)
    if name == "Persistence":
        return _ZeroPersistence()
    if name == "Ridge":
        model = Pipeline((("scale", StandardScaler()), ("ridge", Ridge(alpha=config.ridge_alpha))))
        model.fit(flattened, y)
        return model
    if name == "RBF_Ridge":
        model = Pipeline((("scale", StandardScaler()),
                          ("rbf", RBFSampler(gamma=config.rbf_gamma, n_components=config.rbf_components, random_state=config.random_seed)),
                          ("ridge", Ridge(alpha=config.ridge_alpha))))
        model.fit(flattened, y)
        return model
    _set_seed(config.random_seed + (17 if name == "TCN" else 29))
    if name == "TCN":
        model: nn.Module = CausalTCN(x.shape[2], y.shape[1], config)
    elif name == "GRU":
        model = SmallGRU(x.shape[2], y.shape[1], config)
    else:
        raise ValueError(f"Unknown V3.3.2 model {name}")
    model.train()
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(config.random_seed + (17 if name == "TCN" else 29))
    loader = DataLoader(data, batch_size=config.deep_batch_size, shuffle=True, generator=generator)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.deep_learning_rate)
    loss = nn.HuberLoss(delta=config.huber_delta)
    for _ in range(config.deep_epochs):
        for batch_x, batch_y in loader:
            optimiser.zero_grad(set_to_none=True)
            error = loss(model(batch_x), batch_y)
            error.backward()
            optimiser.step()
    model.eval()
    return model


def _predict(model: Any, name: str, x: np.ndarray) -> np.ndarray:
    if name in {"Persistence", "Ridge", "RBF_Ridge"}:
        return np.asarray(model.predict(x.reshape(len(x), -1)), dtype=float)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            outputs.append(model(torch.from_numpy(x[start:start + 512])).cpu().numpy())
    return np.vstack(outputs).astype(float) if outputs else np.empty((0, 18), dtype=float)


def _forecast_one_setting(setting: str, source: pd.DataFrame, target: pd.DataFrame, train_end: int,
                          prediction_start: int, config: GradualStateMonitoringV332Config) -> ForecastRun:
    source = validate_label_free_online_frame(source, config)
    target = validate_label_free_online_frame(target, config)
    source_values = source.loc[:, MAIN_FEATURES].to_numpy(float)
    target_values = target.loc[:, MAIN_FEATURES].to_numpy(float)
    source_location, source_scale = _calibration_scale(source_values, config)
    target_location, target_scale = _calibration_scale(target_values, config)
    source_normalised = _normalise(source_values, source_location, source_scale)
    target_normalised = _normalise(target_values, target_location, target_scale)
    train_origins = _sample_origins(_origins(len(source_values), config.feature_history, max(config.horizons), train_end), config)
    train_x, train_y = _make_examples(source_normalised, train_origins, config)
    origins = np.arange(prediction_start, len(target_values), dtype=int)
    valid_history = origins >= config.feature_history - 1
    origins = origins[valid_history]
    x_eval = np.stack([target_normalised[origin - config.feature_history + 1: origin + 1] for origin in origins]).astype(np.float32) if len(origins) else np.empty((0, config.feature_history, len(MAIN_FEATURES)), dtype=np.float32)
    rows: list[dict[str, object]] = []
    h20_errors: dict[str, np.ndarray] = {}
    for model_name in config.state_models:
        model = _make_predictor(model_name, train_x, train_y, config)
        predicted_delta = _predict(model, model_name, x_eval).reshape(len(origins), len(config.horizons), len(MAIN_FEATURES))
        error_at_arrival = np.full(len(target_values), np.nan, dtype=float)
        for origin_position, origin in enumerate(origins):
            for horizon_position, horizon in enumerate(config.horizons):
                target_index = int(origin + horizon)
                available = target_index < len(target_values)
                predicted_physical_delta = predicted_delta[origin_position, horizon_position] * target_scale
                actual_delta = target_values[target_index] - target_values[origin] if available else np.full(len(MAIN_FEATURES), np.nan)
                mae = float(np.mean(np.abs(predicted_physical_delta - actual_delta))) if available else np.nan
                rows.append({
                    "training_setting": setting, "source_dataset": str(source.dataset.iloc[0]), "target_dataset": str(target.dataset.iloc[0]),
                    "model": model_name, "input_index": int(origin), "origin_cycle": float(target.center_cycle.iloc[origin]),
                    "target_index": target_index, "horizon": horizon, "mae": mae,
                    "prediction_available": 1, "target_available": int(available),
                    "training_examples": int(len(train_origins)), "prediction_start_index": int(prediction_start),
                })
                if horizon == 20 and available:
                    error_at_arrival[target_index] = mae
        h20_errors[model_name] = error_at_arrival
        # The desktop runtime has a constrained CPU allocator.  ForecastRun stores
        # arrays and error arrivals, never trained neural parameters; release each
        # transient model before the next predeclared comparison model is trained.
        del model
        gc.collect()
    return ForecastRun(setting, str(source.dataset.iloc[0]), str(target.dataset.iloc[0]), prediction_start, len(train_origins), pd.DataFrame(rows), h20_errors)


def _forecast_setting_isolated(setting: str, source: pd.DataFrame, target: pd.DataFrame, train_end: int,
                               prediction_start: int, config: GradualStateMonitoringV332Config) -> ForecastRun:
    """Run one complete predeclared setting in a fresh interpreter.

    Each child trains its fixed model set sequentially and releases every transient
    model.  Children reload label-free raw input themselves; no model state is
    shared between experiment settings.
    """
    with tempfile.TemporaryDirectory(prefix="v332_forecast_") as temporary:
        temporary_path = Path(temporary)
        request = temporary_path / "request.pkl"
        result_path = temporary_path / "result.pkl"
        payload = {"setting": setting, "source_dataset": str(source.dataset.iloc[0]), "target_dataset": str(target.dataset.iloc[0]),
                   "train_end": train_end, "prediction_start": prediction_start, "model_names": config.state_models, "config": config}
        request.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        completed = subprocess.run([sys.executable, "-m", "gradual_state_monitoring_v332.forecasting", "--worker-request", str(request), "--worker-result", str(result_path)],
                                   cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"{setting} forecast child failed (exit {completed.returncode}): {completed.stdout}\n{completed.stderr}")
        serialised = pickle.loads(result_path.read_bytes())
        return ForecastRun(**serialised)


def _worker_cli(request_path: str, result_path: str) -> None:
    request = pickle.loads(Path(request_path).read_bytes())
    config: GradualStateMonitoringV332Config = request["config"]
    source_data = load_window_data(config.input_path)
    source = source_data[source_data.dataset == request["source_dataset"]].reset_index(drop=True)
    target = source_data[source_data.dataset == request["target_dataset"]].reset_index(drop=True)
    result = _forecast_one_setting(request["setting"], source, target, int(request["train_end"]), int(request["prediction_start"]),
                                   replace(config, state_models=tuple(request["model_names"])))
    serialised = {
        "setting": result.setting, "source_dataset": result.source_dataset, "target_dataset": result.target_dataset,
        "prediction_start_index": result.prediction_start_index, "training_examples": result.training_examples,
        "predictions": result.predictions, "h20_error_at_arrival": result.h20_error_at_arrival,
    }
    Path(result_path).write_bytes(pickle.dumps(serialised, protocol=pickle.HIGHEST_PROTOCOL))


def run_label_free_forecasts(frame_by_dataset: dict[str, pd.DataFrame], config: GradualStateMonitoringV332Config) -> list[ForecastRun]:
    """All training and prediction happens before any Stage artifact is loaded."""
    exp1 = validate_label_free_online_frame(frame_by_dataset["Exp1"], config)
    exp2 = validate_label_free_online_frame(frame_by_dataset["Exp2"], config)
    runs: list[ForecastRun] = []
    for name, frame in (("Exp1", exp1), ("Exp2", exp2)):
        train_end = int(len(frame) * config.within_train_fraction)
        runs.append(_forecast_setting_isolated(f"within_{name}", frame, frame, train_end, train_end, config))
    # Exp1 is entirely historical to Exp2.  Exp2's first 128 target windows are
    # used only for the target normalisation frozen at entry, not model fitting.
    runs.append(_forecast_setting_isolated("Exp1_pretrain_to_Exp2", exp1, exp2, len(exp1), config.scaler_calibration_windows, config))
    return runs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-request", required=True)
    parser.add_argument("--worker-result", required=True)
    arguments = parser.parse_args()
    _worker_cli(arguments.worker_request, arguments.worker_result)
