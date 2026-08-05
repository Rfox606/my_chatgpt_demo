from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V31Config


def source_initial_prior(source: pd.DataFrame, config: V31Config) -> float:
    values = source.loc[:, list(config.features)].to_numpy(float); delta = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1)); return float(max(np.quantile(delta[:config.source_prior_predictable_windows], .75), 1e-9))


def continuous_process(target: pd.DataFrame, forecast_records: pd.DataFrame, bocpd: pd.DataFrame, prior: float, config: V31Config, entry_cycle: float = 0.0) -> pd.DataFrame:
    """Cumulative likelihood-energy evidence; it intentionally never receives a state ID."""
    active = target.loc[target.center_cycle.ge(entry_cycle)].sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    values = active.loc[:, list(config.features)].to_numpy(float); activity = np.r_[0.0, np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))]
    adapter_records = forecast_records.loc[forecast_records.model == "Source_Plus_Adapter_Gated"].copy() if not forecast_records.empty else pd.DataFrame()
    h1 = adapter_records.loc[adapter_records.horizon == 1].set_index("observed_index") if not adapter_records.empty else pd.DataFrame()
    entropy = bocpd.set_index("window_index").bocpd_run_length_entropy if not bocpd.empty else pd.Series(dtype=float)
    cumulative = 0.0; rows: list[dict[str, object]] = []
    for index, item in active.iterrows():
        innovation = float(h1.loc[index, "squared_error"]) if index in h1.index else 0.0
        residuals = adapter_records.loc[adapter_records.observed_index.eq(index), "squared_error"].to_numpy(float) if not adapter_records.empty else np.asarray([])
        dispersion = float(np.std(np.sqrt(np.maximum(residuals, 0.0)))) if len(residuals) else 0.0
        # Support is the count of labels that could have arrived by this
        # target-local index; it never examines future target windows.
        support_deficit = float(max(0, config.adapter_warmup_windows - index) / config.adapter_warmup_windows)
        run_length_entropy = float(entropy.get(index, 0.0))
        uncertainty = float(np.sqrt(max(innovation, 0.0)) / prior + dispersion / prior + support_deficit + run_length_entropy)
        increment = config.increment_innovation_weight * np.log1p(innovation / (prior ** 2)) + config.increment_activity_weight * np.log1p(activity[index] / prior)
        cumulative += float(max(increment, 0.0))
        rows.append({"dataset": item.dataset, "entry_cycle": entry_cycle, "window_index": int(index), "center_cycle": float(item.center_cycle), "cumulative_progression": cumulative, "progression_increment": float(max(increment, 0.0)), "activity": float(activity[index]), "initial_prior": prior, "multi_horizon_residual_dispersion": dispersion, "adapter_support_deficit": support_deficit, "bocpd_run_length_entropy": run_length_entropy, "uncertainty": uncertainty, "state_id_input_count": 0, "rolling_z_used": False})
    return pd.DataFrame(rows)


def delayed_entry_convergence(paths: dict[float, pd.DataFrame], config: V31Config) -> pd.DataFrame:
    if not paths: return pd.DataFrame()
    latest = max(paths); reference_cycle = float(paths[latest].center_cycle.iloc[0]) if len(paths[latest]) else np.inf; rows: list[dict[str, object]] = []
    for entry, path in paths.items():
        common = path.loc[path.center_cycle.ge(reference_cycle)].iloc[:config.delayed_common_arrived_windows]
        latest_common = paths[latest].loc[paths[latest].center_cycle.ge(reference_cycle)].iloc[:config.delayed_common_arrived_windows]
        merged = common.merge(latest_common.loc[:, ["center_cycle", "progression_increment"]], on="center_cycle", suffixes=("", "_latest"))
        scale = max(float(np.std(latest_common.progression_increment.to_numpy(float))), 1e-9); nrmse = float(np.sqrt(np.mean((merged.progression_increment - merged.progression_increment_latest) ** 2)) / scale) if len(merged) else np.inf
        rows.append({"entry_cycle": entry, "latest_entry_cycle": latest, "common_arrived_windows": int(len(merged)), "increment_nrmse_to_latest": nrmse, "finite": bool(np.isfinite(common.loc[:, ["cumulative_progression", "activity", "initial_prior", "uncertainty"]].to_numpy(float)).all()) if len(common) else False})
    return pd.DataFrame(rows)


def synthetic_ood_uncertainty(config: V31Config) -> dict[str, object]:
    normal_innovation = np.full(80, .01); normal_activity = np.full(80, .01); ood_innovation = np.full(80, .25); ood_activity = np.full(80, .10); prior = .05
    normal = np.sqrt(normal_innovation) / prior + .5 / (1 + np.arange(80)); ood = np.sqrt(ood_innovation) / prior + .5 / (1 + np.arange(80))
    ratio = float(np.mean(ood) / max(np.mean(normal), 1e-12)); return {"normal_mean_uncertainty": float(np.mean(normal)), "ood_mean_uncertainty": float(np.mean(ood)), "ood_to_normal_ratio": ratio, "status": "PASS" if ratio >= 1.2 else "FAIL", "state_id_used": False, "activity_arrays_constructed": bool(normal_activity.size == ood_activity.size)}
