from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContinuousStateV2Config
from .data import assert_label_free
from .state_metrics import CausalStateTracker, StateSpace, baseline_tes_reference
from .support_confidence import support_scores


def orthonormal_basis(common_indices: tuple[int, ...], w_common: np.ndarray, w_branch: np.ndarray, n_features: int) -> np.ndarray:
    vectors = []
    if len(common_indices):
        full = np.zeros(n_features); full[list(common_indices)] = w_common; vectors.append(full)
    vectors.append(w_branch)
    matrix = np.column_stack(vectors)
    return np.linalg.qr(matrix)[0][:, :np.linalg.matrix_rank(matrix)]


@dataclass
class Adapter:
    beta: np.ndarray
    learning_rate: float
    Q: np.ndarray
    replay_values: np.ndarray
    replay_initial_p: float
    replay_initial_bd: float


def run_target_online(
    target: pd.DataFrame, space: StateSpace, features: tuple[str, ...], source_support: pd.DataFrame,
    common_features: tuple[str, ...], w_common: np.ndarray, source_features: tuple[str, ...], source_weight: np.ndarray,
    config: ContinuousStateV2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly score with beta[t-1], then conditionally update beta for t+1."""
    assert_label_free(target)
    target = target.sort_values(["center_cycle", "window_index"]).reset_index(drop=True)
    reference = baseline_tes_reference(target, space, config)
    support_limits = source_support.set_index("feature_name")
    tracker = CausalStateTracker(space, reference, config)
    raw = target.loc[:, list(features)].to_numpy(float)
    if len(raw) == len(space.baseline_mask):
        replay_mask = space.baseline_mask
    else:
        replay_mask = ((target.end_cycle <= config.baseline_cycles) & target.is_restart_guard.eq(0)).to_numpy(bool)
    replay = raw[replay_mask][: config.baseline_replay_size]
    baseline_pre = np.array([space.score(item)[:2] for item in replay])
    adapter = Adapter(np.zeros(len(features)), config.adapter_learning_rate, orthonormal_basis(space.common_indices, w_common, space.w_branch, len(features)), replay, float(np.median(baseline_pre[:, 0])), float(np.median(baseline_pre[:, 1])))
    rows, logs = [], []
    for position, record in target.iterrows():
        x_raw = raw[position]
        x_adapted = x_raw - adapter.beta
        pre = tracker.predict(x_adapted)
        support = support_scores(x_raw, features, support_limits, common_features, w_common, space.w_branch, source_features, source_weight)
        pre.update(support)
        pre["source_head_disagreement"] = float("nan")
        pre.update({"pre_update_P_common": pre["P_common"], "pre_update_BD": pre["BD"], "pre_update_B_terminal": pre["B_terminal"], "pre_update_TES": pre["TES"], "pre_update_weighted_oos": pre["weighted_oos_common"], "adapter_updated": 0, "adapter_update_reason": "", "adapter_learning_rate": adapter.learning_rate, "adapter_rollback": 0, "rollback_reason": "", "beta_norm": float(np.linalg.norm(adapter.beta)), "beta_change_norm": 0.0})
        tracker.append_pre_update(pre)
        reason = ""
        if int(record.is_restart_guard): reason = "RESTART_GUARD"
        elif float(support["weighted_oos_common"]) > config.weighted_oos_update_max: reason = "HIGH_OOS"
        elif float(pre["TES"]) > reference["tes_p95"]: reason = "HIGH_TES"
        elif not np.isfinite(x_raw).all(): reason = "MISSING_DATA"
        if not reason:
            state_component = adapter.Q @ (adapter.Q.T @ x_raw) if adapter.Q.size else 0.
            residual = x_raw - state_component
            proposed = (1 - adapter.learning_rate) * adapter.beta + adapter.learning_rate * np.clip(residual, -config.adapter_clip, config.adapter_clip)
            replay_scores = np.asarray([space.score(value - proposed)[:2] for value in adapter.replay_values])
            p_drift = abs(float(np.median(replay_scores[:, 0])) - adapter.replay_initial_p)
            bd_change = abs(float(np.median(replay_scores[:, 1])) - adapter.replay_initial_bd) / (abs(adapter.replay_initial_bd) + config.eps)
            if p_drift > config.baseline_replay_p_drift_max or bd_change > config.baseline_replay_bd_drift_max:
                reason = "BASELINE_REPLAY_FAIL"; pre["adapter_rollback"] = 1; pre["rollback_reason"] = "P_DRIFT" if p_drift > config.baseline_replay_p_drift_max else "BD_DRIFT"; adapter.learning_rate = max(adapter.learning_rate * .5, config.adapter_learning_rate_min)
            else:
                previous = adapter.beta.copy(); adapter.beta = proposed; pre["adapter_updated"] = 1; pre["beta_change_norm"] = float(np.linalg.norm(adapter.beta - previous))
        pre["adapter_update_reason"] = "UPDATED" if pre["adapter_updated"] else reason
        pre["adapter_learning_rate"] = adapter.learning_rate; pre["beta_norm"] = float(np.linalg.norm(adapter.beta))
        rows.append({**record.to_dict(), **pre})
        logs.append({"window_index": record.window_index, "center_cycle": record.center_cycle, "adapter_updated": pre["adapter_updated"], "adapter_update_reason": pre["adapter_update_reason"], "adapter_rollback": pre["adapter_rollback"], "rollback_reason": pre["rollback_reason"], "adapter_learning_rate": pre["adapter_learning_rate"], "beta_norm": pre["beta_norm"], "beta_change_norm": pre["beta_change_norm"]})
    return pd.DataFrame(rows), pd.DataFrame(logs)
