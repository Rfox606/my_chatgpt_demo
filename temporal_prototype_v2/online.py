from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import TemporalPrototypeConfig
from .data import causal_sequences, reject_target_labels
from .model import TemporalPrototypeNet
from .source import SourceBundle


ABLATIONS = (
    "B0_STATIC_SOURCE", "B1_TIME_ONLY_HMM", "B2_STATIC_HMM", "B3_DYNAMIC_PROTOTYPE",
    "B4_TEACHER_MEMORY", "B5_TEMPORAL_RANKING", "B6_FULL_ADAPTATION", "B7_TENT_LITE",
)


def transition_matrix(stay: float = 0.970, forward: float = 0.025, backward: float = 0.005, strength: float = 1.0) -> np.ndarray:
    if strength == 0:
        return np.eye(5)
    matrix = np.zeros((5, 5), dtype=float)
    for state in range(5):
        matrix[state, state] += stay
        if state < 4:
            matrix[state, state + 1] += forward
        if state > 0:
            matrix[state, state - 1] += backward
        matrix[state] /= matrix[state].sum()
    return (1 - strength) * np.eye(5) + strength * matrix


def time_only_hmm(length: int, transition: np.ndarray) -> np.ndarray:
    """A deliberately signal-free reference baseline."""
    posterior = np.zeros(5, dtype=float)
    posterior[0] = 1.0
    rows = []
    for _ in range(length):
        posterior = posterior @ transition
        rows.append(posterior.copy())
    return np.asarray(rows)


def _entropy(probability: np.ndarray) -> float:
    return float(-np.sum(probability * np.log(np.clip(probability, 1e-8, 1))))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    mid = (p + q) / 2
    return float(0.5 * np.sum(p * np.log(np.clip(p / mid, 1e-8, None))) + 0.5 * np.sum(q * np.log(np.clip(q / mid, 1e-8, None))))


def _softmax_distance(embedding: np.ndarray, prototypes: np.ndarray, variance: np.ndarray) -> np.ndarray:
    distances = np.sum((embedding[None, :] - prototypes) ** 2 / np.maximum(variance, 1e-5), axis=1)
    logits = -0.5 * distances
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def _state_compatible(previous: int | None, current: int) -> bool:
    return previous is None or abs(current - previous) <= 1


@dataclass(frozen=True)
class AblationFlags:
    hmm: bool
    dynamic_prototype: bool
    memory: bool
    neural_update: bool
    residual_adapter: bool
    tent_lite: bool


def ablation_flags(name: str) -> AblationFlags:
    return {
        "B0_STATIC_SOURCE": AblationFlags(False, False, False, False, False, False),
        "B1_TIME_ONLY_HMM": AblationFlags(True, False, False, False, False, False),
        "B2_STATIC_HMM": AblationFlags(True, False, False, False, False, False),
        "B3_DYNAMIC_PROTOTYPE": AblationFlags(True, True, False, False, False, False),
        "B4_TEACHER_MEMORY": AblationFlags(True, True, True, False, False, False),
        "B5_TEMPORAL_RANKING": AblationFlags(True, True, True, True, False, False),
        "B6_FULL_ADAPTATION": AblationFlags(True, True, True, True, True, False),
        "B7_TENT_LITE": AblationFlags(True, False, True, True, False, True),
    }[name]


class OnlineRunner:
    def __init__(self, source: SourceBundle, config: TemporalPrototypeConfig, ablation: str, transition_strength: float = 1.0) -> None:
        self.source = source
        self.config = config
        self.ablation = ablation
        self.flags = ablation_flags(ablation)
        self.student = copy.deepcopy(source.model).eval()
        self.teacher = copy.deepcopy(source.model).eval()
        self.source_model = copy.deepcopy(source.model).eval()
        for p in self.source_model.parameters():
            p.requires_grad = False
        self.source_state = copy.deepcopy(source.model.state_dict())
        self.prototypes = source.prototypes.copy()
        self.variance = source.variances.copy()
        self.support = np.zeros(5, dtype=int)
        self.transition = transition_matrix(config.transition_stay, config.transition_forward, config.transition_backward, transition_strength)
        self.transition_strength = transition_strength
        self.posterior = np.array([1.0, 0, 0, 0, 0], dtype=float)
        self.previous_state: int | None = None
        self.memory: list[list[dict[str, Any]]] = [[] for _ in range(5)]
        self.freeze_until = -1
        self.accepted_total = 0
        self.update_count = 0
        self.freeze_count = 0
        self.rollback_count = 0
        self.prototype_updates = 0
        self.events: list[dict[str, Any]] = []
        self.checkpoint: dict[str, Any] | None = None
        self.quality_history: list[tuple[float, float]] = []
        self._freeze_trigger_active = False
        self.initial_adapter = [p.detach().clone() for p in self.student.parameters()]
        self.optimizer = self._make_optimizer()

    def _make_optimizer(self) -> torch.optim.Optimizer | None:
        if not self.flags.neural_update:
            return None
        if self.flags.tent_lite:
            params = self.student.online_parameters(tent_lite=True)
        elif self.flags.residual_adapter:
            params = self.student.online_parameters(tent_lite=False)
        else:
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.student.embedding_head.parameters():
                p.requires_grad = True
            self.student.ordinal_bias.requires_grad = True
            params = [p for p in self.student.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.online_learning_rate)

    def _network(self, model: TemporalPrototypeNet, sequence: np.ndarray) -> dict[str, np.ndarray]:
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(sequence[None]).float())
        return {key: value.detach().cpu().numpy()[0] for key, value in out.items()}

    def _network_batch(self, model: TemporalPrototypeNet, sequences: np.ndarray) -> dict[str, np.ndarray]:
        model.eval()
        output: dict[str, list[np.ndarray]] = {"embedding": [], "stage_probs": [], "health": [], "ordinal_logits": []}
        with torch.no_grad():
            for start in range(0, len(sequences), 512):
                batch = model(torch.from_numpy(sequences[start:start + 512]).float())
                for key in output:
                    output[key].append(batch[key].detach().cpu().numpy())
        return {key: np.concatenate(value, axis=0) for key, value in output.items()}

    def _observation(self, output: dict[str, np.ndarray], prototype: bool = True) -> tuple[np.ndarray, np.ndarray]:
        ordinal = output["stage_probs"]
        proto = _softmax_distance(output["embedding"], self.prototypes, self.variance)
        if prototype:
            return (0.5 * ordinal + 0.5 * proto) / 1.0, proto
        return ordinal, proto

    def _update_prototype(self, stage: int, embedding: np.ndarray, confidence: float, index: int) -> bool:
        candidate = self.prototypes.copy()
        eta = min(self.config.prototype_eta_max, self.config.prototype_eta_base * (1 + confidence))
        candidate[stage - 1] = (1 - eta) * candidate[stage - 1] + eta * embedding
        projection = candidate @ self.source.source_axis
        if np.any(np.diff(projection) < -1e-8):
            self.events.append({"window_index": index, "event": "PROTOTYPE_ORDER_REJECT", "stage": stage})
            return False
        drift = float(np.linalg.norm(candidate[stage - 1] - self.source.prototypes[stage - 1]))
        neighbour = float(np.linalg.norm(self.source.prototypes[min(stage, 4)] - self.source.prototypes[max(stage - 2, 0)]))
        if neighbour > 0 and drift > 0.2 * neighbour:
            self.freeze_until = max(self.freeze_until, index + self.config.freeze_windows)
            self.freeze_count += 1
            self.events.append({"window_index": index, "event": "PROTOTYPE_DRIFT_FREEZE", "stage": stage})
            return False
        self.prototypes = candidate
        error = (embedding - self.prototypes[stage - 1]) ** 2
        self.variance[stage - 1] = 0.99 * self.variance[stage - 1] + 0.01 * np.maximum(error, 1e-5)
        self.support[stage - 1] += 1
        self.prototype_updates += 1
        return True

    def _store_memory(self, record: dict[str, Any]) -> None:
        bucket = self.memory[int(record["state"]) - 1]
        bucket.append(record)
        if len(bucket) > self.config.memory_per_state:
            # Keep confident and non-redundant evidence rather than only the most recent entries.
            anchor = self.prototypes[int(record["state"]) - 1]
            score = np.asarray([item["confidence"] - 0.05 * np.linalg.norm(item["embedding"] - anchor) for item in bucket])
            del bucket[int(np.argmin(score))]

    def _augment(self, sequence: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        result = sequence.copy()
        result[:, :10] *= rng.uniform(0.98, 1.02)
        result[:, :10] += rng.normal(0, 0.01, size=result[:, :10].shape)
        result[:, rng.integers(0, 10)] = 0.0
        return result

    def _online_update(self, index: int) -> None:
        if self.optimizer is None or self.accepted_total < self.config.min_memory_to_update or self.accepted_total % self.config.update_every_accepted:
            return
        items = [item for bucket in self.memory for item in bucket]
        if len(items) < self.config.min_memory_to_update:
            return
        items = sorted(items, key=lambda item: item["confidence"], reverse=True)[: min(128, len(items))]
        sequences = torch.from_numpy(np.stack([item["sequence"] for item in items])).float()
        teacher_probs = torch.from_numpy(np.stack([item["teacher_ordinal"] for item in items])).float()
        states = torch.tensor([item["state"] - 1 for item in items], dtype=torch.long)
        augmented = torch.from_numpy(np.stack([self._augment(item["sequence"], index + j) for j, item in enumerate(items)])).float()
        target_proto = torch.from_numpy(self.prototypes[states.numpy()]).float()
        for _ in range(self.config.online_steps):
            self.student.train()
            out = self.student(augmented)
            student_probs = out["stage_probs"]
            consistency = nn.functional.kl_div(torch.log(student_probs.clamp_min(1e-7)), teacher_probs, reduction="batchmean")
            compact = ((out["embedding"] - target_proto) ** 2).sum(dim=1).mean()
            entropy = -(student_probs * torch.log(student_probs.clamp_min(1e-7))).sum(dim=1).mean()
            source_anchor = torch.zeros((), dtype=torch.float32)
            if self.flags.residual_adapter:
                for current, initial in zip(self.student.parameters(), self.initial_adapter):
                    if current.requires_grad:
                        source_anchor = source_anchor + ((current - initial) ** 2).mean()
            ranking = torch.zeros((), dtype=torch.float32)
            if not self.flags.tent_lite and len(items) > 1:
                order = torch.tensor([item["window_index"] for item in items])
                expected = student_probs @ torch.arange(1, 6, dtype=torch.float32)
                pairs = (order[:, None] - order[None, :]) >= 50
                if pairs.any():
                    late, early = torch.where(pairs)
                    ranking = torch.relu(0.05 - expected[late] + expected[early]).mean()
            loss = entropy if self.flags.tent_lite else consistency + 0.5 * compact + 0.2 * ranking + 0.2 * source_anchor + 0.1 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.student.eval()
        with torch.no_grad():
            for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
                teacher.mul_(self.config.teacher_decay).add_(student * (1 - self.config.teacher_decay))
        self.update_count += 1
        self.events.append({"window_index": index, "event": "ONLINE_UPDATE", "accepted_total": self.accepted_total})

    def _snapshot_payload(self) -> dict[str, Any]:
        return {
            "student": copy.deepcopy(self.student.state_dict()), "teacher": copy.deepcopy(self.teacher.state_dict()),
            "prototypes": self.prototypes.copy(), "variance": self.variance.copy(), "support": self.support.copy(),
            "accepted_total": self.accepted_total, "update_count": self.update_count,
        }

    def restore_adaptation(self, payload: dict[str, Any]) -> None:
        self.student.load_state_dict(payload["student"])
        self.teacher.load_state_dict(payload["teacher"])
        self.prototypes = np.asarray(payload["prototypes"]).copy()
        self.variance = np.asarray(payload["variance"]).copy()
        self.support = np.asarray(payload["support"]).copy()
        self.accepted_total = int(payload.get("accepted_total", 0))
        self.update_count = int(payload.get("update_count", 0))

    def _checkpoint(self, index: int) -> None:
        tail = self.quality_history[-20:]
        quality = (float(np.mean([x[0] for x in tail])) if tail else 0.0, float(np.mean([x[1] for x in tail])) if tail else 1.0)
        self.checkpoint = {"payload": self._snapshot_payload(), "quality": quality, "index": index}
        self.events.append({"window_index": index, "event": "CHECKPOINT"})

    def _rollback_check(self, index: int) -> None:
        if self.checkpoint is None or index - self.checkpoint["index"] < 100 or len(self.quality_history) < 100:
            return
        tail = self.quality_history[-100:]
        agreement, entropy = float(np.mean([x[0] for x in tail])), float(np.mean([x[1] for x in tail]))
        old_agreement, old_entropy = self.checkpoint["quality"]
        ordered = not np.any(np.diff(self.prototypes @ self.source.source_axis) < -1e-8)
        if entropy > old_entropy and agreement < old_agreement and not ordered:
            self.restore_adaptation(self.checkpoint["payload"])
            self.freeze_until = index + self.config.freeze_windows
            self.rollback_count += 1
            self.events.append({"window_index": index, "event": "ROLLBACK_UNSUPERVISED"})

    def run(self, target: pd.DataFrame, snapshot_dir: Path | None = None, save_snapshots: bool = False, permit_updates: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        reject_target_labels(target)
        values, missing = self.source.scaler.transform(target, self.config)
        sequences = causal_sequences(values, target["restart_mask"].to_numpy(bool), self.config.sequence_length)
        snapshots = {int(round(fraction * (len(target) - 1))): fraction for fraction in self.config.snapshot_fractions}
        rows, prototype_rows, memory_rows = [], [], []
        time_only = time_only_hmm(len(target), self.transition) if self.ablation == "B1_TIME_ONLY_HMM" else None
        # Per-window inference deliberately preserves bitwise prefix equivalence.  Batched GRU
        # kernels can differ by a few ulps when a later suffix changes the batch shape.
        cached_student = None
        cached_teacher = None
        for index, meta in target.reset_index(drop=True).iterrows():
            if save_snapshots and index in snapshots and snapshot_dir is not None:
                fraction = snapshots[index]
                torch.save(self._snapshot_payload(), snapshot_dir / f"{self.ablation}_{fraction:.1f}.pt")
            if self.ablation == "B1_TIME_ONLY_HMM":
                q = time_only[index]
                student = {"embedding": np.zeros(16), "stage_probs": q, "health": 0.0, "ordinal_logits": np.zeros(4)}
                teacher = student
                proto = q
                observation = q
            else:
                student = ({key: value[index] for key, value in cached_student.items()} if cached_student is not None else self._network(self.student, sequences[index]))
                teacher = ({key: value[index] for key, value in cached_teacher.items()} if cached_teacher is not None else self._network(self.teacher, sequences[index]))
                use_proto = self.ablation != "B0_STATIC_SOURCE"
                observation, proto = self._observation(student, use_proto)
                q = observation if (not self.flags.hmm or self.transition_strength == 0) else observation * (self.posterior @ self.transition)
                q = q / q.sum()
            predicted_stage = int(q.argmax() + 1)
            teacher_stage = int(teacher["stage_probs"].argmax() + 1)
            student_stage = int(student["stage_probs"].argmax() + 1)
            confidence = float(q.max())
            entropy = _entropy(q)
            margin = float(np.partition(q, -2)[-1] - np.partition(q, -2)[-2])
            agreement = teacher_stage == student_stage
            js = _js(teacher["stage_probs"], student["stage_probs"])
            guard = bool(meta["is_restart_guard"])
            frozen = index <= self.freeze_until
            allowed = (
                confidence >= self.config.confidence_threshold and agreement and js <= self.config.js_threshold
                and margin >= self.config.posterior_margin_threshold and entropy <= self.config.entropy_threshold
                and not guard and float(meta["TES"]) <= self.source.tes_p99 and missing[index] == 0
                and _state_compatible(self.previous_state, predicted_stage) and not frozen
            )
            reasons = []
            if not allowed:
                checks = {"LOW_CONFIDENCE": confidence >= self.config.confidence_threshold, "DISAGREEMENT": agreement,
                          "HIGH_JS": js <= self.config.js_threshold, "LOW_MARGIN": margin >= self.config.posterior_margin_threshold,
                          "HIGH_ENTROPY": entropy <= self.config.entropy_threshold, "RESTART_GUARD": not guard,
                          "TES_OUTLIER": float(meta["TES"]) <= self.source.tes_p99, "MISSING": missing[index] == 0,
                          "STATE_JUMP": _state_compatible(self.previous_state, predicted_stage), "FROZEN": not frozen}
                reasons = [name for name, passed in checks.items() if not passed]
            prototype_updated = False
            # The row above is the immutable pre-update prediction for this window.
            if allowed and self.flags.dynamic_prototype and permit_updates:
                prototype_updated = self._update_prototype(predicted_stage, student["embedding"], confidence, index)
            record = {"state": predicted_stage, "embedding": student["embedding"].copy(), "teacher_ordinal": teacher["stage_probs"].copy(),
                      "student_ordinal": student["stage_probs"].copy(), "posterior": q.copy(), "health": float(student["health"]),
                      "confidence": confidence, "entropy": entropy, "window_index": int(meta["window_index"]),
                      "center_cycle": float(meta["center_cycle"]), "restart": guard, "TES": float(meta["TES"]), "sequence": sequences[index].copy()}
            if allowed and self.flags.memory and permit_updates:
                self._store_memory(record)
                self.accepted_total += 1
                self._online_update(index)
            elif not allowed:
                self.events.append({"window_index": index, "event": "PSEUDO_REJECT", "reasons": ";".join(reasons)})
            freeze_trigger = guard or float(meta["TES"]) > self.source.tes_p99
            # A 500-cycle boundary arrives every 100 windows.  Do not restart a still-active
            # 100-window freeze on its final guarded window, or adaptation would be frozen forever.
            if freeze_trigger and not self._freeze_trigger_active and index > self.freeze_until:
                self.freeze_until = max(self.freeze_until, index + self.config.freeze_windows)
                self.freeze_count += 1
                self.events.append({"window_index": index, "event": "FREEZE_GUARD_OR_TES"})
            self._freeze_trigger_active = freeze_trigger
            self.posterior = q.copy()
            self.previous_state = predicted_stage
            self.quality_history.append((float(agreement), entropy))
            if permit_updates and index > 0 and index % self.config.checkpoint_interval == 0:
                self._checkpoint(index)
            if permit_updates:
                self._rollback_check(index)
            rows.append({
                "window_index": int(meta["window_index"]), "start_cycle": float(meta["start_cycle"]), "end_cycle": float(meta["end_cycle"]),
                "center_cycle": float(meta["center_cycle"]), "predicted_stage": predicted_stage, "posterior_confidence": confidence,
                "posterior_entropy": entropy, "posterior_margin": margin, "teacher_student_agreement": int(agreement), "teacher_student_js": js,
                "neural_health_score": float(student["health"]), "health_score_state": float(q @ np.linspace(0, 1, 5)),
                "final_health_score": float(0.5 * student["health"] + 0.5 * (q @ np.linspace(0, 1, 5))), "accepted": int(allowed),
                "prototype_updated": int(prototype_updated), "restart_guard": int(guard), "missing_fraction": float(missing[index]),
                "freeze_active": int(frozen), "ablation": self.ablation,
                **{f"stage_posterior_{i + 1}": float(q[i]) for i in range(5)}, **{f"embedding_{i + 1}": float(student["embedding"][i]) for i in range(16)},
            })
            if self.ablation in {"B3_DYNAMIC_PROTOTYPE", "B4_TEACHER_MEMORY", "B5_TEMPORAL_RANKING", "B6_FULL_ADAPTATION"}:
                for state in range(5):
                    prototype_rows.append({"window_index": int(meta["window_index"]), "ablation": self.ablation, "state": state + 1,
                                           "support": int(self.support[state]), "drift": float(np.linalg.norm(self.prototypes[state] - self.source.prototypes[state])),
                                           **{f"embedding_{j + 1}": float(self.prototypes[state, j]) for j in range(16)}})
        for stage, bucket in enumerate(self.memory, 1):
            for item in bucket:
                memory_rows.append({"ablation": self.ablation, "memory_stage": stage, "window_index": item["window_index"], "confidence": item["confidence"], "entropy": item["entropy"], "TES": item["TES"]})
        if save_snapshots and snapshot_dir is not None:
            # The terminal snapshot represents all unlabelled target evidence, including its final update.
            torch.save(self._snapshot_payload(), snapshot_dir / f"{self.ablation}_1.0.pt")
        prototype_columns = ["window_index", "ablation", "state", "support", "drift"] + [f"embedding_{j + 1}" for j in range(16)]
        memory_columns = ["ablation", "memory_stage", "window_index", "confidence", "entropy", "TES"]
        event_columns = ["window_index", "event", "reasons", "stage", "accepted_total"]
        return (pd.DataFrame(rows), pd.DataFrame(prototype_rows, columns=prototype_columns),
                pd.DataFrame(memory_rows, columns=memory_columns), pd.DataFrame(self.events, columns=event_columns))
