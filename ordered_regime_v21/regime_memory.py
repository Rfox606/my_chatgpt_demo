from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trajectory import trimmed_mean


@dataclass
class StateMemory:
    stage: int
    level_proto: np.ndarray
    trajectory_proto: np.ndarray
    level_radius: float
    trajectory_radius: float
    records: list[dict] = field(default_factory=list)
    updates: int = 0

    @property
    def support(self) -> int:
        return len(self.records)

    def distances(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.records:
            return np.asarray([self.level_radius]), np.asarray([self.trajectory_radius])
        level = np.stack([record["level"] for record in self.records])
        traj = np.stack([record["trajectory"] for record in self.records])
        return np.linalg.norm(level - self.level_proto, axis=1), np.linalg.norm(traj - self.trajectory_proto, axis=1)

    def add(self, record: dict, capacity: int, every: int, robust: bool) -> bool:
        self.records.append(record)
        if len(self.records) > capacity:
            # Prefer confident, non-redundant evidence instead of a last-in FIFO policy.
            score = np.asarray([item["confidence"] - .02 * np.linalg.norm(item["level"] - self.level_proto) for item in self.records])
            del self.records[int(np.argmin(score))]
        if len(self.records) % every:
            return False
        level = np.stack([item["level"] for item in self.records]); trajectory = np.stack([item["trajectory"] for item in self.records])
        self.level_proto = trimmed_mean(level) if robust else level.mean(axis=0)
        self.trajectory_proto = trimmed_mean(trajectory) if robust else trajectory.mean(axis=0)
        ld = np.linalg.norm(level - self.level_proto, axis=1); td = np.linalg.norm(trajectory - self.trajectory_proto, axis=1)
        self.level_radius = float(np.percentile(ld, 95)); self.trajectory_radius = float(np.percentile(td, 95)); self.updates += 1
        return True
