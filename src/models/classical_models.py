# src/models/classical_models.py
"""Classical streaming decision tree models (M1: Static HT, M2: Adaptive HATC + ADWIN)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

from river import drift, tree


@dataclass(frozen=True)
class ModelStepTiming:
    """Fine-grained latency breakdown in milliseconds for a single model operation."""

    t_encode_ms: float
    t_sim_ms: float
    t_update_ms: float

    @property
    def t_total_ms(self) -> float:
        return self.t_encode_ms + self.t_sim_ms + self.t_update_ms


class StaticHoeffdingTreeModel:
    """M1: Classical Static Control (`River` HoeffdingTreeClassifier).

    Trained strictly on the initial window W_0 (default 200 events) and frozen thereafter.
    """

    name: str = "M1_Classical_Static_HT"

    def __init__(self, warmup_window_size: int = 200, grace_period: int = 25) -> None:
        self.warmup_window_size = warmup_window_size
        self.model = tree.HoeffdingTreeClassifier(grace_period=grace_period)
        self.samples_learned: int = 0
        self.is_frozen: bool = False

    def predict(self, features: Dict[str, float]) -> tuple[int, float, float]:
        """Predict binary label {0, 1}, positive probability, and inference latency (ms)."""
        t0 = time.perf_counter_ns()
        proba_map = self.model.predict_proba_one(features)
        prob_pos = float(proba_map.get(1, proba_map.get(True, 0.5)))
        pred = 1 if prob_pos >= 0.5 else 0
        t_sim_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return pred, prob_pos, t_sim_ms

    def update(self, features: Dict[str, float], label: int) -> float:
        """Update tree only if within initial window W_0; otherwise freeze parameters."""
        if self.samples_learned >= self.warmup_window_size:
            self.is_frozen = True
            return 0.0
        t0 = time.perf_counter_ns()
        self.model.learn_one(features, int(label))
        self.samples_learned += 1
        if self.samples_learned >= self.warmup_window_size:
            self.is_frozen = True
        return (time.perf_counter_ns() - t0) / 1_000_000.0


class AdaptiveHoeffdingTreeModel:
    """M2: Classical Adaptive Baseline (`River` HoeffdingAdaptiveTreeClassifier + ADWIN).

    Continuously updates incrementally on every streaming event and adapts subtrees upon drift.
    """

    name: str = "M2_Classical_Adaptive_HATC"

    def __init__(
        self,
        grace_period: int = 25,
        adwin_delta: float = 0.002,
        seed: int = 42,
    ) -> None:
        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period,
            drift_detector=drift.ADWIN(delta=adwin_delta),
            seed=seed,
        )
        self.samples_learned: int = 0

    def predict(self, features: Dict[str, float]) -> tuple[int, float, float]:
        """Predict binary label {0, 1}, positive probability, and inference latency (ms)."""
        t0 = time.perf_counter_ns()
        proba_map = self.model.predict_proba_one(features)
        prob_pos = float(proba_map.get(1, proba_map.get(True, 0.5)))
        pred = 1 if prob_pos >= 0.5 else 0
        t_sim_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return pred, prob_pos, t_sim_ms

    def update(self, features: Dict[str, float], label: int) -> float:
        """Incrementally update the adaptive tree with the arriving labeled event."""
        t0 = time.perf_counter_ns()
        self.model.learn_one(features, int(label))
        self.samples_learned += 1
        return (time.perf_counter_ns() - t0) / 1_000_000.0\

        