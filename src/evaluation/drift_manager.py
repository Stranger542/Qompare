# src/evaluation/drift_manager.py
"""ADWIN drift detection coordinator and drift recovery rate tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from river import drift


@dataclass
class DriftEventRecord:
    """Metadata recorded whenever ADWIN detects a statistically significant concept drift."""

    event_id: int
    timestamp: str
    residual_mean_before: float
    residual_mean_after: float
    magnitude: float
    triggered_models: List[str] = field(default_factory=list)


class RecoveryTracker:
    """Tracks how many events (N_recovery) each model takes to recover baseline accuracy after drift."""

    def __init__(self, model_names: List[str], recovery_ratio: float = 0.95) -> None:
        self.model_names = list(model_names)
        self.recovery_ratio = recovery_ratio
        self._active_drift_event_id: Dict[str, Optional[int]] = {
            m: None for m in model_names
        }
        self._target_accuracy: Dict[str, float] = {m: 0.0 for m in model_names}
        self.completed_recoveries: Dict[str, List[int]] = {
            m: [] for m in model_names
        }

    def register_drift(
        self, event_id: int, pre_drift_accuracies: Dict[str, float]
    ) -> None:
        """Mark the onset of a drift event and record each model's pre-drift accuracy target."""
        for m in self.model_names:
            base_acc = pre_drift_accuracies.get(m, 0.5)
            self._active_drift_event_id[m] = event_id
            self._target_accuracy[m] = max(0.5, base_acc * self.recovery_ratio)

    def observe_accuracies(
        self, event_id: int, current_rolling_accuracies: Dict[str, float], min_Wait: int = 5
    ) -> None:
        """Check if models awaiting recovery have regained their target rolling accuracy."""
        for m in self.model_names:
            drift_id = self._active_drift_event_id.get(m)
            if drift_id is not None and (event_id - drift_id) >= min_Wait:
                curr_acc = current_rolling_accuracies.get(m, 0.0)
                if curr_acc >= self._target_accuracy[m]:
                    self.completed_recoveries[m].append(event_id - drift_id)
                    self._active_drift_event_id[m] = None

    def mean_recovery_events(self, model_name: str) -> float:
        """Return average N_recovery for a given model (or 0.0 if no recoveries completed)."""
        recs = self.completed_recoveries.get(model_name, [])
        return float(sum(recs) / len(recs)) if recs else 0.0


class ADWINDriftManager:
    """Monitors streaming prequential residuals with River's ADWIN detector."""

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self.detector = drift.ADWIN(delta=delta)
        self.drift_history: List[DriftEventRecord] = []

    @property
    def cumulative_drifts(self) -> int:
        return len(self.drift_history)

    def update_and_check(
        self,
        event_id: int,
        timestamp: str,
        y_true: int,
        y_pred_reference: int,
        triggered_models: Optional[List[str]] = None,
    ) -> Optional[DriftEventRecord]:
        """Update ADWIN with prequential error e_t = |y_t - y_pred| and return DriftEventRecord on alert."""
        prev_estimation = float(self.detector.estimation)
        residual = float(abs(int(y_true) - int(y_pred_reference)))
        self.detector.update(residual)

        if self.detector.drift_detected:
            new_estimation = float(self.detector.estimation)
            magnitude = abs(new_estimation - prev_estimation)
            record = DriftEventRecord(
                event_id=event_id,
                timestamp=timestamp,
                residual_mean_before=prev_estimation,
                residual_mean_after=new_estimation,
                magnitude=magnitude,
                triggered_models=triggered_models or ["M4_Quantum_Adaptive_VQC", "M6_Quantum_Adaptive_QSVM"],
            )
            self.drift_history.append(record)
            return record
        return None

    