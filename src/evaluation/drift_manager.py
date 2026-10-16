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


@dataclass
class RecoveryRecord:
    """Outcome of one model's recovery tracking for one drift event.

    ``n_recovery`` semantics (PLAN.md Sec. 5):
    - If the model's rolling accuracy never dips below
      ``pre_drift_accuracy - recovery_epsilon`` within ``recovery_horizon`` events,
      the model never degraded: ``never_degraded=True`` and ``n_recovery=0``.
    - Otherwise ``n_recovery`` counts events from the drift alert until rolling
      accuracy re-entered the band, confirmed by ``recovery_sustain_events``
      consecutive qualifying events (so one lucky window edge does not count).
    - A still-active recovery censored by a subsequent drift is recorded with
      ``completed=False`` and excluded from means.
    """

    model_name: str
    drift_onset_event_id: int
    target_accuracy: float
    has_dipped: bool = False
    never_degraded: bool = False
    completed: bool = False
    first_hit_event_id: Optional[int] = None
    n_recovery: Optional[int] = None


class RecoveryTracker:
    """Per-model drift recovery tracker implementing the N_recovery definition.

    Tracks multiple concurrent drift episodes (a new drift censors any recovery
    still in flight for the same model).
    """

    def __init__(
        self,
        model_names: List[str],
        recovery_epsilon: float = 0.02,
        sustain_events: int = 20,
        recovery_horizon: int = 300,
    ) -> None:
        self.model_names = list(model_names)
        self.recovery_epsilon = recovery_epsilon
        self.sustain_events = sustain_events
        self.recovery_horizon = recovery_horizon
        self.records: List[RecoveryRecord] = []
        self._active: Dict[str, RecoveryRecord] = {}
        self._sustain_counter: Dict[str, int] = {m: 0 for m in model_names}

    def register_drift(
        self, model_name: str, event_id: int, pre_drift_accuracy: float
    ) -> None:
        """Open a recovery episode for one model at its drift alert."""
        prior = self._active.pop(model_name, None)
        if prior is not None:
            # Drift arrived before recovery resolved: censored, excluded from means.
            prior.completed = False
            self.records.append(prior)
        self._active[model_name] = RecoveryRecord(
            model_name=model_name,
            drift_onset_event_id=event_id,
            target_accuracy=max(0.0, pre_drift_accuracy - self.recovery_epsilon),
        )
        self._sustain_counter[model_name] = 0

    def observe_accuracies(
        self, event_id: int, current_rolling_accuracies: Dict[str, float]
    ) -> None:
        """Advance all active recovery episodes with the current rolling accuracies."""
        for m in list(self._active.keys()):
            rec = self._active[m]
            elapsed = event_id - rec.drift_onset_event_id
            curr_acc = current_rolling_accuracies.get(m, 0.0)

            if not rec.has_dipped:
                if curr_acc < rec.target_accuracy:
                    rec.has_dipped = True
                elif elapsed >= self.recovery_horizon:
                    rec.never_degraded = True
                    rec.completed = True
                    rec.n_recovery = 0
                    self.records.append(self._active.pop(m))
                    continue

            if rec.has_dipped and not rec.completed:
                # Model has dipped; wait for the accuracy band to be re-entered.
                if curr_acc >= rec.target_accuracy:
                    if self._sustain_counter[m] == 0:
                        rec.first_hit_event_id = event_id
                    self._sustain_counter[m] += 1
                    if self._sustain_counter[m] >= self.sustain_events:
                        rec.completed = True
                        rec.n_recovery = rec.first_hit_event_id - rec.drift_onset_event_id
                        self.records.append(self._active.pop(m))
                else:
                    self._sustain_counter[m] = 0
                    rec.first_hit_event_id = None

    def completed_recoveries(self, model_name: str) -> List[RecoveryRecord]:
        """Return completed recovery episodes for a model (never-degraded included)."""
        return [r for r in self.records if r.model_name == model_name and r.completed]

    def mean_recovery_events(
        self, model_name: str, degraded_only: bool = False
    ) -> float:
        """Mean N_recovery for a model (0.0 if no completed episodes)."""
        recs = self.completed_recoveries(model_name)
        if degraded_only:
            recs = [r for r in recs if not r.never_degraded]
        vals = [r.n_recovery for r in recs if r.n_recovery is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0


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
                triggered_models=triggered_models
                or [
                    "M4_Quantum_Adaptive_VQC",
                    "M6_Quantum_Adaptive_QSVM",
                    "M8_Control_RBF_Adaptive",
                ],
            )
            self.drift_history.append(record)
            return record
        return None
