# src/evaluation/prequential_eval.py
"""Test-then-train Prequential Evaluation Engine for the 4x2 Comparative Model Matrix."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from src.config import PipelineConfig
from src.evaluation.drift_manager import ADWINDriftManager, DriftEventRecord, RecoveryTracker
from src.models.classical_models import AdaptiveHoeffdingTreeModel, StaticHoeffdingTreeModel
from src.models.qsvm_models import (
    DriftAdaptiveQuantumKernelModel,
    DriftAdaptiveRBFKernelModel,
    StaticQuantumKernelModel,
    StaticRBFKernelModel,
)
from src.models.vqc_models import DriftAdaptiveVQCModel, StaticVQCModel
from src.worker.feature_worker import ProcessedStreamEvent, StreamFeatureWorker


@dataclass(frozen=True)
class ModelMetricsSnapshot:
    """Rolling and cumulative prequential metrics + latency breakdown for one model at event t."""

    model_name: str
    y_true: int
    y_pred: int
    prob_pos: float
    cumulative_accuracy: float
    cumulative_accuracy_post_warmup: float
    rolling_accuracy: float
    rolling_precision: float
    rolling_recall: float
    rolling_f1: float
    t_encode_ms: float
    t_sim_ms: float
    t_update_ms: float
    t_total_ms: float
    n_updates: int


@dataclass(frozen=True)
class EventEvaluationResult:
    """Complete telemetry bundle emitted for a single processed streaming event."""

    event_id: int
    timestamp: str
    y_true: int
    t_network_ms: float
    t_prep_ms: float
    model_snapshots: Dict[str, ModelMetricsSnapshot]
    drift_events: List[DriftEventRecord] = field(default_factory=list)
    cumulative_drifts: int = 0
    pair_agreements: Dict[str, float] = field(default_factory=dict)
    adaptation_diagnostics: Dict[str, float] = field(default_factory=dict)


class RollingClassificationMetrics:
    """Bounded rolling window calculator for accuracy, precision, recall, and F1."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._pairs: Deque[tuple[int, int]] = deque(maxlen=window_size)
        self.total_seen: int = 0
        self.total_correct: int = 0

    def update(self, y_true: int, y_pred: int) -> None:
        y_t = int(y_true)
        y_p = int(y_pred)
        self._pairs.append((y_t, y_p))
        self.total_seen += 1
        if y_t == y_p:
            self.total_correct += 1

    @property
    def cumulative_accuracy(self) -> float:
        return self.total_correct / self.total_seen if self.total_seen > 0 else 0.0

    def compute_rolling(self) -> tuple[float, float, float, float]:
        """Return (rolling_accuracy, rolling_precision, rolling_recall, rolling_f1)."""
        if not self._pairs:
            return 0.0, 0.0, 0.0, 0.0
        tp = fp = fn = correct = 0
        for y_t, y_p in self._pairs:
            if y_t == y_p:
                correct += 1
            if y_t == 1 and y_p == 1:
                tp += 1
            elif y_t == 0 and y_p == 1:
                fp += 1
            elif y_t == 1 and y_p == 0:
                fn += 1
        acc = correct / len(self._pairs)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return float(acc), float(precision), float(recall), float(f1)


class PrequentialEvaluator:
    """Orchestrates strict test-then-train evaluation across the 4x2 model matrix (M1-M8).

    Protocol per event (PLAN.md Sec. 10.5):
      1. Forward inference for all models (labels withheld).
      2. Reveal label, update rolling metrics, append to buffers.
      3. Model updates (M1/M2 incremental; M3-M8 initial fit at W_0; M4/M6/M8
         adaptation on their own ADWIN drift alerts).
    """

    def __init__(self, config: PipelineConfig, worker: StreamFeatureWorker) -> None:
        self.config = config
        self.worker = worker

        # Instantiate 4x2 Comparative Model Matrix
        self.m1 = StaticHoeffdingTreeModel(warmup_window_size=config.initial_warmup_window)
        self.m2 = AdaptiveHoeffdingTreeModel(
            adwin_delta=config.adwin_delta, seed=config.random_seed
        )
        self.m3 = StaticVQCModel(
            num_qubits=config.num_qubits,
            ansatz_reps=config.vqc_ansatz_reps,
            initial_spsa_steps=config.vqc_initial_spsa_steps,
            adapt_spsa_steps=config.vqc_adapt_spsa_steps,
            spsa_lr=config.vqc_spsa_lr,
            spsa_perturbation=config.vqc_spsa_perturbation,
            adapt_lr_scale=config.vqc_adapt_lr_scale,
            adapt_batch_size=config.vqc_adapt_batch_size,
            readout=config.vqc_readout,
            seed=config.random_seed,
        )
        self.m4 = DriftAdaptiveVQCModel(
            num_qubits=config.num_qubits,
            ansatz_reps=config.vqc_ansatz_reps,
            initial_spsa_steps=config.vqc_initial_spsa_steps,
            adapt_spsa_steps=config.vqc_adapt_spsa_steps,
            spsa_lr=config.vqc_spsa_lr,
            spsa_perturbation=config.vqc_spsa_perturbation,
            adapt_lr_scale=config.vqc_adapt_lr_scale,
            adapt_batch_size=config.vqc_adapt_batch_size,
            readout=config.vqc_readout,
            seed=config.random_seed,
        )
        self.m5 = StaticQuantumKernelModel(
            num_qubits=config.num_qubits,
            max_support_size=config.qsvm_max_support_size,
            replace_ratio=config.qsvm_replace_ratio,
            svm_c=config.qsvm_c,
            seed=config.random_seed,
        )
        self.m6 = DriftAdaptiveQuantumKernelModel(
            num_qubits=config.num_qubits,
            max_support_size=config.qsvm_max_support_size,
            replace_ratio=config.qsvm_replace_ratio,
            svm_c=config.qsvm_c,
            seed=config.random_seed,
        )
        self.m7 = StaticRBFKernelModel(
            max_support_size=config.qsvm_max_support_size,
            replace_ratio=config.qsvm_replace_ratio,
            svm_c=config.rbf_c,
            svm_gamma=config.rbf_gamma,
        )
        self.m8 = DriftAdaptiveRBFKernelModel(
            max_support_size=config.qsvm_max_support_size,
            replace_ratio=config.qsvm_replace_ratio,
            svm_c=config.rbf_c,
            svm_gamma=config.rbf_gamma,
        )

        self.model_names: List[str] = [
            self.m1.name,
            self.m2.name,
            self.m3.name,
            self.m4.name,
            self.m5.name,
            self.m6.name,
            self.m7.name,
            self.m8.name,
        ]
        self.metrics: Dict[str, RollingClassificationMetrics] = {
            name: RollingClassificationMetrics(window_size=config.rolling_metric_window)
            for name in self.model_names
        }
        self.update_counts: Dict[str, int] = {name: 0 for name in self.model_names}

        # One independent ADWIN detector per adaptive model, on its own prequential error.
        self.adaptive_models = {
            self.m4.name: self.m4,
            self.m6.name: self.m6,
            self.m8.name: self.m8,
        }
        self.drift_managers: Dict[str, ADWINDriftManager] = {
            name: ADWINDriftManager(delta=config.adwin_delta) for name in self.adaptive_models
        }
        self.recovery_tracker = RecoveryTracker(
            self.model_names,
            recovery_epsilon=config.recovery_epsilon,
            sustain_events=config.recovery_sustain_events,
            recovery_horizon=config.recovery_horizon,
        )

        self._pair_windows: Dict[str, Deque[int]] = {
            "M2_M4": deque(maxlen=config.rolling_metric_window),
            "M3_M4": deque(maxlen=config.rolling_metric_window),
            "M5_M6": deque(maxlen=config.rolling_metric_window),
            "M7_M8": deque(maxlen=config.rolling_metric_window),
        }
        self._label_counts: List[int] = [0, 0]
        self._post_warmup_seen: Dict[str, int] = {name: 0 for name in self.model_names}
        self._post_warmup_correct: Dict[str, int] = {name: 0 for name in self.model_names}
        self._quantum_initialized: bool = False

    def _majority_label(self) -> int:
        return 1 if self._label_counts[1] > self._label_counts[0] else 0

    def _record_pair_agreement(self, key: str, a: int, b: int) -> None:
        self._pair_windows[key].append(1 if a == b else 0)

    def _pair_agreement(self, key: str) -> float:
        window = self._pair_windows[key]
        return float(sum(window) / len(window)) if window else 1.0

    def evaluate_event(self, event: ProcessedStreamEvent) -> EventEvaluationResult:
        """Execute 1) Forward Inference -> 2) Metric Update -> 3) Drift Tests & Model Updates."""
        x_dict = event.features_dict
        x_arr = event.features_array
        y_true = event.label
        post_warmup = event.event_id >= self.config.initial_warmup_window

        # =====================================================================
        # STEP 1: PREQUENTIAL FORWARD INFERENCE (BEFORE LABEL IS REVEALED)
        # =====================================================================
        # Uninitialized quantum/kernel models predict the running majority class
        # (warmup policy, PLAN.md Sec. 11) instead of a hardcoded constant.
        maj = self._majority_label()
        p1, prob1, sim1 = self.m1.predict(x_dict)
        p2, prob2, sim2 = self.m2.predict(x_dict)
        p3, prob3, enc3, sim3 = self.m3.predict(x_arr)
        p4, prob4, enc4, sim4 = self.m4.predict(x_arr)
        p5, prob5, enc5, sim5 = self.m5.predict(x_arr)
        p6, prob6, enc6, sim6 = self.m6.predict(x_arr)
        p7, prob7, t7 = self.m7.predict(x_arr)
        p8, prob8, t8 = self.m8.predict(x_arr)
        enc7 = enc8 = 0.0
        sim7, sim8 = t7, t8

        if not self.m3.is_initialized:
            p3, prob3 = maj, float(maj)
        if not self.m4.is_initialized:
            p4, prob4 = maj, float(maj)
        if not self.m5.is_initialized:
            p5, prob5 = maj, float(maj)
        if not self.m6.is_initialized:
            p6, prob6 = maj, float(maj)
        if not self.m7.is_initialized:
            p7, prob7 = maj, float(maj)
        if not self.m8.is_initialized:
            p8, prob8 = maj, float(maj)

        preds = {
            self.m1.name: (p1, prob1, 0.0, sim1),
            self.m2.name: (p2, prob2, 0.0, sim2),
            self.m3.name: (p3, prob3, enc3, sim3),
            self.m4.name: (p4, prob4, enc4, sim4),
            self.m5.name: (p5, prob5, enc5, sim5),
            self.m6.name: (p6, prob6, enc6, sim6),
            self.m7.name: (p7, prob7, enc7, sim7),
            self.m8.name: (p8, prob8, enc8, sim8),
        }

        # =====================================================================
        # STEP 2: REVEAL LABEL & UPDATE PREQUENTIAL ROLLING METRICS
        # =====================================================================
        pre_drift_acc: Dict[str, float] = {}
        for m_name, (pred_label, _, _, _) in preds.items():
            pre_drift_acc[m_name] = self.metrics[m_name].compute_rolling()[0]
            self.metrics[m_name].update(y_true, pred_label)
            if post_warmup:
                self._post_warmup_seen[m_name] += 1
                if pred_label == y_true:
                    self._post_warmup_correct[m_name] += 1

        self._label_counts[y_true] += 1
        self._record_pair_agreement("M2_M4", p2, p4)
        self._record_pair_agreement("M3_M4", p3, p4)
        self._record_pair_agreement("M5_M6", p5, p6)
        self._record_pair_agreement("M7_M8", p7, p8)

        # Append current labeled event to sliding & warmup windows now that prediction is done
        self.worker.record_post_inference(event)

        # =====================================================================
        # STEP 3: DRIFT DETECTION & PERMITTED MODEL UPDATES
        # =====================================================================
        upd1 = self.m1.update(x_dict, y_true)
        self.update_counts[self.m1.name] += 1
        upd2 = self.m2.update(x_dict, y_true)
        self.update_counts[self.m2.name] += 1
        upd3 = upd4 = upd5 = upd6 = upd7 = upd8 = 0.0

        # Check if initial warmup window W_0 has just filled to initialize M3-M8
        if (
            not self._quantum_initialized
            and len(self.worker.warmup_buffer) >= self.config.initial_warmup_window
        ):
            X_w0, y_w0 = self.worker.warmup_buffer.get_arrays()
            upd3 = self.m3.fit_initial(X_w0, y_w0)
            # Share initial weights between M3 and M4 for an identical starting point at end of W_0
            self.m4.weights = self.m3.weights.copy()
            self.m4.is_initialized = True
            upd4 = upd3
            upd5 = self.m5.fit_initial(X_w0, y_w0)
            upd6 = self.m6.fit_initial(X_w0, y_w0)
            upd7 = self.m7.fit_initial(X_w0, y_w0)
            upd8 = self.m8.fit_initial(X_w0, y_w0)
            for name in (
                self.m3.name,
                self.m4.name,
                self.m5.name,
                self.m6.name,
                self.m7.name,
                self.m8.name,
            ):
                self.update_counts[name] += 1
            self._quantum_initialized = True

        # Per-model ADWIN drift detection on each adaptive model's own prequential error
        drift_records: List[DriftEventRecord] = []
        adaptation_diagnostics: Dict[str, float] = {}
        adaptive_preds = {
            self.m4.name: p4,
            self.m6.name: p6,
            self.m8.name: p8,
        }
        update_times = {
            self.m4.name: upd4,
            self.m6.name: upd6,
            self.m8.name: upd8,
        }
        for m_name, model in self.adaptive_models.items():
            drift_record = self.drift_managers[m_name].update_and_check(
                event_id=event.event_id,
                timestamp=event.timestamp,
                y_true=y_true,
                y_pred_reference=adaptive_preds[m_name],
                triggered_models=[m_name],
            )
            if drift_record is not None:
                drift_records.append(drift_record)
                if self._quantum_initialized:
                    # Open recovery episodes for ALL models so static-vs-adaptive
                    # recovery is comparable under the same alert (PLAN.md Hypothesis 2).
                    for m_name_all in self.model_names:
                        self.recovery_tracker.register_drift(
                            m_name_all, event.event_id, pre_drift_acc[m_name_all]
                        )
                    X_wt, y_wt = self.worker.sliding_buffer.get_arrays()
                    update_times[m_name] += model.adapt_on_drift(X_wt, y_wt)
                    self.update_counts[m_name] += 1
                    if m_name == self.m4.name:
                        adaptation_diagnostics["M4_weight_delta_norm"] = (
                            self.m4.last_weight_delta_norm
                        )
                    elif m_name == self.m6.name:
                        adaptation_diagnostics["M6_support_churn"] = self.m6.last_support_churn
                    elif m_name == self.m8.name:
                        adaptation_diagnostics["M8_support_churn"] = self.m8.last_support_churn

        updates = {
            self.m1.name: upd1,
            self.m2.name: upd2,
            self.m3.name: upd3,
            self.m4.name: update_times[self.m4.name],
            self.m5.name: upd5,
            self.m6.name: update_times[self.m6.name],
            self.m7.name: upd7,
            self.m8.name: update_times[self.m8.name],
        }

        # Build per-model telemetry snapshots
        snapshots: Dict[str, ModelMetricsSnapshot] = {}
        current_rolling_acc: Dict[str, float] = {}
        for m_name in self.model_names:
            pred_label, prob_pos, t_enc, t_sim = preds[m_name]
            t_upd = updates[m_name]
            r_acc, r_prec, r_rec, r_f1 = self.metrics[m_name].compute_rolling()
            current_rolling_acc[m_name] = r_acc
            n_seen = self._post_warmup_seen[m_name]
            post_acc = (
                self._post_warmup_correct[m_name] / n_seen if n_seen > 0 else 0.0
            )
            t_tot = event.t_network_ms + event.t_prep_ms + t_enc + t_sim + t_upd

            snapshots[m_name] = ModelMetricsSnapshot(
                model_name=m_name,
                y_true=y_true,
                y_pred=pred_label,
                prob_pos=prob_pos,
                cumulative_accuracy=self.metrics[m_name].cumulative_accuracy,
                cumulative_accuracy_post_warmup=post_acc,
                rolling_accuracy=r_acc,
                rolling_precision=r_prec,
                rolling_recall=r_rec,
                rolling_f1=r_f1,
                t_encode_ms=t_enc,
                t_sim_ms=t_sim,
                t_update_ms=t_upd,
                t_total_ms=t_tot,
                n_updates=self.update_counts[m_name],
            )

        self.recovery_tracker.observe_accuracies(event.event_id, current_rolling_acc)

        return EventEvaluationResult(
            event_id=event.event_id,
            timestamp=event.timestamp,
            y_true=y_true,
            t_network_ms=event.t_network_ms,
            t_prep_ms=event.t_prep_ms,
            model_snapshots=snapshots,
            drift_events=drift_records,
            cumulative_drifts=sum(dm.cumulative_drifts for dm in self.drift_managers.values()),
            pair_agreements={
                "M2_M4": self._pair_agreement("M2_M4"),
                "M3_M4": self._pair_agreement("M3_M4"),
                "M5_M6": self._pair_agreement("M5_M6"),
                "M7_M8": self._pair_agreement("M7_M8"),
            },
            adaptation_diagnostics=adaptation_diagnostics,
        )
