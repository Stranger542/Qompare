# tests/test_pipeline.py
"""Unit and integration tests for the Quantum vs Classical Streaming Pipeline."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.config import FEATURE_NAMES, PipelineConfig
from src.evaluation.drift_manager import ADWINDriftManager, RecoveryTracker
from src.evaluation.prequential_eval import PrequentialEvaluator
from src.models.classical_models import AdaptiveHoeffdingTreeModel, StaticHoeffdingTreeModel
from src.models.qsvm_models import (
    DriftAdaptiveQuantumKernelModel,
    DriftAdaptiveRBFKernelModel,
    StaticQuantumKernelModel,
    StaticRBFKernelModel,
)
from src.models.vqc_models import DriftAdaptiveVQCModel, StaticVQCModel
from src.producer.elec2_producer import serialize_elec2_event
from src.sink.influx_sink import InfluxTelemetrySink
from src.worker.feature_worker import (
    OnlineAngleMinMaxScaler,
    SlidingWindowBuffer,
    StreamFeatureWorker,
)


def _sample_raw_features(offset: float = 0.0) -> dict[str, float]:
    return {
        "nswprice": 0.05 + offset,
        "nswdemand": 0.42 + offset,
        "vicprice": 0.03 + offset,
        "vicdemand": 0.40 + offset,
        "transfer": 0.60 - offset,
    }


def test_scaler_bounds_strictly_within_zero_and_pi() -> None:
    """Verify OnlineAngleMinMaxScaler bounds all features strictly within [0, pi]."""
    scaler = OnlineAngleMinMaxScaler(FEATURE_NAMES)
    for idx in range(25):
        feats = _sample_raw_features(offset=idx * 0.02)
        scaled_dict, scaled_arr = scaler.partial_fit_transform(feats)
        assert scaled_arr.shape == (5,)
        assert np.all(scaled_arr >= 0.0)
        assert np.all(scaled_arr <= math.pi)
        for val in scaled_dict.values():
            assert 0.0 <= val <= math.pi


def test_event_serialization_and_edge_cases() -> None:
    """Verify serialize_elec2_event handles valid inputs and rejects missing/invalid fields."""
    valid = serialize_elec2_event(10, _sample_raw_features(), True)
    assert valid["event_id"] == 10
    assert valid["label"] == 1
    assert set(valid["features"].keys()) == set(FEATURE_NAMES)

    with pytest.raises(ValueError, match="Missing required feature"):
        bad_features = _sample_raw_features()
        del bad_features["transfer"]
        serialize_elec2_event(11, bad_features, 0)

    with pytest.raises(ValueError, match="Unexpected label"):
        serialize_elec2_event(12, _sample_raw_features(), "INVALID_LABEL")


def test_worker_deduplication_and_sliding_window() -> None:
    """Verify duplicate event IDs are ignored and SlidingWindowBuffer enforces FIFO bounds."""
    cfg = PipelineConfig(
        initial_warmup_window=10,
        sliding_window_size=10,
        influxdb_enabled=False,
    )
    worker = StreamFeatureWorker(cfg)
    payload = serialize_elec2_event(1, _sample_raw_features(), 1)

    ev1 = worker.process_raw_payload(payload)
    assert ev1 is not None
    # Buffer must still be empty before record_post_inference is called (no label leakage)
    assert len(worker.sliding_buffer) == 0

    worker.record_post_inference(ev1)
    assert len(worker.sliding_buffer) == 1

    # Duplicate event_id=1 must return None
    ev_dup = worker.process_raw_payload(payload)
    assert ev_dup is None


def test_classical_static_vs_adaptive_models() -> None:
    """Verify M1 freezes after warmup_window_size while M2 continues learning."""
    m1 = StaticHoeffdingTreeModel(warmup_window_size=10)
    m2 = AdaptiveHoeffdingTreeModel()

    for i in range(15):
        feats = _sample_raw_features(offset=i * 0.01)
        label = i % 2
        pred1, prob1, _ = m1.predict(feats)
        pred2, prob2, _ = m2.predict(feats)
        assert pred1 in (0, 1) and 0.0 <= prob1 <= 1.0
        assert pred2 in (0, 1) and 0.0 <= prob2 <= 1.0
        t_upd1 = m1.update(feats, label)
        t_upd2 = m2.update(feats, label)
        if i >= 10:
            assert m1.is_frozen is True
            assert t_upd1 == 0.0
        assert t_upd2 > 0.0


def test_qiskit_vqc_static_and_adaptive_predictions() -> None:
    """Verify Qiskit VQC (M3 & M4) produce binary predictions {0, 1} and warm-start adapt."""
    rng = np.random.default_rng(42)
    X_w0 = rng.uniform(0.0, math.pi, size=(12, 5))
    y_w0 = np.array([0, 1] * 6, dtype=np.int64)

    m3 = StaticVQCModel(num_qubits=5, ansatz_reps=2, initial_spsa_steps=2, adapt_spsa_steps=2)
    m4 = DriftAdaptiveVQCModel(num_qubits=5, ansatz_reps=2, initial_spsa_steps=2, adapt_spsa_steps=2)
    assert m3.num_weights == 15  # 5 qubits * (2 reps + 1) = 15 trainable parameters

    m3.fit_initial(X_w0, y_w0)
    m4.fit_initial(X_w0, y_w0)
    w_before = m4.weights.copy()

    pred3, prob3, enc3, sim3 = m3.predict(X_w0[0])
    assert pred3 in (0, 1)
    assert 0.0 <= prob3 <= 1.0
    assert enc3 >= 0.0 and sim3 > 0.0

    t_adapt = m4.adapt_on_drift(X_w0, y_w0)
    assert t_adapt > 0.0
    assert m4.adaptation_count == 1
    assert not np.allclose(w_before, m4.weights)


def test_pennylane_qsvm_static_and_adaptive_predictions() -> None:
    """Verify PennyLane Quantum Kernel (M5 & M6) produce binary predictions and refresh supports."""
    rng = np.random.default_rng(7)
    X_w0 = rng.uniform(0.0, math.pi, size=(12, 5))
    y_w0 = np.array([0, 1] * 6, dtype=np.int64)

    m5 = StaticQuantumKernelModel(num_qubits=5, max_support_size=10, replace_ratio=0.2)
    m6 = DriftAdaptiveQuantumKernelModel(num_qubits=5, max_support_size=10, replace_ratio=0.2)

    m5.fit_initial(X_w0, y_w0)
    m6.fit_initial(X_w0, y_w0)

    pred5, prob5, enc5, sim5 = m5.predict(X_w0[0])
    assert pred5 in (0, 1)
    assert 0.0 <= prob5 <= 1.0
    assert enc5 >= 0.0 and sim5 > 0.0

    X_new = rng.uniform(0.0, math.pi, size=(6, 5))
    y_new = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    t_adapt = m6.adapt_on_drift(X_new, y_new)
    assert t_adapt > 0.0
    assert m6.adaptation_count == 1


def test_prequential_evaluator_and_influx_points_end_to_end() -> None:
    """Integration test verifying test-then-train pipeline across 15 events and telemetry point formatting."""
    cfg = PipelineConfig(
        initial_warmup_window=10,
        sliding_window_size=10,
        rolling_metric_window=10,
        vqc_initial_spsa_steps=2,
        vqc_adapt_spsa_steps=2,
        qsvm_max_support_size=10,
        influxdb_enabled=False,
    )
    cfg.validate()
    worker = StreamFeatureWorker(cfg)
    evaluator = PrequentialEvaluator(cfg, worker)
    sink = InfluxTelemetrySink(cfg)

    for idx in range(15):
        payload = serialize_elec2_event(idx, _sample_raw_features(offset=idx * 0.03), idx % 2)
        ev = worker.process_raw_payload(payload)
        assert ev is not None
        res = evaluator.evaluate_event(ev)
        assert len(res.model_snapshots) == 8
        points = sink.build_points(res)
        # 8 models * 2 measurements (accuracy + latency) + 1 drift measurement = 17 Influx points
        assert len(points) == 17


def test_edge_cases_empty_window_and_non_finite_features() -> None:
    """Verify empty SlidingWindowBuffer behavior and rejection of NaN/Inf feature values."""
    buf = SlidingWindowBuffer(maxlen=10)
    X_empty, y_empty = buf.get_arrays()
    assert X_empty.shape == (0, 5)
    assert y_empty.shape == (0,)

    scaler = OnlineAngleMinMaxScaler(FEATURE_NAMES)
    bad_nan = _sample_raw_features()
    bad_nan["nswprice"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite value"):
        scaler.partial_fit_transform(bad_nan)

    bad_inf = _sample_raw_features()
    bad_inf["vicdemand"] = float("inf")
    with pytest.raises(ValueError, match="Non-finite value"):
        scaler.partial_fit_transform(bad_inf)


def test_adwin_drift_and_recovery_tracker() -> None:
    """Verify ADWIN detects a synthetic error jump and RecoveryTracker computes N_recovery."""
    dm = ADWINDriftManager(delta=0.01)

    # Feed 100 zero-error predictions followed by high-error predictions
    drift_rec = None
    for i in range(200):
        y_true = 1
        y_pred = 1 if i < 100 else 0
        rec = dm.update_and_check(i, "1997-01-01T00:00:00Z", y_true, y_pred)
        if rec is not None:
            drift_rec = rec
            break

    assert drift_rec is not None
    assert dm.cumulative_drifts >= 1

    # N_recovery: dip below (pre-drift - eps) must occur, then a sustained re-entry
    # into the band counts from the drift onset to the first qualifying event.
    rt = RecoveryTracker(
        ["M4_Quantum_Adaptive_VQC"],
        recovery_epsilon=0.02,
        sustain_events=3,
        recovery_horizon=50,
    )
    rt.register_drift("M4_Quantum_Adaptive_VQC", 100, 0.80)  # target band: 0.78
    rt.observe_accuracies(110, {"M4_Quantum_Adaptive_VQC": 0.70})  # dip
    rt.observe_accuracies(120, {"M4_Quantum_Adaptive_VQC": 0.79})  # first hit
    rt.observe_accuracies(121, {"M4_Quantum_Adaptive_VQC": 0.79})  # sustained
    rt.observe_accuracies(122, {"M4_Quantum_Adaptive_VQC": 0.79})  # confirmed
    assert rt.mean_recovery_events("M4_Quantum_Adaptive_VQC") == 20.0


def test_recovery_tracker_never_degraded_and_censoring() -> None:
    """A model that never dips scores N_recovery=0; a new drift censors a pending episode."""
    rt = RecoveryTracker(
        ["M1", "M2"], recovery_epsilon=0.02, sustain_events=2, recovery_horizon=10
    )

    # M1 never degrades: stays in the band until the horizon passes
    rt.register_drift("M1", 100, 0.80)
    for t in range(101, 116):
        rt.observe_accuracies(t, {"M1": 0.85})
    assert rt.mean_recovery_events("M1") == 0.0
    assert rt.completed_recoveries("M1")[0].never_degraded is True

    # M2 dips but a second drift arrives before recovery: episode is censored
    rt.register_drift("M2", 200, 0.80)
    rt.observe_accuracies(205, {"M2": 0.50})
    rt.register_drift("M2", 210, 0.60)  # censors the in-flight episode
    censored = [r for r in rt.records if r.model_name == "M2" and not r.completed]
    assert len(censored) == 1


def test_rbf_static_and_adaptive_controls() -> None:
    """Verify M7 fits once and freezes while M8 refreshes its support set on adaptation."""
    rng = np.random.default_rng(11)
    X_w0 = rng.uniform(0.0, math.pi, size=(12, 5))
    y_w0 = np.array([0, 1] * 6, dtype=np.int64)

    m7 = StaticRBFKernelModel(max_support_size=10, replace_ratio=0.2)
    m8 = DriftAdaptiveRBFKernelModel(max_support_size=10, replace_ratio=0.2)
    m7.fit_initial(X_w0, y_w0)
    m8.fit_initial(X_w0, y_w0)

    pred7, prob7, t7 = m7.predict(X_w0[0])
    assert pred7 in (0, 1) and 0.0 <= prob7 <= 1.0 and t7 >= 0.0

    X_new = rng.uniform(0.0, math.pi, size=(6, 5))
    y_new = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    assert m7.adapt_on_drift(X_new, y_new) == 0.0  # static control never adapts
    assert m8.adapt_on_drift(X_new, y_new) > 0.0
    assert m8.adaptation_count == 1
    assert 0.0 < m8.last_support_churn <= 1.0


def test_uninitialized_models_predict_majority_class() -> None:
    """Warmup policy: uninitialized quantum/kernel models follow the running majority label."""
    cfg = PipelineConfig(
        initial_warmup_window=50,
        sliding_window_size=10,
        vqc_initial_spsa_steps=2,
        vqc_adapt_spsa_steps=2,
        qsvm_max_support_size=10,
        influxdb_enabled=False,
    )
    worker = StreamFeatureWorker(cfg)
    evaluator = PrequentialEvaluator(cfg, worker)

    # Ten events labeled 1 -> majority class is 1 once any label has been seen
    for idx in range(10):
        payload = serialize_elec2_event(idx, _sample_raw_features(offset=idx * 0.03), 1)
        ev = worker.process_raw_payload(payload)
        res = evaluator.evaluate_event(ev)
        if ev.event_id == 0:
            continue  # no labels observed yet at the very first event
        for name in evaluator.model_names[2:]:  # M3-M8 uninitialized during warmup
            assert res.model_snapshots[name].y_pred == 1


def test_post_warmup_metrics_and_update_counts() -> None:
    """Post-warmup accuracy is tracked and update counts grow only on actual updates."""
    cfg = PipelineConfig(
        initial_warmup_window=10,
        sliding_window_size=10,
        rolling_metric_window=10,
        vqc_initial_spsa_steps=2,
        vqc_adapt_spsa_steps=2,
        qsvm_max_support_size=10,
        influxdb_enabled=False,
    )
    worker = StreamFeatureWorker(cfg)
    evaluator = PrequentialEvaluator(cfg, worker)

    last_res = None
    for idx in range(15):
        payload = serialize_elec2_event(idx, _sample_raw_features(offset=idx * 0.03), idx % 2)
        ev = worker.process_raw_payload(payload)
        last_res = evaluator.evaluate_event(ev)

    snap_m3 = last_res.model_snapshots["M3_Quantum_Static_VQC"]
    assert 0.0 <= snap_m3.cumulative_accuracy_post_warmup <= 1.0
    assert snap_m3.n_updates == 1  # initial fit only
    snap_m2 = last_res.model_snapshots["M2_Classical_Adaptive_HATC"]
    assert snap_m2.n_updates == 15  # updates every event
    assert last_res.pair_agreements.keys() == {"M2_M4", "M3_M4", "M5_M6", "M7_M8"}


def test_config_requires_influx_token_when_enabled() -> None:
    """The InfluxDB sink refuses to run enabled without a token from the environment."""
    import os

    os.environ.pop("INFLUXDB_TOKEN", None)
    cfg = PipelineConfig(influxdb_enabled=True)
    with pytest.raises(ValueError, match="INFLUXDB_TOKEN"):
        cfg.validate()
    cfg_off = PipelineConfig(influxdb_enabled=False)
    cfg_off.validate()  # disabled sink needs no token