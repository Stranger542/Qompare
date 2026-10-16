# src/models/vqc_models.py
"""Variational Quantum Classifier models (M3: Static VQC, M4: Drift-Adaptive VQC) using Qiskit."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector


class BaseQiskitVQC:
    """5-Qubit Variational Quantum Classifier with ZFeatureMap and RealAmplitudes ansatz.

    Separates state preparation / parameter binding (`t_encode_ms`) from statevector
    simulation (`t_sim_ms`) and SPSA optimization (`t_update_ms`) for telemetry.
    """

    def __init__(
        self,
        num_qubits: int = 5,
        ansatz_reps: int = 2,
        initial_spsa_steps: int = 40,
        adapt_spsa_steps: int = 10,
        spsa_lr: float = 0.15,
        spsa_perturbation: float = 0.10,
        adapt_lr_scale: float = 0.75,
        adapt_batch_size: int = 0,
        readout: str = "parity",
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.ansatz_reps = ansatz_reps
        self.initial_spsa_steps = initial_spsa_steps
        self.adapt_spsa_steps = adapt_spsa_steps
        self.spsa_lr = spsa_lr
        self.spsa_perturbation = spsa_perturbation
        self.adapt_lr_scale = adapt_lr_scale
        self.adapt_batch_size = adapt_batch_size
        self.readout = readout
        self.rng = np.random.default_rng(seed)

        # Build Qiskit Feature Map (5 qubits) and Variational Ansatz (depth=2 -> 15 params)
        self.feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
        self.ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            entanglement="linear",
            reps=ansatz_reps,
        )
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)
        # Decompose composite library blueprints into primitive gates for fast binding
        self._decomposed_circuit = self.circuit.decompose()

        self._feature_params = list(self.feature_map.parameters)
        self._weight_params = list(self.ansatz.parameters)
        self.num_weights = len(self._weight_params)

        # Initialize variational parameters near small random angles
        self.weights: np.ndarray = self.rng.uniform(
            -0.3, 0.3, size=self.num_weights
        ).astype(np.float64)

        # Readout observable with eigenvalues in {-1, +1}:
        # 'parity' -> Z^{\otimes 5}, 'z0' -> Z on qubit 0 (Pauli label is little-endian).
        if readout == "parity":
            pauli_label = "Z" * num_qubits
        elif readout == "z0":
            pauli_label = "I" * (num_qubits - 1) + "Z"
        else:
            raise ValueError(f"Unknown readout '{readout}'; expected 'parity' or 'z0'.")
        self.observable = SparsePauliOp.from_list([(pauli_label, 1.0)])
        self.is_initialized: bool = False

    def _forward_single(
        self, x_angle: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Bind circuit parameters and simulate parity expectation value.

        Returns:
            Tuple of (prob_pos, t_encode_ms, t_sim_ms).
        """
        t_enc_start = time.perf_counter_ns()
        param_binding = {}
        for p, val in zip(self._feature_params, x_angle):
            param_binding[p] = float(val)
        for p, val in zip(self._weight_params, weights):
            param_binding[p] = float(val)
        bound_circuit = self._decomposed_circuit.assign_parameters(
            param_binding, inplace=False
        )
        t_encode_ms = (time.perf_counter_ns() - t_enc_start) / 1_000_000.0

        t_sim_start = time.perf_counter_ns()
        sv = Statevector.from_instruction(bound_circuit)
        exp_val = float(np.real(sv.expectation_value(self.observable)))
        # Map expectation value in [-1, 1] to probability in [0, 1]
        prob_pos = float(np.clip(0.5 * (1.0 - exp_val), 1e-6, 1.0 - 1e-6))
        t_sim_ms = (time.perf_counter_ns() - t_sim_start) / 1_000_000.0
        return prob_pos, t_encode_ms, t_sim_ms

    def predict(self, x_angle: np.ndarray) -> Tuple[int, float, float, float]:
        """Predict binary label {0, 1} for a single normalized 5-feature vector.

        Returns:
            Tuple of (pred_label, prob_pos, t_encode_ms, t_sim_ms).
        """
        prob_pos, t_encode_ms, t_sim_ms = self._forward_single(x_angle, self.weights)
        pred = 1 if prob_pos >= 0.5 else 0
        return pred, prob_pos, t_encode_ms, t_sim_ms

    def _batch_cross_entropy_loss(
        self, X_batch: np.ndarray, y_batch: np.ndarray, weights: np.ndarray
    ) -> float:
        """Compute binary cross-entropy loss over a mini-batch of quantum circuit evaluations."""
        losses = []
        for x_i, y_i in zip(X_batch, y_batch):
            prob_pos, _, _ = self._forward_single(x_i, weights)
            loss = -(
                y_i * np.log(prob_pos) + (1.0 - y_i) * np.log(1.0 - prob_pos)
            )
            losses.append(loss)
        return float(np.mean(losses))

    def _run_spsa_optimization(
        self,
        X_window: np.ndarray,
        y_window: np.ndarray,
        n_steps: int,
        batch_size: Optional[int] = None,
        lr_scale: float = 1.0,
    ) -> float:
        """Run warm-started SPSA optimization steps on the provided window.

        Args:
            X_window: Array of shape (N, 5) with features in [0, pi].
            y_window: Array of shape (N,) with binary labels {0, 1}.
            n_steps: Number of SPSA simultaneous perturbation steps.
            batch_size: Subsample size per step; None -> self.adapt_batch_size;
                0 -> full window per step (no subsampling).
            lr_scale: Multiplier on base learning rate for warm fine-tuning.

        Returns:
            Elapsed update time in milliseconds (`t_update_ms`).
        """
        if len(X_window) == 0 or n_steps <= 0:
            return 0.0

        t_start = time.perf_counter_ns()
        n_samples = len(X_window)
        if batch_size is None:
            batch_size = self.adapt_batch_size
        effective_batch = n_samples if batch_size <= 0 else min(batch_size, n_samples)

        for k in range(1, n_steps + 1):
            if n_samples > effective_batch:
                idx = self.rng.choice(n_samples, size=effective_batch, replace=False)
                X_sub, y_sub = X_window[idx], y_window[idx]
            else:
                X_sub, y_sub = X_window, y_window

            a_k = (self.spsa_lr * lr_scale) / ((k + 5.0) ** 0.602)
            c_k = self.spsa_perturbation / (k**0.101)

            # Rademacher +/-1 simultaneous perturbation vector
            delta = self.rng.choice([-1.0, 1.0], size=self.num_weights)
            w_plus = self.weights + c_k * delta
            w_minus = self.weights - c_k * delta

            loss_plus = self._batch_cross_entropy_loss(X_sub, y_sub, w_plus)
            loss_minus = self._batch_cross_entropy_loss(X_sub, y_sub, w_minus)

            grad_est = (loss_plus - loss_minus) / (2.0 * c_k * delta)
            self.weights = self.weights - a_k * grad_est

        self.is_initialized = True
        return (time.perf_counter_ns() - t_start) / 1_000_000.0


class StaticVQCModel(BaseQiskitVQC):
    """M3: Static VQC Baseline (Herbst et al. style).

    Trained once on the initial warmup window W_0 and frozen for the remainder of the stream.
    """

    name: str = "M3_Quantum_Static_VQC"

    def fit_initial(self, X_warmup: np.ndarray, y_warmup: np.ndarray) -> float:
        """Fit initial circuit weights on W_0 if not already frozen."""
        if self.is_initialized:
            return 0.0
        return self._run_spsa_optimization(
            X_warmup,
            y_warmup,
            n_steps=self.initial_spsa_steps,
            batch_size=24,
            lr_scale=1.0,
        )


class DriftAdaptiveVQCModel(BaseQiskitVQC):
    """M4: Drift-Adaptive VQC (Proposed Novelty).

    Performs fast forward inference during stationary periods and triggers warm-started,
    few-step SPSA fine-tuning on the sliding window buffer W_t upon an ADWIN drift alert.
    """

    name: str = "M4_Quantum_Adaptive_VQC"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adaptation_count: int = 0
        self.last_weight_delta_norm: float = 0.0

    def fit_initial(self, X_warmup: np.ndarray, y_warmup: np.ndarray) -> float:
        """Fit initial circuit weights on W_0."""
        if self.is_initialized:
            return 0.0
        return self._run_spsa_optimization(
            X_warmup,
            y_warmup,
            n_steps=self.initial_spsa_steps,
            batch_size=24,
            lr_scale=1.0,
        )

    def adapt_on_drift(self, X_sliding: np.ndarray, y_sliding: np.ndarray) -> float:
        """Execute warm-started SPSA fine-tuning on recent sliding window W_t.

        Batch size and learning-rate scale come from configuration
        (`vqc_adapt_batch_size`, `vqc_adapt_lr_scale`) so adaptation strength can
        be ablated; records the weight displacement for adaptation diagnostics.
        """
        if len(X_sliding) < 4:
            return 0.0
        weights_before = self.weights.copy()
        t_update_ms = self._run_spsa_optimization(
            X_sliding,
            y_sliding,
            n_steps=self.adapt_spsa_steps,
            batch_size=self.adapt_batch_size,
            lr_scale=self.adapt_lr_scale,
        )
        self.adaptation_count += 1
        self.last_weight_delta_norm = float(np.linalg.norm(self.weights - weights_before))
        return t_update_ms


    