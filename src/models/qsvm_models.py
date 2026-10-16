# src/models/qsvm_models.py
"""Quantum Kernel Classifier models (M5: Static QSVM, M6: Drift-Adaptive QSVM) using PennyLane."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import pennylane as qml
from sklearn.svm import SVC


class BasePennyLaneQSVM:
    """5-Qubit Fidelity Quantum Kernel Classifier with AngleEmbedding and CNOT entanglers."""

    def __init__(
        self,
        num_qubits: int = 5,
        max_support_size: int = 80,
        replace_ratio: float = 0.20,
        svm_c: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.max_support_size = max_support_size
        self.replace_ratio = replace_ratio
        self.svm_c = svm_c
        self.rng = np.random.default_rng(seed)

        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface=None)
        def _embedding_state(x: np.ndarray):
            qml.AngleEmbedding(features=x, wires=range(num_qubits), rotation="Y")
            for w in range(num_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[num_qubits - 1, 0])
            qml.AngleEmbedding(features=x, wires=range(num_qubits), rotation="Z")
            return qml.state()

        self._qnode = _embedding_state
        self.support_X: Optional[np.ndarray] = None
        self.support_states: Optional[np.ndarray] = None  # Shape (M, 2^5) complex128
        self.support_y: Optional[np.ndarray] = None
        self.svc: Optional[SVC] = None
        self.is_initialized: bool = False

    def _compute_quantum_state(self, x_angle: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Prepare normalized feature vector and execute PennyLane statevector circuit.

        Returns:
            Tuple of (quantum_state_vector, t_encode_ms, t_sim_ms).
        """
        t_enc_start = time.perf_counter_ns()
        x_clean = np.asarray(x_angle, dtype=np.float64)
        t_encode_ms = (time.perf_counter_ns() - t_enc_start) / 1_000_000.0

        t_sim_start = time.perf_counter_ns()
        state = np.asarray(self._qnode(x_clean), dtype=np.complex128)
        t_sim_ms = (time.perf_counter_ns() - t_sim_start) / 1_000_000.0
        return state, t_encode_ms, t_sim_ms

    def _fit_from_support_arrays(
        self, X_sup: np.ndarray, y_sup: np.ndarray
    ) -> float:
        """Compute quantum states for support prototypes, build fidelity Gram matrix, and fit SVC."""
        t_start = time.perf_counter_ns()
        if len(X_sup) > self.max_support_size:
            X_sup = X_sup[-self.max_support_size :]
            y_sup = y_sup[-self.max_support_size :]

        states = []
        for x_i in X_sup:
            st, _, _ = self._compute_quantum_state(x_i)
            states.append(st)
        state_matrix = np.vstack(states)  # Shape (M, 32)

        # Fidelity kernel K_ij = |<phi(x_i) | phi(x_j)>|^2
        inner_products = state_matrix @ np.conj(state_matrix.T)
        gram_matrix = np.abs(inner_products) ** 2

        self.support_X = X_sup.copy()
        self.support_states = state_matrix
        self.support_y = y_sup.copy()

        # Guard against single-class windows during early stream startup
        unique_classes = np.unique(y_sup)
        if len(unique_classes) >= 2:
            self.svc = SVC(kernel="precomputed", C=self.svm_c, probability=True, random_state=42)
            self.svc.fit(gram_matrix, y_sup)
        else:
            self.svc = None

        self.is_initialized = True
        return (time.perf_counter_ns() - t_start) / 1_000_000.0

    def predict(self, x_angle: np.ndarray) -> Tuple[int, float, float, float]:
        """Predict binary label {0, 1} by evaluating quantum kernel against active support set.

        Returns:
            Tuple of (pred_label, prob_pos, t_encode_ms, t_sim_ms).
        """
        state_x, t_encode_ms, t_sim_ms = self._compute_quantum_state(x_angle)

        t_kern_start = time.perf_counter_ns()
        if not self.is_initialized or self.support_states is None or self.support_y is None:
            prob_pos = 0.5
            pred = 0
        elif self.svc is None:
            majority_class = int(np.round(np.mean(self.support_y)))
            prob_pos = float(majority_class)
            pred = majority_class
        else:
            # Fidelity kernel vector k(x, X_support) = |<phi(x) | phi(x_s)>|^2
            overlaps = state_x @ np.conj(self.support_states.T)
            k_vec = (np.abs(overlaps) ** 2).reshape(1, -1)
            proba = self.svc.predict_proba(k_vec)[0]
            classes = list(self.svc.classes_)
            prob_pos = float(proba[classes.index(1)]) if 1 in classes else 0.5
            pred = 1 if prob_pos >= 0.5 else 0

        t_sim_ms += (time.perf_counter_ns() - t_kern_start) / 1_000_000.0
        return pred, prob_pos, t_encode_ms, t_sim_ms


class StaticQuantumKernelModel(BasePennyLaneQSVM):
    """M5: Static Quantum Kernel Baseline (Tscharke et al. style).

    Selects support prototypes from initial window W_0 and keeps them fixed thereafter.
    """

    name: str = "M5_Quantum_Static_QSVM"

    def fit_initial(self, X_warmup: np.ndarray, y_warmup: np.ndarray) -> float:
        """Fit initial support set from W_0 once."""
        if self.is_initialized:
            return 0.0
        return self._fit_from_support_arrays(X_warmup, y_warmup)


class DriftAdaptiveQuantumKernelModel(BasePennyLaneQSVM):
    """M6: Drift-Adaptive Quantum Kernel (Proposed Novelty).

    Replaces the oldest 20% of support vectors with recent post-drift prototypes from
    the sliding window W_t whenever ADWIN signals concept drift.
    """

    name: str = "M6_Quantum_Adaptive_QSVM"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adaptation_count: int = 0

    def fit_initial(self, X_warmup: np.ndarray, y_warmup: np.ndarray) -> float:
        """Fit initial support set from W_0."""
        if self.is_initialized:
            return 0.0
        return self._fit_from_support_arrays(X_warmup, y_warmup)

    def adapt_on_drift(self, X_sliding: np.ndarray, y_sliding: np.ndarray) -> float:
        """Replace oldest 20% of support prototypes with newest samples from W_t and refit SVC."""
        if len(X_sliding) == 0:
            return 0.0
        if not self.is_initialized or self.support_X is None or self.support_y is None:
            return self._fit_from_support_arrays(X_sliding, y_sliding)

        n_current = len(self.support_X)
        n_replace = max(1, int(round(n_current * self.replace_ratio)))
        n_replace = min(n_replace, len(X_sliding))

        retained_X = self.support_X[n_replace:]
        retained_y = self.support_y[n_replace:]
        new_X = X_sliding[-n_replace:]
        new_y = y_sliding[-n_replace:]

        updated_X = np.vstack([retained_X, new_X])
        updated_y = np.concatenate([retained_y, new_y])

        # Ensure both classes are represented if available in the sliding window
        if len(np.unique(updated_y)) < 2 and len(np.unique(y_sliding)) >= 2:
            updated_X = X_sliding[-self.max_support_size :]
            updated_y = y_sliding[-self.max_support_size :]

        self.adaptation_count += 1
        return self._fit_from_support_arrays(updated_X, updated_y)

class ClassicalRBFKernelModel:
    """Classical RBF Kernel SVM control with identical sliding support set size and refresh rule.

    Used to evaluate Hypothesis 4 (Sample Efficiency) in Section 5 of PLAN.md:
    comparing Drift-Adaptive Quantum Kernel (M6) against a Classical RBF Kernel
    across varying support set budgets.
    """

    name: str = "Control_Classical_RBF"

    def __init__(
        self,
        max_support_size: int = 80,
        replace_ratio: float = 0.20,
        svm_c: float = 1.0,
        svm_gamma: str | float = "scale",
    ) -> None:
        self.max_support_size = max_support_size
        self.replace_ratio = replace_ratio
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.support_X: Optional[np.ndarray] = None
        self.support_y: Optional[np.ndarray] = None
        self.svc: Optional[SVC] = None
        self.is_initialized: bool = False
        self.adaptation_count: int = 0
        self.last_support_churn: float = 0.0

    def fit_support(self, X_sup: np.ndarray, y_sup: np.ndarray) -> float:
        t0 = time.perf_counter_ns()
        if len(X_sup) > self.max_support_size:
            X_sup = X_sup[-self.max_support_size :]
            y_sup = y_sup[-self.max_support_size :]
        self.support_X = X_sup.copy()
        self.support_y = y_sup.copy()
        if len(np.unique(y_sup)) >= 2:
            self.svc = SVC(
                kernel="rbf",
                C=self.svm_c,
                gamma=self.svm_gamma,
                probability=True,
                random_state=42,
            )
            self.svc.fit(self.support_X, self.support_y)
        else:
            self.svc = None
        self.is_initialized = True
        return (time.perf_counter_ns() - t0) / 1_000_000.0

    def predict(self, x_angle: np.ndarray) -> Tuple[int, float, float]:
        t0 = time.perf_counter_ns()
        if not self.is_initialized or self.support_y is None:
            return 0, 0.5, (time.perf_counter_ns() - t0) / 1_000_000.0
        if self.svc is None:
            maj = int(np.round(np.mean(self.support_y)))
            return maj, float(maj), (time.perf_counter_ns() - t0) / 1_000_000.0
        proba = self.svc.predict_proba(np.asarray(x_angle, dtype=np.float64).reshape(1, -1))[0]
        classes = list(self.svc.classes_)
        prob_pos = float(proba[classes.index(1)]) if 1 in classes else 0.5
        pred = 1 if prob_pos >= 0.5 else 0
        return pred, prob_pos, (time.perf_counter_ns() - t0) / 1_000_000.0

    def adapt_on_drift(self, X_sliding: np.ndarray, y_sliding: np.ndarray) -> float:
        if len(X_sliding) == 0:
            return 0.0
        if not self.is_initialized or self.support_X is None or self.support_y is None:
            return self.fit_support(X_sliding, y_sliding)
        n_replace = min(
            max(1, int(round(len(self.support_X) * self.replace_ratio))),
            len(X_sliding),
        )
        updated_X = np.vstack([self.support_X[n_replace:], X_sliding[-n_replace:]])
        updated_y = np.concatenate([self.support_y[n_replace:], y_sliding[-n_replace:]])
        if len(np.unique(updated_y)) < 2 and len(np.unique(y_sliding)) >= 2:
            updated_X = X_sliding[-self.max_support_size :]
            updated_y = y_sliding[-self.max_support_size :]
        self.adaptation_count += 1
        self.last_support_churn = n_replace / max(1, len(self.support_X))
        return self.fit_support(updated_X, updated_y)


class StaticRBFKernelModel(ClassicalRBFKernelModel):
    """M7: Static Classical RBF control.

    Fitted once on the initial warmup window W_0 and frozen thereafter; mirrors M5's
    support budget so the quantum kernel has a capacity-matched classical counterpart.
    """

    name: str = "M7_Control_RBF_Static"

    def fit_initial(self, X_warmup: np.ndarray, y_warmup: np.ndarray) -> float:
        """Fit the RBF SVM on W_0 once."""
        if self.is_initialized:
            return 0.0
        return self.fit_support(X_warmup, y_warmup)

    def adapt_on_drift(self, X_sliding: np.ndarray, y_sliding: np.ndarray) -> float:
        """Static control never adapts."""
        return 0.0


class DriftAdaptiveRBFKernelModel(ClassicalRBFKernelModel):
    """M8: Drift-Adaptive Classical RBF control.

    Applies the exact same sliding-support refresh rule as M6 on ADWIN drift alerts,
    isolating the quantum-vs-classical kernel comparison from the adaptation mechanism.
    """

    name: str = "M8_Control_RBF_Adaptive"

    def fit_initial(self, X_warmup: np.ndarray, y_warmup: np.ndarray) -> float:
        """Fit the RBF SVM on W_0 once."""
        if self.is_initialized:
            return 0.0
        return self.fit_support(X_warmup, y_warmup)
    