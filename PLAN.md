# IOE413-Quantum-Streaming: Quantum vs. Classical Streaming Data Analytics

An end-to-end streaming analytics benchmark comparing **Classical Streaming Machine Learning** (`River` Hoeffding Trees) against **Static and Drift-Adaptive Quantum Machine Learning** (`Qiskit` VQC and `PennyLane` Quantum Kernel SVM) on the non-stationary `ELEC2` electricity market stream.

---

## 1. Architecture & 3×2 Comparative Model Matrix

Data flows continuously through five decoupled stages:
1. **Ingestion (`src/producer/elec2_producer.py`):** Replays the 45,312-event `ELEC2` dataset into Apache Kafka topic `elec2-stream`.
2. **Preprocessing (`src/worker/feature_worker.py`):** Consumes events, applies leak-free online MinMax scaling to $[0, \pi]$, and maintains a bounded FIFO sliding window ($W=100$).
3. **Parallel 3×2 Model Inference (`src/models/`):**
   - **M1:** Classical Static Hoeffding Tree (trained on $W_0=200$, frozen).
   - **M2:** Classical Adaptive Hoeffding Tree + ADWIN.
   - **M3:** Quantum Static 5-Qubit VQC (`ZFeatureMap` + `RealAmplitudes`, depth=2, frozen after $W_0$).
   - **M4:** Quantum Drift-Adaptive 5-Qubit VQC (warm-started 10-step SPSA fine-tuning on ADWIN drift alert).
   - **M5:** Quantum Static 5-Qubit Fidelity Kernel SVM (frozen support set from $W_0$).
   - **M6:** Quantum Drift-Adaptive 5-Qubit Fidelity Kernel SVM (replaces oldest 20% of support vectors from $W_t$ on ADWIN drift alert).
4. **Prequential Evaluation & Telemetry (`src/evaluation/`, `src/sink/influx_sink.py`):** Strict test-then-train evaluation written to InfluxDB v2 (`streaming_quantum` bucket).
5. **Monitoring (`grafana/`):** Pre-provisioned real-time Grafana dashboard on port `3000`.

> **Note on Quantum Execution:** All quantum circuits in M3–M6 execute via exact statevector simulation (`qiskit.quantum_info.Statevector` and PennyLane `default.qubit`) to provide reproducible latency decomposition between circuit encoding ($t_{\text{encode}}$), statevector simulation ($t_{\text{sim}}$), and parameter/support updates ($t_{\text{update}}$).

---

## 2. Prerequisites & Installation

- **Python:** `>=3.10, <3.13`
- **Docker & Docker Compose:** v2+

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Copy environment configuration
cp .env.example .env

