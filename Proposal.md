# Course Project Proposal: Drift-Adaptive Quantum vs. Classical Streaming Data Analytics
**Course:** IOE413 — Track C (Generative & Quantum Methods, Topic 4)

## 1. Problem Statement & Research Question
Real-world IoT and energy telemetry streams arrive continuously at high velocity and exhibit non-stationary concept drift. While Quantum Machine Learning (QML) promises enhanced feature expressiveness and sample efficiency in low-qubit regimes, existing evaluations operate in offline batch mode on static, shuffled datasets. This study investigates:
> *"Does the theoretical expressiveness and sample-efficiency advantage of quantum machine learning survive under the real-time constraints, high velocity, and concept drift of a true streaming environment?"*

## 2. Literature Review Summary
1. **Domingos & Hulten (2000) — *Mining High-Speed Data Streams*:**
   - **Summary:** Introduced the Very Fast Decision Tree (VFDT / Hoeffding Tree) using the Hoeffding bound to guarantee constant-time incremental node splits on high-speed data streams.
   - **Role in Study:** Forms our classical static (M1) and ADWIN-augmented adaptive (M2) streaming baselines.
2. **Herbst et al. (2024) — *Quantum Machine Learning for IoT/Edge Data Streams*:**
   - **Summary:** Evaluated Variational Quantum Classifiers (VQC) using angle feature maps and shallow hardware-efficient ansatz circuits (`RealAmplitudes`) on sensor classification tasks.
   - **Limitation & Role:** Evaluated with frozen weights after batch training; reproduced as Model M3 (Static VQC) and extended via Model M4 (Drift-Adaptive VQC).
3. **Tscharke et al. (2023) — *Quantum Kernel Methods for Anomaly & Stream Classification*:**
   - **Summary:** Demonstrated fidelity-based Quantum Support Vector Machines (QSVM) with angle embedding and entangling layers on compact support sets.
   - **Limitation & Role:** Fixed support vectors degrade under concept drift; reproduced as Model M5 (Static Quantum Kernel) and extended via Model M6 (Drift-Adaptive Quantum Kernel).

## 3. Dataset & Track C Tool Justification
- **Dataset:** Australian New South Wales Electricity Market (`ELEC2`, 45,312 half-hourly records, 5 continuous predictors: `nswprice`, `nswdemand`, `vicprice`, `vicdemand`, `transfer`). Maps 1-to-1 onto a 5-qubit register without lossy PCA compression and contains genuine seasonal and regulatory concept drifts.
- **Toolchain:**
  - **Apache Kafka:** Decouples chronological stream ingestion (`elec2-stream`) from stateful stream processing.
  - **River + Qiskit + PennyLane:** Implements the $3 \times 2$ classical/quantum model matrix under identical prequential evaluation.
  - **InfluxDB v2 + Grafana:** Persists real-time accuracy, latency decomposition, and ADWIN drift alerts for live operational monitoring.

## 4. Proposed Novelty: Dual Drift-Adaptive Quantum Learning
Rather than retraining quantum circuits on every incoming event (which is computationally infeasible), we monitor prequential residuals $e_t = \vert{}y_t - y_{\text{pred}}\vert{}$ with an **ADWIN** drift detector:
- **Stationary Regime:** Execute low-latency forward quantum inference with frozen circuit parameters and support sets.
- **Drift Alert Regime:** Trigger targeted adaptation using a sliding FIFO window $W_t$ ($W=100$):
  1. **M4 (Drift-Adaptive VQC):** Executes warm-started 10-step SPSA fine-tuning starting from current variational weights $\theta_t$.
  2. **M6 (Drift-Adaptive Quantum Kernel):** Replaces the oldest 20% of quantum kernel support prototypes with post-drift samples from $W_t$.
  