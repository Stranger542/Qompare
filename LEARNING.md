# LEARNING.md — Understanding the Qompare Project End to End

> **Purpose:** after reading this document you should be able to explain the project's
> motivation, theory, design, implementation, and results — including defending the design
> choices under questioning. Companion files: `PLAN.md` (design spec + deviations),
> `README.md` (how to run), `Proposal.md` (original scope), `experiments/demo_script.md`
> (live demo), `experiments/plots/` (result artifacts).

---

## 1. The Big Picture

**One-sentence summary:** Qompare is an end-to-end streaming analytics pipeline that runs
an 8-model "classical vs quantum × static vs adaptive" tournament on a real, drifting
electricity-market data stream, evaluating every model prequentially (predict first, learn
after) through Kafka → models → InfluxDB → Grafana, to answer:

> *Does the expressiveness / sample-efficiency appeal of Quantum Machine Learning survive
> the latency bounds and concept drift of a true streaming environment?*

**The honest answer the data gives:** partly. Drift-triggered adaptation genuinely helps
quantum models (+4 to +5 accuracy points over frozen ones), but the best quantum model only
*ties* a capacity-matched classical kernel at ~20× the per-event compute, and both trail a
classical adaptive decision tree. The quantum advantage does not survive streaming on this
task — a clean, honestly reported negative result with a working novelty mechanism.

**Course context (IOE413 Streaming Data Analytics, Topic 4, Track C):** reproduce one base
paper's core method (Domingos & Hulten's Hoeffding tree), adapt two quantum base papers
(Herbst et al. 2024, Tscharke et al. 2023) to streaming, and add a clearly-scoped
enhancement (ADWIN-triggered drift adaptation applied symmetrically to variational and
kernel QML).

---

## 2. Theory Background

### 2.1 Streaming ML and prequential evaluation

A **data stream** is an ordered, unbounded, high-velocity sequence of events. Constraints:
one pass (or bounded memory), per-event latency budgets, and **non-stationarity** — the
data-generating distribution changes over time.

**Prequential (test-then-train) evaluation** is the streaming answer to train/test splits:
for each event `(x_t, y_t)` in arrival order:

1. **Predict** `ŷ_t = model(x_t)` *before* the model may see `y_t`.
2. **Score** the prediction against the revealed `y_t`.
3. **Update** the model with `(x_t, y_t)` — only now.

This yields an unbiased online accuracy estimate with no held-out set and no cheating. Any
leakage (scaler peeking at future values, labels entering buffers before prediction) biases
every number the project reports, which is why the code enforces the ordering structurally
(Section 4.4).

### 2.2 Concept drift

**Concept drift** is a change over time in `P(y | x)` (the label semantics), as opposed to
data drift (`P(x)` only). Types: **sudden** (regime change), **gradual**, **incremental**,
and **recurring**. Unadapted models degrade because their learned boundary becomes stale.

**ELEC2** (the project's dataset) is a classic drift benchmark precisely because its drifts
are real and documented: the NSW/Victoria electricity market experienced deregulation,
price-cap interventions, seasonal demand swings, and inter-state transfer constraints —
each shifts `P(price up | features)`.

### 2.3 ADWIN (ADaptive WINdowing)

ADWIN (Bifet & Gavaldà) is the drift detector used here (River's `drift.ADWIN`, δ = 0.002).

- It keeps a variable-length window `W` of recent values (here: the per-event **prequential
  error** `e_t = |y_t − ŷ_t| ∈ {0,1}`).
- For every candidate cut splitting `W = W₀ ∪ W₁`, it tests `H₀: μ(W₀) = μ(W₁)` using
  Hoeffding/Chernoff bounds with confidence δ (corrected for multiple cut tests).
- If some cut shows a statistically significant mean difference, the **older sub-window is
  dropped** and a drift alert is raised. Estimates therefore adapt at bounded memory.

Why error-rate input: drift matters when a model starts being *wrong* more often, so the
error sequence is the right signal (not raw features). In this project each adaptive model
runs its **own ADWIN on its own errors** — causally clean ("this model's errors changed →
adapt this model").

### 2.4 Hoeffding trees (VFDT) — the classical baseline

Domingos & Hulten (2000), the **reproduced** base paper.

- An incremental decision tree where each node keeps only per-attribute sufficient
  statistics (class counts per split candidate).
- A split is made only when the **Hoeffding bound** guarantees the empirically best
  attribute is (almost certainly) the true best: with `n` observations of a variable in
  range `R`, the true mean lies within `ε = sqrt(R² · ln(1/δ) / (2n))` of the empirical
  mean with probability `1−δ`. If `best − second_best > ε`, split — otherwise wait for
  more data.
- Result: constant time/memory per example, one pass, accuracy converging to the batch
  tree. Model **M1** is this tree, trained on the first `W₀ = 200` events and then frozen
  (a "static" control).

**Hoeffding Adaptive Tree (HAT)** (Bifet et al. 2009 — model **M2**, often conflated with
Domingos & Hulten; the correct citation matters in defense) adds drift handling *inside*
the tree: each node keeps an ADWIN-driven alternative subtree; when drift is detected, the
stale subtree is replaced by the challenger. M2 also learns on every event (never frozen),
which is why it is the strongest model in the results.

### 2.5 Quantum computing essentials (what you need to explain the models)

- **Qubit & statevector:** an `n`-qubit state is a unit vector `|ψ⟩ ∈ ℂ^(2ⁿ)`. This project
  uses `n = 5`, so each state is a 32-dimensional complex vector. Gates are unitary
  matrices applied to the state.
- **Parameterized gates:** rotations `R_x(θ), R_y(θ), R_z(θ)` whose angle `θ` is a classical
  number — the interface between data/parameters and the quantum state.
- **Measurement & expectation:** measuring an observable (e.g., parity `Z⊗Z⊗Z⊗Z⊗Z`, whose
  eigenvalues are ±1) on state `|ψ⟩` yields expectation `⟨ψ|O|ψ⟩ ∈ [−1, +1]`. With exact
  statevector simulation this is computed deterministically (no shot noise).
- **Simulator, not hardware:** all circuits run on `qiskit.quantum_info.Statevector` and
  PennyLane `default.qubit`. This is *simulated* quantum execution — reproducible and fast
  for 5 qubits, but it does **not** model NISQ hardware noise. The project says so
  explicitly in README/PLAN.

### 2.6 Variational Quantum Classifiers (VQC) — models M3/M4

A VQC is a quantum circuit with classical trainable angles, trained like a neural network:

```
|0⟩^5  --[ ZFeatureMap(x) ]-->  --[ RealAmplitudes(θ) ]-->  measure parity Z⊗5
         data encoding (fixed)     variational ansatz (trained)
```

- **Feature map (encoding):** Qiskit `ZFeatureMap` (reps=1) — Hadamard layer plus
  single-qubit Z-phase rotations carrying the 5 features. Features are normalized to
  `[0, π]` first because rotation angles are 2π-periodic.
- **Ansatz:** `RealAmplitudes` with linear entanglement, reps=2 → **15 trainable
  parameters** for 5 qubits (`5 × (reps+1)`). Entangling CNOTs create feature
  interactions that a purely classical linear model cannot cheaply represent — this is the
  "expressiveness" QML argument.
- **Readout:** parity observable `Z⊗5` (configurable `z0` = `Z` on qubit 0). Expectation
  `e ∈ [−1,1]` is mapped to `P(y=1) = (1 − e)/2`; prediction is the threshold at 0.5.
- **Loss:** binary cross-entropy between that probability and the label.
- **Optimizer — SPSA** (Simultaneous Perturbation Stochastic Approximation): instead of
  2×15 finite-difference evaluations, SPSA estimates the whole gradient with **two** loss
  evaluations using a random Rademacher perturbation vector `Δ ∈ {−1,+1}¹⁵`:

  `g_k = [L(θ + c_k Δ) − L(θ − c_k Δ)] / (2 c_k) · Δ⁻¹`

  with standard schedules `a_k = a·lr_scale / (k+5)^0.602`, `c_k = c / k^0.101`
  (exponents from the SPSA literature). SPSA was the best optimizer in Herbst et al.'s
  672-config search — one reason it is used here.
- **M3 (static):** train 40 SPSA steps on `W₀`, freeze forever.
- **M4 (drift-adaptive — novelty):** identical circuit and initialization to M3 (weights
  are literally copied at the end of `W₀` for a fair comparison); on each ADWIN alert,
  run a **warm-started** short SPSA fine-tune (10 steps, full sliding window `W_t`,
  reduced learning rate) instead of retraining from scratch — the computational
  impossibility of per-event retraining is the streaming constraint the design respects.

### 2.7 Quantum kernel methods — models M5/M6

The kernel trick: a classifier can operate in a high-dimensional feature space using only
inner products `K(x, x')`. A **quantum kernel** defines that inner product implicitly via
state overlaps:

- Encode each point into a state `|φ(x)⟩ = U(x)|0⟩` (here: PennyLane `AngleEmbedding` with
  Y-rotations, a CNOT chain + ring, then Z-rotations — an entangled 5-qubit embedding).
- **Fidelity kernel:** `K(x, x') = |⟨φ(x)|φ(x')⟩|² = Tr(ρ(x)ρ(x'))` — provably
  positive semi-definite, so it plugs into scikit-learn's `SVC(kernel="precomputed")`.
- Each kernel entry costs **two statevector simulations**; a training set of `m` points
  costs `m²` entries. This quadratic kernel cost is exactly why the design **caps the
  support set at K = 80 prototypes** and refits only on drift alerts (Section 3.4).
- At prediction time: compute `k(x, support)` (one state sim + cheap overlaps) and evaluate
  the SVM decision function; probabilities come from Platt scaling (`probability=True`).
- **M5 (static):** prototypes selected from `W₀`, SVC fit once, frozen.
- **M6 (drift-adaptive — novelty):** on ADWIN alert, replace the oldest 20% of prototypes
  with the newest points from sliding window `W_t` and refit. The static/adaptive pair
  isolates *the adaptation mechanism* as the only variable.

**Known QML caveat (good defense material):** in high-dimensional embeddings, quantum
kernels can approach the identity matrix (exponential dimension → near-orthogonal states),
which hurts generalization. With 5 qubits and modest data this is mild here, but it is part
of why the quantum kernel does not beat RBF (Section 5).

### 2.8 Classical kernel control — models M7/M8

A quantum-vs-quantum comparison cannot answer "is the quantum kernel *sample-efficient*?"
So the matrix includes a **classical RBF-SVM** (`gamma="scale"`) with the **identical**
support budget (K = 80), replacement rule (oldest 20% on alert), and SVC configuration as
M5/M6. This is the capacity-matched control for Hypothesis 4.

### 2.9 Base papers and how they map to this project

| Paper | Their work | Here |
|---|---|---|
| Domingos & Hulten 2000 | VFDT/Hoeffding tree on high-speed streams | **M1 — direct reproduction** |
| Bifet et al. 2009 | Adaptive Hoeffding trees (HAT + ADWIN) | M2 (classical adaptive anchor; cite correctly!) |
| Herbst et al. 2024 | Variational quantum *regressor* on Seoul Bike; 672-config search; SPSA best | M3/M4 — **adapted** to streaming binary classification (ZFeatureMap + SPSA choices follow their findings) |
| Tscharke et al. 2023 | Quantum-kernel SVR for semisupervised anomaly detection | M5/M6 — **adapted** fidelity-kernel classifier |

Only M1 is a reproduction; the quantum papers are **adaptations** to a different task and
dataset (PLAN.md §11 documents this explicitly). Never claim their MSE/AUC results were
reproduced.

---

## 3. Experiment Design

### 3.1 Dataset: ELEC2

- Australian NSW/Victoria electricity market, **45,312 half-hourly events** (≈ 2 years),
  shipped inside `river.datasets.Elec2`.
- **Task:** binary classification — will the NSW spot price be **UP** relative to its
  24-hour moving average? (label balance ≈ 43% UP in early segments).
- **Features (5):** `nswprice, nswdemand, vicprice, vicdemand, transfer` — chosen so 5
  features map to 5 qubits with **no lossy PCA compression**.
- Drifts are genuine (deregulation, price caps, seasons) — the stream is chronological, no
  shuffling, ever.

### 3.2 The 4×2 model matrix

| | Static (trained on W₀=200, frozen) | Drift-Adaptive (ADWIN-triggered updates) |
|---|---|---|
| Classical tree | **M1** Hoeffding Tree | **M2** HAT (updates every event + internal ADWIN) |
| Classical kernel | **M7** RBF-SVM (K=80) | **M8** RBF-SVM + support refresh |
| Quantum variational | **M3** VQC (15 params) | **M4** VQC + warm-start SPSA fine-tune |
| Quantum kernel | **M5** Fidelity-kernel SVM (K=80) | **M6** + support refresh |

Every model sees the **same events in the same order** with the **same scaler**. The only
differences are model class and update policy — that is what makes the comparison fair.
Update budgets differ by design and are logged (`N_Updates` column), so accuracy can be
read alongside how much learning each model was allowed.

### 3.3 The novelty: dual drift-adaptive quantum learning

Both quantum paradigms get the *same* lightweight adaptation pattern:

1. During stationary periods: **inference only** (low latency).
2. On that model's ADWIN alert: use the recent sliding window `W_t` (last 100 labeled
   events) to adapt:
   - **M4:** warm-started 10-step SPSA fine-tune (never from scratch — bounded `t_update`).
   - **M6/M8:** churn 20% of the support prototypes and refit the SVC.
3. Everything is logged (`adaptation_diagnostics`: weight delta norm for M4, support churn
   for M6/M8) so the report can *prove* the models moved at each alert.

Relative to the literature: drift-triggered retraining is known classically; the scoped
novelty claim is **the first systematic prequential evaluation of event-triggered drift
adaptation applied symmetrically to variational and kernel QML under streaming latency
constraints**. Do not claim more than that.

### 3.4 Why the budget caps exist (feasibility design)

A kernel SVM on `m` points needs `m²` quantum kernel evaluations; per-event refitting over
45k events is computationally impossible. Design answers:

- Support set capped at **K = 80** prototypes (last-K selection from the window).
- Refits **only on drift alerts** (189 in the full run) — not per event.
- VQC fine-tuning is 10 SPSA steps ≈ 20-24 loss evaluations on a subsampled batch, not a
  full retrain.
- All buffers are bounded (sliding window 100, rolling metrics 100, dedup cache 10k).
- Kafka producer pacing (25 ms default) is the pressure valve if simulation lags.

### 3.5 Hypotheses and the metrics that test them

| Hypothesis | Metric | Where computed |
|---|---|---|
| H1: adaptive quantum models survive drifts better than static ones | Post-warmup prequential accuracy & rolling F1 | `prequential_eval.py`, `benchmark_summary.csv` |
| H2: adaptation speeds recovery after drift | `N_recovery` (see below) | `drift_manager.RecoveryTracker` |
| H3: quantum has a latency cost | Decomposition `t_network + t_prep + t_encode + t_sim + t_update` | `latency_telemetry`, Panel 3 |
| H4: quantum kernel is sample-efficient | Accuracy vs support budget K ∈ {10,20,40,80}, M6 vs M8 | `--sample-efficiency` sweep |

**N_recovery definition (dip-then-sustained):** after a drift alert, a model must first
*degrade* below `pre-drift rolling accuracy − ε` (ε = 2pp). `N_recovery` counts events from
the alert until rolling accuracy (window 100) re-enters the band, confirmed by **20
consecutive** qualifying events (so one lucky window edge doesn't count). Models that never
degrade within 300 events score 0 ("never degraded"); episodes censored by the next drift
are excluded from means. This definition exists because a naive "back above baseline"
measure is polluted by the pre-drift events still inside the rolling window.

**Warmup policy:** M3–M8 cannot predict meaningfully before `W₀` fills, so during warmup
they predict the running majority class, and **headline accuracy is computed post-warmup**
(full-stream numbers are secondary). Without this, quantum models get penalized 200 blind
guesses.

---

## 4. Implementation Walkthrough

### 4.1 Architecture (data flow)

```
river.datasets.Elec2 (45,312 events)
  → src/producer/elec2_producer.py   serialize + pace into Kafka topic 'elec2-stream'
  → src/worker/feature_worker.py     validate, dedup, MinMax-scale to [0, π], buffers
  → src/evaluation/prequential_eval.py   8 models, test-then-train, drift, recovery
  → src/sink/influx_sink.py          4 measurements into InfluxDB bucket streaming_quantum
  → grafana/dashboards/...           6-panel real-time dashboard
```

`src/main.py` is the orchestrator (`--mode kafka|direct`, `--max-events`), with SIGINT/
SIGTERM graceful shutdown. `experiments/run_offline_benchmark.py` reuses the *same*
evaluator without Kafka for reproducible paper figures — live demo numbers and paper
numbers cannot diverge because there is one evaluation engine.

### 4.2 The event schema (Kafka message)

```json
{
  "event_id": 142,
  "timestamp": "1997-01-01T00:30:00Z",
  "producer_ts_ns": 1729170000000000000,
  "features": {"nswprice": 0.056, "nswdemand": 0.432, "vicprice": 0.003,
               "vicdemand": 0.412, "transfer": 0.612},
  "label": 1
}
```

`producer_ts_ns` enables the `t_network_ms` latency term (producer-send → worker-receive).

### 4.3 Preprocessing (`feature_worker.py`)

- **`OnlineAngleMinMaxScaler`:** expanding min/max fitted **past-only** (`partial_fit` on
  `x_t` only — never on future data), scaling to `[0, π]` with **clamping** so post-scaler
  updates cannot push angles out of range (angles wrap mod 2π — silently wrong encodings).
  Rejects NaN/Inf/missing features with explicit errors.
- **`SlidingWindowBuffer`:** bounded FIFO deques — `W₀` buffer (200) for initial fits,
  `W_t` buffer (100) for drift adaptation.
- **Dedup:** rolling event-id cache (10k) skips duplicate deliveries (Kafka is
  at-least-once).
- **Labels are held back:** `process_raw_payload` never touches training buffers;
  `record_post_inference` is called by the evaluator only after all predictions exist.

### 4.4 The prequential core (`prequential_eval.py`)

`evaluate_event` is the heart — three phases per event:

1. **STEP 1 — inference (labels withheld):** all 8 models predict. Uninitialized
   quantum/kernel models use the running majority class (warmup policy).
2. **STEP 2 — reveal & score:** label enters rolling metrics (accuracy/precision/recall/F1,
   window 100) and post-warmup counters; static/adaptive pair agreements (M3_M4, M5_M6,
   M7_M8, M2_M4) are tracked as rolling match fractions — the direct evidence of whether
   adaptation changes predictions. *Then* the event joins the buffers.
3. **STEP 3 — updates:** M1 learns until frozen at `W₀`, M2 learns every event. At `W₀`
   fill, M3–M8 fit on `W₀` and **M4 inherits M3's exact weights** (fair start). Each
   adaptive model's own ADWIN sees its own error; on its alert it registers recovery
   episodes (for *all* models, so recovery is comparable under the same shock) and runs its
   adaptation on `W_t`.

### 4.5 Drift & recovery (`drift_manager.py`)

- `ADWINDriftManager` wraps River `drift.ADWIN(delta=0.002)` per adaptive model; an alert
  produces a `DriftEventRecord` (estimation before/after, magnitude).
- `RecoveryTracker` implements the N_recovery state machine: episode per (model, drift) with
  `has_dipped → first_hit → sustained(k=20) → completed`, plus never-degraded (horizon 300)
  and censoring (a new drift archives a pending episode as incomplete).

### 4.6 Models (`models/`)

- **`classical_models.py`:** River `HoeffdingTreeClassifier` (frozen after `W₀`) and
  `HoeffdingAdaptiveTreeClassifier` with internal ADWIN + seed.
- **`vqc_models.py`:** Qiskit `ZFeatureMap` + `RealAmplitudes`; parameter binding split from
  simulation for clean `t_encode` vs `t_sim`; SPSA with configurable steps/`lr_scale`/
  batch size (0 = full window) and `parity`/`z0` readout — the knobs used for adaptation
  ablation; M4 records `last_weight_delta_norm`.
- **`qsvm_models.py`:** PennyLane embedding state-node; fidelity Gram matrix via
  `|state_matrix · state_matrix†|²`; `SVC(kernel="precomputed", probability=True)`;
  class-guard for single-class windows; M6 records `last_support_churn`; `StaticRBFKernelModel`
  / `DriftAdaptiveRBFKernelModel` mirror the quantum pair exactly.

### 4.7 Telemetry (`influx_sink.py`) and Grafana

Four measurements per event (batched per event, 8 models):

| Measurement | Contents |
|---|---|
| `prequential_accuracy` | rolling + cumulative + post-warmup accuracy/F1, y_true/y_pred, `n_updates` |
| `latency_telemetry` | t_network, t_prep, t_encode, t_sim, t_update, t_total (ms) |
| `drift_events` | alert flag, magnitude, triggered models, cumulative count, pair agreements |
| `adaptation_diagnostics` | M4 weight delta, M6/M8 support churn (only on alert events) |

Dashboard panels: 1A/1B throughput & event counter, **Panel 2** rolling accuracy of all 8
models with drift markers (the money chart), **Panel 3** latency breakdown per model
(pivoted bar chart — `t_sim` tells the quantum-cost story), Panel 4 M2-vs-M4 agreement
gauge. Panels use `v.timeRangeStart/Stop` so the Grafana time picker controls the window.

### 4.8 Configuration & reproducibility (`config.py`)

Everything is env-overridable (`.env.example` documents all of it): windows (200/100/100),
seeds (42), SPSA (40 initial / 10 adapt steps, lr 0.15, perturbation 0.10, lr_scale 0.75),
kernel (K=80, churn 20%, C=1.0), ADWIN δ=0.002, recovery (ε=0.02, sustain=20, horizon=300),
readout, and service endpoints. Secrets (Influx token, passwords) come from `.env` and are
**required** when the sink is enabled — `run_metadata.json` records versions + full config
with the token redacted on every benchmark run.

### 4.9 Tests (14, all passing)

Scaler bounds & non-finite rejection · serialization edge cases · dedup + no-pre-inference
buffer writes (leakage guard) · M1 freeze vs M2 continuous learning · VQC/QSVM/RBF
predict + adapt + churn · prequential integration (17 Influx points/event) · new recovery
semantics (dip, sustain, never-degraded, censoring) · warmup majority policy · update
counts · token validation. Run: `.venv/bin/python -m pytest tests/ -q`.

---

## 5. Results (full 45,312-event run, 189 drift alerts)

From `experiments/plots/benchmark_summary.csv` (all values post-warmup):

| Model | Accuracy | Final rolling F1 | Mean t_sim (ms) | N_updates | Mean N_recovery |
|---|---|---|---|---|---|
| M1 Static HT | 0.651 | 0.00 | 0.016 | 45,312* | 91.3 |
| M2 Adaptive HAT | **0.808** | 0.83 | 0.030 | 45,312 | 95.3 |
| M3 Static VQC | 0.528 | 0.65 | 1.10 | 1 | 104.4 |
| M4 Adaptive VQC | 0.569 | 0.23 | 1.03 | 54 | 110.7 |
| M5 Static QKernel | 0.591 | 0.11 | 2.00 | 1 | 106.3 |
| M6 Adaptive QKernel | 0.642 | 0.41 | 1.80 | 58 | 105.4 |
| M7 Static RBF | 0.623 | 0.00 | 0.094 | 1 | 88.8 |
| M8 Adaptive RBF | 0.644 | 0.64 | 0.084 | 80 | 109.6 |

\* M1's counter increments per event though it stops learning at W₀ — read "learned only
during W₀".

**How to read it:**

1. **Adaptation works (H1 confirmed):** M4 > M3 (+4.1pp), M6 > M5 (+5.1pp), M8 > M7
   (+2.1pp). The drift-adaptive mechanism — the project's novelty — has a measurable
   effect across 189 real drift alerts.
2. **No quantum advantage (H4 answered negatively):** M6 (0.642) ≈ M8 (0.644) — the
   quantum kernel ties its capacity-matched classical twin while `t_sim` is ~20× larger
   (1.8 ms vs 0.08 ms). Early in the stream (first 3k events) the quantum kernel leads
   slightly at tiny budgets (K=10: 0.704 vs 0.682; K=20: 0.726 vs 0.624) and ties at
   K≥40 (`sample_efficiency.csv`) — a sample-efficiency edge that evaporates as data grows.
3. **Classical streaming ML wins outright (H2):** M2 (0.808) beats every kernel and quantum
   model, with sub-millisecond inference. Static models visibly collapse in late segments
   (final rolling F1 ≈ 0), the textbook drift failure mode.
4. **Latency (H3):** quantum encode+sim ≈ 1–2 ms/event vs ≈ 0.02–0.09 ms classical; M4's
   mean update time (2.8 ms) is dominated by occasional SPSA fine-tunes.

**Scientific-integrity notes:** results are what the data says, not what the pitch hoped
for. The report should present the negative quantum result as a first-class finding — the
rubric explicitly rewards honest negative reporting — with the working adaptation
mechanism as the contribution.

---

## 6. Limitations & Threats to Validity

- **Simulated quantum only:** statevector, no shot noise, no hardware noise. NISQ devices
  would add error and latency; direction of that effect is unfavorable to QML here.
- **One dataset/task:** ELEC2 binary classification. Broader claims need more streams
  (Seoul Bike, NSL-KDD were scoped out — PLAN.md §11).
- **Labels assumed immediate** (replayed labeled data). Delayed/weak labels are out of
  scope.
- **Prototype selection is recency-based** (last-K), not clustering-based; a smarter
  selector could change kernel results.
- **Update-budget asymmetry:** M2 learns per event, M4/M6/M8 only on alerts; `N_Updates`
  columns make this visible but a fixed-budget ablation would be even tighter.
- **ADWIN coupling:** a model's detector adapts that model only; shared-detection variants
  might behave differently (documented choice).
- **Platt scaling** on small support sets makes probabilities coarse near 0/1.

---

## 7. Defense Prep — Likely Questions & Short Answers

**Q: What exactly is the novelty?** A: event-triggered drift adaptation applied
*symmetrically* to variational (warm-started SPSA) and kernel (support churn) QML under
streaming latency budgets, evaluated prequentially on a real drifting stream — and shown to
work (+4–5pp over static) even though the quantum models don't beat classical ones.

**Q: Why not real quantum hardware?** A: course compute and determinism; 5-qubit exact
statevector simulation gives reproducible latency decomposition and clean algorithmic
comparisons. Hardware noise would not improve QML's position.

**Q: Why ELEC2?** A: real, well-documented drift; 5 features → 5 qubits without lossy PCA;
standard streaming benchmark; binary label maps to a single expectation readout.

**Q: Why is M2 attributed to Bifet, not Domingos & Hulten?** A: M1 (plain Hoeffding tree)
is Domingos & Hulten 2000 — the reproduced baseline. HAT + ADWIN subtree swapping is Bifet
et al. 2009.

**Q: Isn't warm-start retraining on drift already known?** A: classically, yes — we scope
the claim to its systematic application to *both* QML paradigms under streaming
constraints, with per-model detectors and prequential evaluation.

**Q: How do you prevent leakage?** A: structurally — models predict before labels touch any
buffer (`record_post_inference`), the scaler fits past-only, evaluation is chronological,
and tests assert the ordering. Warmup events are excluded from headline metrics.

**Q: Why does the quantum kernel tie RBF instead of winning?** A: at 5 qubits the fidelity
kernel lives in a modest feature space; RBF with `gamma="scale"` is already flexible in 5
dimensions. Exponential-dimensional advantage needs larger embeddings, which brings the
near-identity-kernel problem and quadratic simulation cost.

**Q: How is N_recovery defined and why?** A: dip below pre-drift − 2pp must occur; recovery
is re-entry into the band sustained 20 events; never-degraded = 0; censored episodes
excluded. The definition avoids the stale-rolling-window artifact of naive baselines.

**Q: Is the pipeline genuinely streaming?** A: yes — live Kafka topic with paced producer,
consumer worker, per-event evaluation and telemetry, real-time Grafana; the offline runner
shares the same evaluator and is only for reproducible figures.

**Q: What would you do next?** A: more streams (recurring-drift datasets), a
projected-quantum-kernel variant, fixed-update-budget ablations, shot-noise robustness
study, and delayed-label protocols.

---

## 8. Quick Orientation Map

| I want to… | Go to |
|---|---|
| Run the pipeline / demo | `README.md`, `experiments/demo_script.md` |
| Understand design decisions | `PLAN.md` (esp. §5, §11 deviations) |
| See the evaluation protocol | `src/evaluation/prequential_eval.py` |
| Understand the quantum models | `src/models/vqc_models.py`, `src/models/qsvm_models.py` |
| Understand drift/recovery math | `src/evaluation/drift_manager.py` |
| See numbers/plots | `experiments/plots/` |
| Check test coverage | `tests/test_pipeline.py` |
