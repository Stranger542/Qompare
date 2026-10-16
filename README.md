# Qompare — Quantum vs Classical Streaming Data Analytics

An end-to-end streaming analytics pipeline comparing **Classical Streaming Machine Learning**
(`River` Hoeffding trees) against **Static and Drift-Adaptive Quantum Machine Learning**
(`Qiskit` VQC, `PennyLane` quantum-kernel SVM) on the non-stationary `ELEC2` electricity
market stream (45,312 half-hourly events), with real data flowing through **Kafka →
stream worker → 8-model prequential evaluation → InfluxDB → Grafana**.

**Research question:** does the expressiveness / sample-efficiency appeal of QML survive
real-time latency bounds and concept drift of a true streaming environment?

The system design and experimental methodology are specified in [`PLAN.md`](PLAN.md);
scope changes from the original [`Proposal.md`](Proposal.md) are documented in
`PLAN.md` Section 11 ("Deviations from Proposal & Scope Notes").

## Model matrix (4×2)

| | Static (trained on W₀, frozen) | Drift-Adaptive (ADWIN-triggered updates) |
|---|---|---|
| **Classical tree** | M1 Hoeffding Tree (Domingos & Hulten 2000 reproduction) | M2 Hoeffding Adaptive Tree (Bifet et al. 2009) |
| **Classical kernel** | M7 RBF-SVM (support budget K=80) | M8 RBF-SVM + sliding support refresh |
| **Quantum variational** | M3 VQC (ZFeatureMap + RealAmplitudes, 15 params, adapted from Herbst et al. 2024) | M4 VQC + warm-started SPSA fine-tuning on drift |
| **Quantum kernel** | M5 Fidelity-kernel SVM (adapted from Tscharke et al. 2023) | M6 Fidelity-kernel SVM + sliding support refresh |

All quantum circuits run on **exact statevector simulation** (`qiskit.quantum_info.Statevector`,
PennyLane `default.qubit`) — this is *simulated* quantum execution, not quantum hardware.

## Prerequisites

- Python `>=3.10, <3.13` (experiments were recorded under CPython 3.11)
- Docker + Docker Compose v2 (for the Kafka/InfluxDB/Grafana stack)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set INFLUXDB_TOKEN, INFLUXDB_ADMIN_PASSWORD, GRAFANA_ADMIN_PASSWORD.
# Generate secrets with: python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Dependency versions used for recorded experiments are captured automatically in
`experiments/plots/run_metadata.json` on every benchmark run.

## Running

### 1. Infrastructure stack

```bash
docker compose up -d          # Kafka (+Zookeeper), InfluxDB v2, Grafana
docker compose ps             # wait until all services are healthy
```

- Grafana: http://localhost:3000 (pre-provisioned dashboard `streaming_quantum_analytics`)
- InfluxDB: http://localhost:8086, bucket `streaming_quantum`

### 2. Live streaming pipeline (Kafka mode)

```bash
# Terminal 1 — replay ELEC2 into Kafka topic 'elec2-stream' (25 ms pacing default)
python -m src.producer.elec2_producer

# Terminal 2 — consume, evaluate all 8 models prequentially, stream telemetry
python -m src.main --mode kafka
# optional: --max-events 500   (smoke test)
```

### 3. Offline benchmark (no Kafka; generates report figures)

```bash
python experiments/run_offline_benchmark.py --max-events 1000     # quick run
python experiments/run_offline_benchmark.py --max-events 45312    # full stream
```

Outputs to `experiments/plots/`: prequential accuracy curves (with ADWIN drift markers),
latency breakdown chart, `benchmark_summary.csv`, `ieee_results_table.tex`,
`run_metadata.json` (versions, seed, config — secrets excluded).

### 4. Tests

```bash
python -m pytest tests/ -q
```

Unit + integration tests cover preprocessing bounds, serialization edge cases, dedup,
leak-free test-then-train ordering, model freeze/adaptation behavior, the drift-recovery
metric, and InfluxDB point formatting.

## Evaluation methodology

- **Prequential (test-then-train):** every model predicts *before* the label is revealed;
  labels only enter training buffers after inference (`record_post_inference`).
- **Warmup policy:** M3–M8 fit on the initial window W₀ = 200 events. Headline accuracy
  metrics are computed post-warmup; full-stream metrics are reported as secondary.
- **Drift detection:** one ADWIN detector (δ = 0.002) per adaptive model pair on the
  prequential error sequence; alerts trigger warm-started SPSA fine-tuning (M4) or
  support-set refresh (M6/M8).
- **Drift recovery (N_recovery):** a model must first degrade below
  (pre-drift rolling accuracy − 2 pp); recovery then counts events from the alert until
  rolling accuracy re-enters the band, confirmed by 20 consecutive qualifying events.
  Models that never degrade within the 300-event horizon score 0; episodes censored by
  the next drift are excluded from means.
- **Reproducibility:** all seeds (`RANDOM_SEED=42`), dependency versions, and full config
  are recorded per run in `run_metadata.json`.

## Configuration

All knobs live in `src/config.py` and can be overridden via environment variables —
see `.env.example` for the complete annotated list (windows, SPSA settings, support-set
budget, ADWIN δ, recovery definition, readout observable, credentials).

## Resetting experiment state between runs

```bash
docker compose down -v                       # wipes Kafka topics + InfluxDB data + Grafana state
docker compose up -d
```

The offline benchmark is stateless — each invocation rebuilds all model state from the
deterministic stream and overwrites `experiments/plots/` artifacts.

## Known limitations

- Quantum models are simulated (statevector); results do not transfer directly to NISQ
  hardware. Shot-noise robustness is not modeled by default.
- Labels are assumed to arrive with each event (replayed labeled dataset); delayed-label
  streams are out of scope.
- The Herbst et al. and Tscharke et al. models are *adapted* to streaming binary
  classification; their published MSE/AUC numbers are not directly comparable (PLAN.md §11).

## Repository layout

```
docker-compose.yml        Kafka, Zookeeper, InfluxDB v2, Grafana (pinned images, healthchecks)
src/                      producer / worker / models / evaluation / sink / orchestrator
tests/                    unit + integration tests
experiments/              offline benchmark runner + generated plots/tables
grafana/                  dashboard + provisioning
PLAN.md, Proposal.md      design spec (incl. deviations) and original proposal
```
