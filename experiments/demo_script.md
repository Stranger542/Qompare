# Demo Script — Quantum vs Classical Streaming Analytics (≤ 10 minutes)

Live walkthrough of the Qompare pipeline. Keep the Grafana dashboard visible in a browser
window throughout (http://localhost:3000, dashboard `streaming_quantum_analytics`).

## 0. Before the demo (do NOT skip)

```bash
cd Qompare
cp .env.example .env       # set INFLUXDB_TOKEN, INFLUXDB_ADMIN_PASSWORD, GRAFANA_ADMIN_PASSWORD
source .venv/bin/activate
docker compose down -v     # clean state: fresh Kafka topics, Influx bucket, Grafana state
docker compose up -d
docker compose ps          # wait for all four services to be healthy
```

## 1. Architecture & research question (60 s)

- Open `PLAN.md` Section 2 architecture diagram.
- One-liner: "Does the QML expressiveness advantage survive streaming latency and concept
  drift? We run an 8-model 4×2 matrix — classical/quantum × static/adaptive — on the ELEC2
  electricity market stream, evaluating prequentially (predict first, learn after)."

## 2. Start the stream (30 s)

```bash
python -m src.producer.elec2_producer --max-events 2000
```
Show the JSON event schema going into Kafka topic `elec2-stream` (log output).

## 3. Start the evaluation pipeline (30 s)

```bash
python -m src.main --mode kafka --max-events 2000
```
Point at the log lines: per-50-event rolling accuracy for M2/M4/M6/M8.

## 4. Watch the live dashboard (2–3 min)

- Panel 1: throughput gauge + cumulative event counter climbing.
- Panel 2: rolling accuracy curves for all 8 models with ADWIN drift markers appearing.
- Panel 3: latency breakdown — classical sub-millisecond vs quantum encode/sim bars
  (log scale; the quantum overhead story).
- Panel 4: M2 vs M4 prediction-agreement line.
- Wait for (or point at) a `[DRIFT ALERT]` log line — the moment of the demo.

## 5. Explain the adaptation mechanism (60 s)

On the drift alert: warm-started SPSA fine-tuning (M4) / support-set refresh (M6, M8) on the
sliding window W_t. Show the `adaptation_diagnostics` measurement (weight delta / support
churn) proving the models actually moved.

## 6. Offline results for the report (2 min)

```bash
python experiments/run_offline_benchmark.py --max-events 45312   # full stream (pre-run!)
python experiments/run_offline_benchmark.py --sample-efficiency --max-events 3000
```
Show `experiments/plots/`:
- `prequential_accuracy_curves.png` — accuracy through drifts with markers.
- `latency_breakdown.png` — the quantum/classical trade-off.
- `sample_efficiency.png` — Hypothesis 4: quantum kernel vs RBF across support budgets.
- `benchmark_summary.csv` / `ieee_results_table.tex` — headline numbers.

## 7. Wrap-up & honest findings (60 s)

- Prequential protocol and reproducibility (seeds, versions in `run_metadata.json`).
- What the results say about the quantum advantage under streaming constraints
  (including negative results, if that is what the numbers show).
- Limitations: statevector simulation only; labels assumed immediate.

## Teardown

```bash
docker compose down       # add -v to wipe state for a clean re-run
```

## Pre-demo checklist

- [ ] Full 45,312-event benchmark completed; plots regenerated
- [ ] Sample-efficiency sweep completed
- [ ] `docker compose up -d` healthy before audience arrives
- [ ] Producer + worker commands copied to terminal tabs, un-run
- [ ] Grafana dashboard opens at http://localhost:3000 without auth prompts mid-demo
- [ ] Tests green (`python -m pytest tests/ -q`) in case of a "how do you validate" question
