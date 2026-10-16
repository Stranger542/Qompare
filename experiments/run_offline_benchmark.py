# experiments/run_offline_benchmark.py
"""Offline benchmark runner that generates IEEE report plots and LaTeX evaluation tables."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# Allow running as `python experiments/run_offline_benchmark.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane
import qiskit
import river
import sklearn
from river import datasets

from src.config import PipelineConfig, configure_logging
from src.evaluation.prequential_eval import PrequentialEvaluator
from src.producer.elec2_producer import serialize_elec2_event
from src.worker.feature_worker import StreamFeatureWorker

logger = logging.getLogger(__name__)


def _run_stream(
    cfg: PipelineConfig, max_events: int, progress_every: int = 2500
) -> "tuple[PrequentialEvaluator, object]":
    """Replay the ELEC2 stream through the shared prequential evaluator (stateless)."""
    worker = StreamFeatureWorker(cfg)
    evaluator = PrequentialEvaluator(cfg, worker)
    final_result = None
    for idx, (raw_x, raw_y) in enumerate(datasets.Elec2()):
        if idx >= max_events:
            break
        payload = serialize_elec2_event(idx, raw_x, raw_y)
        event = worker.process_raw_payload(payload)
        if event is None:
            continue
        final_result = evaluator.evaluate_event(event)
        if progress_every and (idx + 1) % progress_every == 0:
            logger.info("Progress: %d/%d events evaluated.", idx + 1, max_events)
    return evaluator, final_result


def run_benchmark(max_events: int = 1000, output_dir: Path = Path("experiments/plots")) -> None:
    """Run deterministic prequential benchmark across M1-M8 and save plots + LaTeX metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(influxdb_enabled=False)
    cfg.validate()
    configure_logging(cfg.log_level)

    worker = StreamFeatureWorker(cfg)
    evaluator = PrequentialEvaluator(cfg, worker)
    model_names = evaluator.model_names

    event_ids: List[int] = []
    rolling_acc_history: Dict[str, List[float]] = {m: [] for m in model_names}
    encode_latencies: Dict[str, List[float]] = {m: [] for m in model_names}
    sim_latencies: Dict[str, List[float]] = {m: [] for m in model_names}
    update_latencies: Dict[str, List[float]] = {m: [] for m in model_names}
    drift_points: List[Dict[str, object]] = []
    final_result = None

    dataset = datasets.Elec2()
    for idx, (raw_x, raw_y) in enumerate(dataset):
        if idx >= max_events:
            break
        payload = serialize_elec2_event(idx, raw_x, raw_y)
        event = worker.process_raw_payload(payload)
        if event is None:
            continue

        res = evaluator.evaluate_event(event)
        final_result = res
        event_ids.append(res.event_id)
        for drift_rec in res.drift_events:
            drift_points.append(
                {
                    "event_id": drift_rec.event_id,
                    "models": drift_rec.triggered_models,
                    "magnitude": drift_rec.magnitude,
                }
            )

        for m in model_names:
            snap = res.model_snapshots[m]
            rolling_acc_history[m].append(snap.rolling_accuracy)
            encode_latencies[m].append(snap.t_encode_ms)
            sim_latencies[m].append(snap.t_sim_ms)
            update_latencies[m].append(snap.t_update_ms)

        if (idx + 1) % 2500 == 0:
            logger.info("Progress: %d/%d events evaluated.", idx + 1, max_events)

    drift_x = [int(d["event_id"]) for d in drift_points]

    # 1. Save Prequential Accuracy Curve Plot with ADWIN Drift Markers
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    for m in model_names:
        ax.plot(event_ids, rolling_acc_history[m], label=m, linewidth=1.6)
    for idx_d, dp in enumerate(drift_x):
        ax.axvline(
            x=dp,
            color="crimson",
            linestyle="--",
            alpha=0.5,
            label="ADWIN Drift Alert" if idx_d == 0 else None,
        )
    ax.set_title("Rolling Prequential Accuracy Across 4x2 Model Matrix (ELEC2 Stream)")
    ax.set_xlabel("Chronological Streaming Event ID")
    ax.set_ylabel("Rolling Prequential Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "prequential_accuracy_curves.png")
    plt.close(fig)

    # 2. Save Fine-Grained Latency Breakdown Bar Chart (Log Scale)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    x_pos = np.arange(len(model_names))
    mean_enc = [float(np.mean(encode_latencies[m])) for m in model_names]
    mean_sim = [float(np.mean(sim_latencies[m])) for m in model_names]
    mean_upd = [float(np.mean(update_latencies[m])) for m in model_names]

    width = 0.26
    ax.bar(x_pos - width, mean_enc, width, label="State / Circuit Encode (ms)")
    ax.bar(x_pos, mean_sim, width, label="Forward Simulation / Inference (ms)")
    ax.bar(x_pos + width, mean_upd, width, label="Online / Drift Update (ms)")
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("_", "\n") for m in model_names], fontsize=8)
    ax.set_ylabel("Mean Latency per Event (ms, log scale)")
    ax.set_title("Fine-Grained Latency Decomposition: Classical vs Quantum Streaming")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_breakdown.png")
    plt.close(fig)

    # 3. Export CSV & IEEE LaTeX Summary Table (headline = post-warmup accuracy)
    csv_path = output_dir / "benchmark_summary.csv"
    latex_path = output_dir / "ieee_results_table.tex"

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "Model",
                "Cumulative_Accuracy_PostWarmup",
                "Cumulative_Accuracy_FullStream",
                "Final_Rolling_F1",
                "Mean_Encode_ms",
                "Mean_Sim_ms",
                "Mean_Update_ms",
                "N_Updates",
                "Mean_Recovery_Events",
            ]
        )
        for m in model_names:
            snap = final_result.model_snapshots[m]
            _, _, _, r_f1 = evaluator.metrics[m].compute_rolling()
            rec_ev = evaluator.recovery_tracker.mean_recovery_events(m)
            writer.writerow(
                [
                    m,
                    f"{snap.cumulative_accuracy_post_warmup:.4f}",
                    f"{evaluator.metrics[m].cumulative_accuracy:.4f}",
                    f"{r_f1:.4f}",
                    f"{np.mean(encode_latencies[m]):.4f}",
                    f"{np.mean(sim_latencies[m]):.4f}",
                    f"{np.mean(update_latencies[m]):.4f}",
                    evaluator.update_counts[m],
                    f"{rec_ev:.1f}",
                ]
            )

    with latex_path.open("w", encoding="utf-8") as f_tex:
        f_tex.write("\\begin{table}[htbp]\n\\centering\n")
        f_tex.write(
            "\\caption{Prequential Accuracy (post-warmup), Latency Decomposition, and Drift Recovery on ELEC2}\n"
        )
        f_tex.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f_tex.write(
            "Model & Acc. & Rolling F1 & $t_{\\text{enc}}$ (ms) & $t_{\\text{sim}}$ (ms) & $N_{\\text{recovery}}$ \\\\\n\\hline\n"
        )
        for m in model_names:
            snap = final_result.model_snapshots[m]
            _, _, _, r_f1 = evaluator.metrics[m].compute_rolling()
            rec_ev = evaluator.recovery_tracker.mean_recovery_events(m)
            clean_m = m.replace("_", "\\_")
            f_tex.write(
                f"{clean_m} & {snap.cumulative_accuracy_post_warmup:.3f} & {r_f1:.3f} & {np.mean(encode_latencies[m]):.3f} & {np.mean(sim_latencies[m]):.3f} & {rec_ev:.1f} \\\\\n"
            )
        f_tex.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # 4. Save Reproducibility Metadata (Section 10.5) — secrets excluded
    config_dump = asdict(cfg)
    config_dump["influxdb_token"] = "<redacted>" if config_dump.get("influxdb_token") else ""
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "river": river.__version__,
            "qiskit": qiskit.__version__,
            "pennylane": pennylane.__version__,
        },
        "config": config_dump,
        "events_evaluated": len(event_ids),
        "adwin_drift_alerts": drift_points,
        "pair_agreements": final_result.pair_agreements if final_result else {},
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(
        "Saved benchmark plots and IEEE LaTeX table to %s (%d events, %d drift alerts).",
        output_dir,
        len(event_ids),
        len(drift_x),
    )


def run_sample_efficiency(
    max_events: int = 3000,
    budgets: List[int] | None = None,
    output_dir: Path = Path("experiments/plots"),
) -> None:
    """Hypothesis 4 sweep: M6 (quantum kernel) vs M8 (RBF) accuracy across support budgets K."""
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = budgets or [10, 20, 40, 80]

    results: Dict[str, Dict[int, float]] = {"M6": {}, "M8": {}}
    for k in budgets:
        cfg = PipelineConfig(influxdb_enabled=False, qsvm_max_support_size=k)
        cfg.validate()
        configure_logging(cfg.log_level)
        _, final = _run_stream(cfg, max_events, progress_every=0)
        snap6 = final.model_snapshots["M6_Quantum_Adaptive_QSVM"]
        snap8 = final.model_snapshots["M8_Control_RBF_Adaptive"]
        results["M6"][k] = snap6.cumulative_accuracy_post_warmup
        results["M8"][k] = snap8.cumulative_accuracy_post_warmup
        logger.info(
            "Support budget K=%d | post-warmup acc: M6 (quantum)=%.4f, M8 (RBF)=%.4f",
            k,
            results["M6"][k],
            results["M8"][k],
        )

    csv_path = output_dir / "sample_efficiency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Support_Budget_K", "M6_Quantum_Acc", "M8_RBF_Acc"])
        for k in budgets:
            writer.writerow([k, f"{results['M6'][k]:.4f}", f"{results['M8'][k]:.4f}"])

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(budgets, [results["M6"][k] for k in budgets], "o-", label="M6 Quantum Kernel")
    ax.plot(budgets, [results["M8"][k] for k in budgets], "s--", label="M8 Classical RBF")
    ax.set_xlabel("Support Set Budget K (training points)")
    ax.set_ylabel("Post-Warmup Prequential Accuracy")
    ax.set_title("Sample Efficiency: Quantum vs Classical Kernel (ELEC2)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sample_efficiency.png")
    plt.close(fig)
    logger.info("Saved sample-efficiency sweep to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Benchmark & Plot Generator")
    parser.add_argument(
        "--max-events",
        type=int,
        default=1000,
        help="Number of chronological ELEC2 events to benchmark (use 45312 for full dataset).",
    )
    parser.add_argument(
        "--sample-efficiency",
        action="store_true",
        help="Run the Hypothesis 4 support-budget sweep (M6 vs M8) instead of the main benchmark.",
    )
    args = parser.parse_args()
    if args.sample_efficiency:
        run_sample_efficiency(max_events=args.max_events)
    else:
        run_benchmark(max_events=args.max_events)


if __name__ == "__main__":
    main()
