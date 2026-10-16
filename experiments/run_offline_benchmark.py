# experiments/run_offline_benchmark.py
"""Offline benchmark runner that generates IEEE report plots and LaTeX evaluation tables."""

from __future__ import annotations
# Add imports at top of experiments/run_offline_benchmark.py:
import json
import platform
from dataclasses import asdict
import pennylane
import qiskit
import river
import sklearn
from src.models.qsvm_models import ClassicalRBFKernelModel, DriftAdaptiveQuantumKernelModel

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from river import datasets

from src.config import PipelineConfig, configure_logging
from src.evaluation.prequential_eval import PrequentialEvaluator
from src.producer.elec2_producer import serialize_elec2_event
from src.worker.feature_worker import StreamFeatureWorker

logger = logging.getLogger(__name__)


def run_benchmark(max_events: int = 1000, output_dir: Path = Path("experiments/plots")) -> None:
    """Run deterministic prequential benchmark across M1-M6 and save plots + LaTeX metrics."""
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
    drift_points: List[int] = []

    dataset = datasets.Elec2()
    for idx, (raw_x, raw_y) in enumerate(dataset):
        if idx >= max_events:
            break
        payload = serialize_elec2_event(idx, raw_x, raw_y)
        event = worker.process_raw_payload(payload)
        if event is None:
            continue

        res = evaluator.evaluate_event(event)
        event_ids.append(res.event_id)
        if res.drift_event is not None:
            drift_points.append(res.event_id)

        for m in model_names:
            snap = res.model_snapshots[m]
            rolling_acc_history[m].append(snap.rolling_accuracy)
            encode_latencies[m].append(snap.t_encode_ms)
            sim_latencies[m].append(snap.t_sim_ms)
            update_latencies[m].append(snap.t_update_ms)

    # 1. Save Prequential Accuracy Curve Plot with ADWIN Drift Markers
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    for m in model_names:
        ax.plot(event_ids, rolling_acc_history[m], label=m, linewidth=1.6)
    for idx_d, dp in enumerate(drift_points):
        ax.axvline(
            x=dp,
            color="crimson",
            linestyle="--",
            alpha=0.5,
            label="ADWIN Drift Alert" if idx_d == 0 else None,
        )
    ax.set_title("Rolling Prequential Accuracy Across 3x2 Model Matrix (ELEC2 Stream)")
    ax.set_xlabel("Chronological Streaming Event ID")
    ax.set_ylabel("Rolling Prequential Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    acc_plot_path = output_dir / "prequential_accuracy_curves.png"
    fig.savefig(acc_plot_path)
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
    lat_plot_path = output_dir / "latency_breakdown.png"
    fig.savefig(lat_plot_path)
    plt.close(fig)

    # 3. Export CSV & IEEE LaTeX Summary Table
    csv_path = output_dir / "benchmark_summary.csv"
    latex_path = output_dir / "ieee_results_table.tex"

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "Model",
                "Cumulative_Accuracy",
                "Final_Rolling_F1",
                "Mean_Encode_ms",
                "Mean_Sim_ms",
                "Mean_Update_ms",
                "Mean_Recovery_Events",
            ]
        )
        for m in model_names:
            cum_acc = evaluator.metrics[m].cumulative_accuracy
            _, _, _, r_f1 = evaluator.metrics[m].compute_rolling()
            rec_ev = evaluator.recovery_tracker.mean_recovery_events(m)
            writer.writerow(
                [
                    m,
                    f"{cum_acc:.4f}",
                    f"{r_f1:.4f}",
                    f"{np.mean(encode_latencies[m]):.4f}",
                    f"{np.mean(sim_latencies[m]):.4f}",
                    f"{np.mean(update_latencies[m]):.4f}",
                    f"{rec_ev:.1f}",
                ]
            )

    with latex_path.open("w", encoding="utf-8") as f_tex:
        f_tex.write("\\begin{table}[htbp]\n\\centering\n")
        f_tex.write(
            "\\caption{Prequential Accuracy, Latency Decomposition, and Drift Recovery on ELEC2}\n"
        )
        f_tex.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f_tex.write(
            "Model & Cum. Acc. & Rolling F1 & $t_{\\text{enc}}$ (ms) & $t_{\\text{sim}}$ (ms) & $N_{\\text{recovery}}$ \\\\\n\\hline\n"
        )
        for m in model_names:
            cum_acc = evaluator.metrics[m].cumulative_accuracy
            _, _, _, r_f1 = evaluator.metrics[m].compute_rolling()
            rec_ev = evaluator.recovery_tracker.mean_recovery_events(m)
            clean_m = m.replace("_", "\\_")
            f_tex.write(
                f"{clean_m} & {cum_acc:.3f} & {r_f1:.3f} & {np.mean(encode_latencies[m]):.3f} & {np.mean(sim_latencies[m]):.3f} & {rec_ev:.1f} \\\\\n"
            )
        f_tex.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    logger.info(
        "Saved benchmark plots and IEEE LaTeX table to %s (%d events, %d drift alerts).",
        output_dir,
        len(event_ids),
        len(drift_points),
    )
    
    # 4. Save Reproducibility Metadata (Section 10.5)
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
        "config": asdict(cfg),
        "events_evaluated": len(event_ids),
        "adwin_drift_alerts": drift_points,
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Benchmark & Plot Generator")
    parser.add_argument(
        "--max-events",
        type=int,
        default=1000,
        help="Number of chronological ELEC2 events to benchmark (use 45312 for full dataset).",
    )
    args = parser.parse_args()
    run_benchmark(max_events=args.max_events)


if __name__ == "__main__":
    main()


