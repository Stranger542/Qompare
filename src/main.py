# src/main.py
"""Main orchestrator running the end-to-end Quantum vs Classical Streaming Pipeline."""

from __future__ import annotations

import argparse
import logging
import signal
from types import FrameType
from typing import Iterator, Optional

from river import datasets

from src.config import PipelineConfig, configure_logging
from src.evaluation.prequential_eval import PrequentialEvaluator
from src.producer.elec2_producer import serialize_elec2_event
from src.sink.influx_sink import InfluxTelemetrySink
from src.worker.feature_worker import ProcessedStreamEvent, StreamFeatureWorker

logger = logging.getLogger(__name__)


class StreamingPipelineOrchestrator:
    """Coordinates stream ingestion, preprocessing, 6-model evaluation, and InfluxDB telemetry."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.worker = StreamFeatureWorker(config)
        self.evaluator = PrequentialEvaluator(config, self.worker)
        self.sink = InfluxTelemetrySink(config)
        self._stop_requested: bool = False

    def request_stop(self, signum: Optional[int] = None, frame: Optional[FrameType] = None) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Shutdown signal received (signum=%s); finishing active event...", signum)
        self._stop_requested = True

    def _iter_direct_events(self, max_events: Optional[int]) -> Iterator[ProcessedStreamEvent]:
        """Yield ProcessedStreamEvents directly from river.datasets.Elec2 without Kafka."""
        dataset = datasets.Elec2()
        for event_id, (raw_x, raw_y) in enumerate(dataset):
            if max_events is not None and event_id >= max_events:
                break
            payload = serialize_elec2_event(event_id, raw_x, raw_y)
            processed = self.worker.process_raw_payload(payload)
            if processed is not None:
                yield processed

    def run(self, mode: str = "kafka", max_events: Optional[int] = None) -> int:
        """Execute the streaming pipeline loop.

        Args:
            mode: 'kafka' to consume from Apache Kafka topic, or 'direct' for local replay.
            max_events: Optional limit on number of events to process.

        Returns:
            Number of events evaluated.
        """
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        self.sink.connect()
        event_stream = (
            self.worker.iter_kafka_events(max_events=max_events)
            if mode == "kafka"
            else self._iter_direct_events(max_events=max_events)
        )

        processed_count = 0
        try:
            for event in event_stream:
                if self._stop_requested:
                    break
                result = self.evaluator.evaluate_event(event)
                self.sink.write_result(result)
                processed_count += 1

                if result.drift_events:
                    for drift_rec in result.drift_events:
                        logger.info(
                            "[DRIFT ALERT] event_id=%d | model=%s | magnitude=%.4f | total_drifts=%d",
                            result.event_id,
                            ",".join(drift_rec.triggered_models),
                            drift_rec.magnitude,
                            result.cumulative_drifts,
                        )

                if processed_count % 50 == 0:
                    m2_acc = result.model_snapshots["M2_Classical_Adaptive_HATC"].rolling_accuracy
                    m3_acc = result.model_snapshots["M3_Quantum_Static_VQC"].rolling_accuracy
                    m4_acc = result.model_snapshots["M4_Quantum_Adaptive_VQC"].rolling_accuracy
                    m6_acc = result.model_snapshots["M6_Quantum_Adaptive_QSVM"].rolling_accuracy
                    logger.info(
                        "Event %d | Rolling Acc -> M2(HATC): %.3f | M3(StatVQC): %.3f | M4(AdaptVQC): %.3f | M6(AdaptQSVM): %.3f",
                        result.event_id,
                        m2_acc,
                        m3_acc,
                        m4_acc,
                        m6_acc,
                    )
        finally:
            self.worker.close()
            self.sink.close()

        logger.info("Pipeline stopped cleanly after evaluating %d events.", processed_count)
        return processed_count


def main() -> None:
    """CLI entry point for the Quantum vs Classical Streaming Orchestrator."""
    parser = argparse.ArgumentParser(
        description="Quantum vs Classical Streaming Data Analytics Orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["kafka", "direct"],
        default="kafka",
        help="Execution mode: 'kafka' (consume from broker) or 'direct' (in-process stream).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of streaming events to evaluate.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.validate()
    configure_logging(cfg.log_level)

    orchestrator = StreamingPipelineOrchestrator(cfg)
    orchestrator.run(mode=args.mode, max_events=args.max_events)


if __name__ == "__main__":
    main()


    