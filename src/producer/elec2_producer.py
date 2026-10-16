# src/producer/elec2_producer.py
"""Kafka Producer service that replays the ELEC2 dataset as a real-time event stream."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

from kafka import KafkaProducer
from kafka.errors import KafkaError
from river import datasets

from src.config import FEATURE_NAMES, PipelineConfig, configure_logging

logger = logging.getLogger(__name__)


def serialize_elec2_event(
    event_id: int,
    raw_features: Dict[str, Any],
    label: Any,
    base_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Format a single ELEC2 record into the standardized streaming JSON schema.

    Args:
        event_id: Monotonically increasing 0-based or 1-based event identifier.
        raw_features: Dictionary containing raw record attributes from ELEC2.
        label: Binary target (True/False or 1/0) indicating price movement.
        base_time: Base UTC timestamp (defaults to 1997-01-01T00:00:00Z).

    Returns:
        Dictionary ready for JSON serialization and Kafka transmission.

    Raises:
        ValueError: If required features are missing or non-numeric, or label is invalid.
    """
    if base_time is None:
        base_time = datetime(1997, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    extracted_features: Dict[str, float] = {}
    for fname in FEATURE_NAMES:
        if fname not in raw_features or raw_features[fname] is None:
            raise ValueError(f"Missing required feature '{fname}' in event {event_id}.")
        try:
            val = float(raw_features[fname])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid non-numeric value for feature '{fname}' in event {event_id}."
            ) from exc
        extracted_features[fname] = val

    if isinstance(label, bool):
        int_label = 1 if label else 0
    elif label in (0, 1, "0", "1", "True", "False", "UP", "DOWN"):
        int_label = 1 if str(label).upper() in ("1", "TRUE", "UP") else 0
    else:
        raise ValueError(f"Unexpected label value '{label}' in event {event_id}.")

    event_timestamp = (base_time + timedelta(minutes=30 * event_id)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "event_id": int(event_id),
        "timestamp": event_timestamp,
        "producer_ts_ns": time.time_ns(),
        "features": extracted_features,
        "label": int_label,
    }


def iter_elec2_stream(max_events: Optional[int] = None) -> Iterator[Tuple[int, Dict[str, float], int]]:
    """Iterate chronologically over the ELEC2 dataset from river.datasets.Elec2.

    Args:
        max_events: Optional cap on number of events yielded.

    Yields:
        Tuple of (event_id, 5-feature dict, integer binary label {0, 1}).
    """
    dataset = datasets.Elec2()
    for idx, (x, y) in enumerate(dataset):
        if max_events is not None and idx >= max_events:
            break
        payload = serialize_elec2_event(idx, x, y)
        yield idx, payload["features"], payload["label"]


class Elec2KafkaProducer:
    """Streams chronological ELEC2 events into Apache Kafka with configurable pacing."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.producer: Optional[KafkaProducer] = None

    def connect(self, max_retries: int = 5, retry_backoff_sec: float = 2.0) -> None:
        """Connect to the Kafka broker with bounded exponential backoff."""
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: str(k).encode("utf-8"),
                    acks="all",
                    retries=3,
                )
                logger.info(
                    "Connected KafkaProducer to %s on attempt %d",
                    self.config.kafka_bootstrap_servers,
                    attempt,
                )
                return
            except KafkaError as err:
                last_err = err
                logger.warning(
                    "Kafka connection attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    err,
                )
                time.sleep(retry_backoff_sec * attempt)
        raise RuntimeError(
            f"Unable to connect to Kafka broker at {self.config.kafka_bootstrap_servers}"
        ) from last_err

    def stream_dataset(self, max_events: Optional[int] = None) -> int:
        """Publish ELEC2 events to the configured Kafka topic.

        Args:
            max_events: Optional maximum number of records to publish.

        Returns:
            Total number of events published.
        """
        if self.producer is None:
            self.connect()
        assert self.producer is not None

        published_count = 0
        delay_sec = max(0.0, self.config.producer_delay_ms / 1000.0)
        dataset = datasets.Elec2()

        try:
            for event_id, (raw_x, raw_y) in enumerate(dataset):
                if max_events is not None and event_id >= max_events:
                    break
                event_payload = serialize_elec2_event(event_id, raw_x, raw_y)
                self.producer.send(
                    self.config.kafka_topic_raw,
                    key=event_id,
                    value=event_payload,
                )
                published_count += 1
                if published_count % 250 == 0:
                    self.producer.flush()
                    logger.info(
                        "Published %d events to topic '%s'",
                        published_count,
                        self.config.kafka_topic_raw,
                    )
                if delay_sec > 0.0:
                    time.sleep(delay_sec)
        finally:
            self.close()

        logger.info("Completed streaming %d ELEC2 events.", published_count)
        return published_count

    def close(self) -> None:
        """Flush pending messages and close the Kafka producer cleanly."""
        if self.producer is not None:
            try:
                self.producer.flush(timeout=10.0)
                self.producer.close(timeout=10.0)
            except KafkaError as err:
                logger.error("Error while closing KafkaProducer: %s", err)
            finally:
                self.producer = None


def main() -> None:
    """CLI entry point for running the ELEC2 Kafka producer service."""
    parser = argparse.ArgumentParser(description="ELEC2 Kafka Stream Producer")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of ELEC2 events to publish (default: full 45,312 stream).",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=None,
        help="Override inter-event pacing delay in milliseconds.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.delay_ms is not None:
        object.__setattr__(cfg, "producer_delay_ms", args.delay_ms)
    cfg.validate()
    configure_logging(cfg.log_level)

    producer = Elec2KafkaProducer(cfg)
    producer.stream_dataset(max_events=args.max_events)


if __name__ == "__main__":
    main()


    