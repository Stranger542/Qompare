# src/worker/feature_worker.py
"""Online MinMax feature scaler to [0, pi], sliding window buffer, and Kafka consumer worker."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.config import FEATURE_NAMES, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedStreamEvent:
    """Container for a validated, normalized streaming event ready for model inference."""

    event_id: int
    timestamp: str
    features_dict: Dict[str, float]
    features_array: np.ndarray  # Shape (5,), strictly bounded in [0, pi]
    label: int
    t_network_ms: float
    t_prep_ms: float


class OnlineAngleMinMaxScaler:
    """Online running Min-Max scaler that maps continuous features strictly into [0, pi].

    To prevent target or future-data leakage, `transform` scales using current bounds
    clipped to [0, pi], and `partial_fit_transform` updates running extrema using only
    the current observation x_t.
    """

    def __init__(self, feature_names: Tuple[str, ...] = FEATURE_NAMES) -> None:
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self._min: np.ndarray = np.full(self.n_features, np.inf, dtype=np.float64)
        self._max: np.ndarray = np.full(self.n_features, -np.inf, dtype=np.float64)
        self._n_seen: int = 0

    def _dict_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        vec = np.empty(self.n_features, dtype=np.float64)
        for idx, name in enumerate(self.feature_names):
            if name not in features or features[name] is None:
                raise ValueError(f"Missing feature '{name}' during scaling.")
            val = float(features[name])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite value '{val}' for feature '{name}'.")
            vec[idx] = val
        return vec

    def partial_fit(self, features: Dict[str, float]) -> None:
        """Update running feature minima and maxima with a single observation."""
        vec = self._dict_to_vector(features)
        self._min = np.minimum(self._min, vec)
        self._max = np.maximum(self._max, vec)
        self._n_seen += 1

    def transform(self, features: Dict[str, float]) -> Tuple[Dict[str, float], np.ndarray]:
        """Scale features into [0, pi] using running statistics."""
        vec = self._dict_to_vector(features)
        if self._n_seen == 0:
            scaled = np.full(self.n_features, math.pi / 2.0, dtype=np.float64)
        else:
            span = self._max - self._min
            # Where max == min (e.g., first sample or constant feature), map to midpoint pi/2
            safe_span = np.where(span > 1e-12, span, 1.0)
            normalized_01 = np.where(
                span > 1e-12, (vec - self._min) / safe_span, 0.5
            )
            scaled = np.clip(normalized_01 * math.pi, 0.0, math.pi)

        scaled_dict = {
            name: float(scaled[idx]) for idx, name in enumerate(self.feature_names)
        }
        return scaled_dict, scaled

    def partial_fit_transform(
        self, features: Dict[str, float]
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Update running bounds with current features x_t and return scaled [0, pi] values."""
        self.partial_fit(features)
        return self.transform(features)


class SlidingWindowBuffer:
    """Bounded FIFO sliding window storing recent (x_scaled, y) pairs for drift adaptation."""

    def __init__(self, maxlen: int = 100) -> None:
        if maxlen <= 0:
            raise ValueError("SlidingWindowBuffer maxlen must be positive.")
        self.maxlen = maxlen
        self._buffer: Deque[Tuple[np.ndarray, int]] = deque(maxlen=maxlen)

    def append(self, x_scaled: np.ndarray, label: int) -> None:
        """Add a normalized feature vector and integer binary label to the FIFO buffer."""
        if label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {label}.")
        self._buffer.append((np.asarray(x_scaled, dtype=np.float64).copy(), int(label)))

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) numpy arrays for all samples currently in the sliding window."""
        if not self._buffer:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), np.empty(
                (0,), dtype=np.int64
            )
        xs = np.vstack([item[0] for item in self._buffer])
        ys = np.array([item[1] for item in self._buffer], dtype=np.int64)
        return xs, ys

    def __len__(self) -> int:
        return len(self._buffer)


class StreamFeatureWorker:
    """Consumes raw events, deduplicates event IDs, normalizes features to [0, pi], and manages buffers."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.scaler = OnlineAngleMinMaxScaler(FEATURE_NAMES)
        self.sliding_buffer = SlidingWindowBuffer(maxlen=config.sliding_window_size)
        self.warmup_buffer = SlidingWindowBuffer(maxlen=config.initial_warmup_window)
        self._seen_event_ids: Set[int] = set()
        self._seen_order: Deque[int] = deque(maxlen=10000)
        self.consumer: Optional[KafkaConsumer] = None

    def process_raw_payload(self, payload: Dict[str, Any]) -> Optional[ProcessedStreamEvent]:
        """Validate, deduplicate, and scale a raw JSON payload into a ProcessedStreamEvent.

         Note: Does NOT append the label to the training buffers yet so prequential
        test-before-train order is strictly enforced. Call `record_post_inference`
        after predictions are generated.
        """
        t0_ns = time.perf_counter_ns()
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object dictionary.")

        if "event_id" not in payload or "features" not in payload or "label" not in payload:
            raise ValueError("Malformed event payload: missing required keys.")

        event_id = int(payload["event_id"])
        if event_id in self._seen_event_ids:
            logger.warning("Duplicate event_id=%d detected; skipping.", event_id)
            return None

        label = int(payload["label"])
        if label not in (0, 1):
            raise ValueError(f"Invalid binary label {label} in event {event_id}.")

        raw_features = payload["features"]
        if not isinstance(raw_features, dict):
            raise ValueError("Event 'features' field must be a dictionary.")

        scaled_dict, scaled_array = self.scaler.partial_fit_transform(raw_features)

        if len(self._seen_order) == self._seen_order.maxlen:
            oldest = self._seen_order.popleft()
            self._seen_event_ids.discard(oldest)
        self._seen_order.append(event_id)
        self._seen_event_ids.add(event_id)

        t_prep_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
        producer_ts_ns = payload.get("producer_ts_ns")
        if isinstance(producer_ts_ns, (int, float)) and producer_ts_ns > 0:
            t_network_ms = max(0.0, (time.time_ns() - int(producer_ts_ns)) / 1_000_000.0)
        else:
            t_network_ms = 0.0

        return ProcessedStreamEvent(
            event_id=event_id,
            timestamp=str(payload.get("timestamp", "")),
            features_dict=scaled_dict,
            features_array=scaled_array,
            label=label,
            t_network_ms=t_network_ms,
            t_prep_ms=t_prep_ms,
        )

    def record_post_inference(self, event: ProcessedStreamEvent) -> None:
        """Append the labeled observation to sliding/warmup buffers AFTER inference completes."""
        self.sliding_buffer.append(event.features_array, event.label)
        if len(self.warmup_buffer) < self.config.initial_warmup_window:
            self.warmup_buffer.append(event.features_array, event.label)

    def connect_kafka(self, max_retries: int = 5, retry_backoff_sec: float = 2.0) -> None:
        """Connect KafkaConsumer to the raw stream topic with bounded retries."""
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                self.consumer = KafkaConsumer(
                    self.config.kafka_topic_raw,
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    group_id=self.config.kafka_consumer_group,
                    auto_offset_reset="earliest",
                    enable_auto_commit=True,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    consumer_timeout_ms=self.config.kafka_consumer_timeout_ms,
                )
                logger.info(
                    "Connected KafkaConsumer to topic '%s' on attempt %d",
                    self.config.kafka_topic_raw,
                    attempt,
                )
                return
            except KafkaError as err:
                last_err = err
                logger.warning(
                    "KafkaConsumer connection attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    err,
                )
                time.sleep(retry_backoff_sec * attempt)
        raise RuntimeError(
            f"Unable to connect KafkaConsumer to {self.config.kafka_bootstrap_servers}"
        ) from last_err

    def iter_kafka_events(self, max_events: Optional[int] = None) -> Iterator[ProcessedStreamEvent]:
        """Yield validated and normalized events from Kafka."""
        if self.consumer is None:
            self.connect_kafka()
        assert self.consumer is not None

        yielded = 0
        for msg in self.consumer:
            try:
                processed = self.process_raw_payload(msg.value)
                if processed is not None:
                    yield processed
                    yielded += 1
                    if max_events is not None and yielded >= max_events:
                        break
            except Exception as err:
                logger.error("Skipping malformed Kafka message: %s", err)

    def close(self) -> None:
        """Close the Kafka consumer cleanly."""
        if self.consumer is not None:
            try:
                self.consumer.close()
            except KafkaError as err:
                logger.error("Error closing KafkaConsumer: %s", err)
            finally:
                self.consumer = None


