"""InfluxDB v2 telemetry writer for prequential accuracy, latency decomposition, and drift alerts."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from src.config import PipelineConfig
from src.evaluation.prequential_eval import EventEvaluationResult

logger = logging.getLogger(__name__)


class InfluxTelemetrySink:
    """Writes streaming evaluation results to InfluxDB v2 bucket `streaming_quantum`."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.enabled = config.influxdb_enabled
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None

    def connect(self, max_retries: int = 3, backoff_sec: float = 1.5) -> bool:
        """Connect to InfluxDB v2 with bounded retries. Disables sink gracefully if unreachable."""
        if not self.enabled:
            logger.info("InfluxTelemetrySink is disabled via configuration.")
            return False

        for attempt in range(1, max_retries + 1):
            try:
                self.client = InfluxDBClient(
                    url=self.config.influxdb_url,
                    token=self.config.influxdb_token,
                    org=self.config.influxdb_org,
                    timeout=5000,
                )
                if self.client.ping():
                    self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                    logger.info(
                        "Connected InfluxTelemetrySink to %s (bucket=%s)",
                        self.config.influxdb_url,
                        self.config.influxdb_bucket,
                    )
                    return True
            except Exception as err:
                logger.warning(
                    "InfluxDB connection attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    err,
                )
                time.sleep(backoff_sec * attempt)

        logger.error(
            "InfluxDB unreachable at %s; continuing pipeline execution without remote sink.",
            self.config.influxdb_url,
        )
        self.enabled = False
        return False

    def build_points(self, result: EventEvaluationResult) -> List[Point]:
        """Convert an EventEvaluationResult into InfluxDB Point objects across all 3 measurements."""
        now_ns = time.time_ns()
        points: List[Point] = []

        for model_name, snap in result.model_snapshots.items():
            # 1. prequential_accuracy measurement
            p_acc = (
                Point("prequential_accuracy")
                .tag("model", model_name)
                .field("event_id", int(result.event_id))
                .field("rolling_accuracy", float(snap.rolling_accuracy))
                .field("cumulative_accuracy", float(snap.cumulative_accuracy))
                .field("rolling_precision", float(snap.rolling_precision))
                .field("rolling_recall", float(snap.rolling_recall))
                .field("rolling_f1", float(snap.rolling_f1))
                .field("y_true", int(snap.y_true))
                .field("y_pred", int(snap.y_pred))
                .time(now_ns, WritePrecision.NS)
            )
            points.append(p_acc)

            # 2. latency_telemetry measurement
            p_lat = (
                Point("latency_telemetry")
                .tag("model", model_name)
                .field("event_id", int(result.event_id))
                .field("t_network_ms", float(result.t_network_ms))
                .field("t_prep_ms", float(result.t_prep_ms))
                .field("t_encode_ms", float(snap.t_encode_ms))
                .field("t_sim_ms", float(snap.t_sim_ms))
                .field("t_update_ms", float(snap.t_update_ms))
                .field("t_total_ms", float(snap.t_total_ms))
                .time(now_ns, WritePrecision.NS)
            )
            points.append(p_lat)

        # 3. drift_events measurement
        drift_flag = 1 if result.drift_event is not None else 0
        drift_mag = (
            float(result.drift_event.magnitude)
            if result.drift_event is not None
            else 0.0
        )
        triggered_str = (
            ",".join(result.drift_event.triggered_models)
            if result.drift_event is not None
            else "none"
        )
        p_drift = (
            Point("drift_events")
            .tag("detector", "ADWIN")
            .field("event_id", int(result.event_id))
            .field("event_timestamp", str(result.timestamp))
            .field("drift_detected", int(drift_flag))
            .field("drift_magnitude", float(drift_mag))
            .field("triggered_models", triggered_str)
            .field("cumulative_drifts", int(result.cumulative_drifts))
            .field("m2_m4_agreement", float(result.m2_m4_agreement))
            .time(now_ns, WritePrecision.NS)
        )
        points.append(p_drift)
        return points

    def write_result(self, result: EventEvaluationResult) -> bool:
        """Write all telemetry points for a single event to InfluxDB."""
        if not self.enabled or self.write_api is None:
            return False
        points = self.build_points(result)
        try:
            self.write_api.write(
                bucket=self.config.influxdb_bucket,
                org=self.config.influxdb_org,
                record=points,
            )
            return True
        except Exception as err:
            logger.error(
                "Failed writing telemetry for event %d to InfluxDB: %s",
                result.event_id,
                err,
            )
            return False

    def close(self) -> None:
        """Flush and close the InfluxDB client cleanly."""
        if self.write_api is not None:
            try:
                self.write_api.close()
            except Exception as err:
                logger.error("Error closing InfluxDB write_api: %s", err)
            finally:
                self.write_api = None
        if self.client is not None:
            try:
                self.client.close()
            except Exception as err:
                logger.error("Error closing InfluxDBClient: %s", err)
            finally:
                self.client = None
