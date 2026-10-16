"""Central configuration module for the Quantum vs Classical Streaming pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()

FEATURE_NAMES: Tuple[str, ...] = (
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Validated configuration container for all pipeline services and models."""

    # Kafka settings
    kafka_bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    kafka_topic_raw: str = field(
        default_factory=lambda: os.getenv("KAFKA_TOPIC_RAW", "elec2-stream")
    )
    kafka_consumer_group: str = field(
        default_factory=lambda: os.getenv(
            "KAFKA_CONSUMER_GROUP", "qml-streaming-worker-group"
        )
    )
    kafka_consumer_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("KAFKA_CONSUMER_TIMEOUT_MS", "60000"))
    )

    # Stream & Window settings
    producer_delay_ms: int = field(
        default_factory=lambda: int(os.getenv("PRODUCER_DELAY_MS", "25"))
    )
    initial_warmup_window: int = field(
        default_factory=lambda: int(os.getenv("INITIAL_WARMUP_WINDOW", "200"))
    )
    sliding_window_size: int = field(
        default_factory=lambda: int(os.getenv("SLIDING_WINDOW_SIZE", "100"))
    )
    rolling_metric_window: int = field(
        default_factory=lambda: int(os.getenv("ROLLING_METRIC_WINDOW", "100"))
    )
    random_seed: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )

    # Quantum Circuit & Optimizer settings
    num_qubits: int = field(
        default_factory=lambda: int(os.getenv("NUM_QUBITS", "5"))
    )
    vqc_ansatz_reps: int = field(
        default_factory=lambda: int(os.getenv("VQC_ANSATZ_REPS", "2"))
    )
    vqc_initial_spsa_steps: int = field(
        default_factory=lambda: int(os.getenv("VQC_INITIAL_SPSA_STEPS", "40"))
    )
    vqc_adapt_spsa_steps: int = field(
        default_factory=lambda: int(os.getenv("VQC_ADAPT_SPSA_STEPS", "10"))
    )
    vqc_spsa_lr: float = field(
        default_factory=lambda: float(os.getenv("VQC_SPSA_LR", "0.15"))
    )
    vqc_spsa_perturbation: float = field(
        default_factory=lambda: float(os.getenv("VQC_SPSA_PERTURBATION", "0.10"))
    )

    # Quantum Kernel settings
    qsvm_max_support_size: int = field(
        default_factory=lambda: int(os.getenv("QSVM_MAX_SUPPORT_SIZE", "80"))
    )
    qsvm_replace_ratio: float = field(
        default_factory=lambda: float(os.getenv("QSVM_REPLACE_RATIO", "0.20"))
    )
    qsvm_c: float = field(
        default_factory=lambda: float(os.getenv("QSVM_C", "1.0"))
    )

    # Drift Detection settings
    adwin_delta: float = field(
        default_factory=lambda: float(os.getenv("ADWIN_DELTA", "0.002"))
    )

    # InfluxDB v2 settings
    influxdb_url: str = field(
        default_factory=lambda: os.getenv("INFLUXDB_URL", "http://localhost:8086")
    )
    influxdb_token: str = field(
        default_factory=lambda: os.getenv(
            "INFLUXDB_TOKEN", "ioe413-super-secret-auth-token"
        )
    )
    influxdb_org: str = field(
        default_factory=lambda: os.getenv("INFLUXDB_ORG", "ioe413-org")
    )
    influxdb_bucket: str = field(
        default_factory=lambda: os.getenv("INFLUXDB_BUCKET", "streaming_quantum")
    )
    influxdb_enabled: bool = field(
        default_factory=lambda: os.getenv("INFLUXDB_ENABLED", "true").lower()
        in ("1", "true", "yes")
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    def validate(self) -> None:
        """Validate configuration invariants and raise ValueError on invalid settings."""
        if self.num_qubits != len(FEATURE_NAMES):
            raise ValueError(
                f"num_qubits ({self.num_qubits}) must equal number of ELEC2 features ({len(FEATURE_NAMES)})."
            )
        if self.initial_warmup_window < 10:
            raise ValueError("initial_warmup_window must be at least 10 records.")
        if self.sliding_window_size < 10:
            raise ValueError("sliding_window_size must be at least 10 records.")
        if self.rolling_metric_window < 1:
            raise ValueError("rolling_metric_window must be >= 1.")
        if not (0.0 < self.qsvm_replace_ratio <= 1.0):
            raise ValueError("qsvm_replace_ratio must be in (0.0, 1.0].")
        if self.vqc_ansatz_reps < 1:
            raise ValueError("vqc_ansatz_reps must be >= 1.")
        if self.adwin_delta <= 0.0 or self.adwin_delta >= 1.0:
            raise ValueError("adwin_delta must be strictly between 0 and 1.")


def configure_logging(level_name: str = "INFO") -> None:
    """Configure structured root logging for all pipeline components."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


