import logging
import os
from pathlib import Path

import jax
import numpy as np
import structlog

from .config import LoggingSettings


class FormattedFloat(float):
    def __repr__(self) -> str:
        return f"{self:.3f}"


def custom_serializer_processor(logger, method_name, event_dict):
    def format_floats_recursive(obj):
        if isinstance(obj, float):
            return FormattedFloat(obj)
        if isinstance(obj, list):
            return [format_floats_recursive(item) for item in obj]
        if isinstance(obj, dict):
            return {k: format_floats_recursive(v) for k, v in obj.items()}
        return obj

    for key, value in event_dict.items():
        # Handle JAX arrays in addition to TF tensors
        if hasattr(value, "numpy"):  # Covers TF tensors
            value = value.numpy()
        if isinstance(value, jax.Array):
            value = np.array(value)
        if isinstance(value, (np.generic, np.ndarray)):
            value = value.item() if value.size == 1 else value.tolist()

        value = format_floats_recursive(value)

        if isinstance(value, Path):
            value = str(value)
        event_dict[key] = value
    return event_dict


def logfile_renderer(logger, method_name, event_dict):
    event = event_dict.get("event", "")
    timestamp = event_dict.get("timestamp", "")

    return f"{timestamp}: {event}"


def configure_logging():
    """Configure logging for the application."""
    log_settings = LoggingSettings()
    log_settings.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_settings.output_dir / "hw06.log"
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=log_path,
        filemode="w",
    )

    # Set the level for the application's logger
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.getLogger("hw06").setLevel(log_level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            custom_serializer_processor,
            logfile_renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
