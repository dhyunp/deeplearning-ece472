from pathlib import Path
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DataSettings(BaseModel):
    """Settings for data generation."""

    percent_train: int = 90


class ModelSettings(BaseModel):
    """Settings for model architecture."""

    input_height: int = 28
    input_width: int = 28
    input_depth: int = 1
    layer_depths: list[int] = [32, 64, 128]
    layer_kernel_sizes: list[tuple[int, int]] = [(5, 5), (3, 3), (3, 3)]
    num_classes: int = 10


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 64
    epochs: int = 500
    learning_rate: float = 0.01
    l2_reg: float = 0.01


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 200
    output_dir: Path = Path("hw03/artifacts")


class LoggingSettings(BaseModel):
    """Settings for logging."""

    log_level: str = "INFO"
    log_format: str = "plain"  # "json" or "plain"
    log_output: str = "stdout"  # "stdout" or "file"
    output_dir: Path = Path("hw03/artifacts")


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    plotting: PlottingSettings = PlottingSettings()
    logging: LoggingSettings = LoggingSettings()


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
