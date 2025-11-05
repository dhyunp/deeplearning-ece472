from pathlib import Path
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DataSettings(BaseModel):
    """Settings for data generation."""


class ModelSettings(BaseModel):
    """Settings for model architecture."""

    num_heads: int = 8
    num_layers: int = 6
    embedding_dim: int = 384
    context_length: int = 32
    vocab_size: int = 10000


class TrainingSettings(BaseModel):
    """Settings for model training."""

    k_folds: int = 5
    batch_size: int = 128
    num_iters: int = 10000
    learning_rate: float = 0.001
    l2_reg: float = 0.0001


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 200
    output_dir: Path = Path("hw06/artifacts")


class LoggingSettings(BaseModel):
    """Settings for logging."""

    log_level: str = "INFO"
    log_format: str = "plain"  # "json" or "plain"
    log_output: str = "stdout"  # "stdout" or "file"
    output_dir: Path = Path("hw06/artifacts")


class SavingSettings(BaseModel):
    """Settings for model saving."""

    output_dir: Path = Path("hw06/saves/transformer")


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    plotting: PlottingSettings = PlottingSettings()
    logging: LoggingSettings = LoggingSettings()
    saving: SavingSettings = SavingSettings()


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
