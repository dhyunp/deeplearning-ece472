from pathlib import Path
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DataSettings(BaseModel):
    """Settings for data generation."""

    dataset_name: str = "ag_news_subset"
    percent_train: int = 90


class ModelSettings(BaseModel):
    """Settings for model architecture."""

    input_depth: int = 384  # all-MiniLM-L6-v2 outputs size 384 vector
    layer_depths: int = 512
    num_hidden_layers: int = 2
    num_classes: int = 4  # World, Sports, Business, Sci/Tech


class TrainingSettings(BaseModel):
    """Settings for model training."""

    k_folds: int = 5
    batch_size: int = 1024
    num_iters: int = 5000
    learning_rate: float = 0.001
    l2_reg: float = 0.001


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 200
    output_dir: Path = Path("hw05/artifacts")


class LoggingSettings(BaseModel):
    """Settings for logging."""

    log_level: str = "INFO"
    log_format: str = "plain"  # "json" or "plain"
    log_output: str = "stdout"  # "stdout" or "file"
    output_dir: Path = Path("hw05/artifacts")


class SavingSettings(BaseModel):
    """Settings for model saving."""

    output_dir: Path = Path("hw05/saves/ag-news")


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
