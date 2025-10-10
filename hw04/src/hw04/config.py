from pathlib import Path
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DataSettings(BaseModel):
    """Settings for data generation."""

    dataset_name: str = "cifar10"
    percent_train: int = 90


class ModelSettings(BaseModel):
    """Settings for model architecture."""

    input_height: int = 32
    input_width: int = 32
    input_depth: int = 3
    layer_depths: list[int] = [
        32,
        32,
        32,
        64,
        64,
        64,
        128,
        128,
        128,
    ]  # resnet 18 layers
    strides: list[int] = [1, 1, 1, 2, 1, 1, 2, 1, 1]
    num_groups: list[int] = [1, 1, 1, 2, 2, 2, 4, 4, 4]
    layer_kernel_sizes: list[tuple[int, int]] = [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]
    num_classes: int = 10  # 10 for cifar10, 100 for cifar100


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 128
    epochs: int = 12500
    learning_rate: float = 0.001
    l2_reg: float = 0.0001
    momentum: float = 0.9


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 200
    output_dir: Path = Path("hw04/artifacts")


class LoggingSettings(BaseModel):
    """Settings for logging."""

    log_level: str = "INFO"
    log_format: str = "plain"  # "json" or "plain"
    log_output: str = "stdout"  # "stdout" or "file"
    output_dir: Path = Path("hw04/artifacts")


class SavingSettings(BaseModel):
    """Settings for model saving."""

    output_dir_10: Path = Path("hw04/saves/cifar-10")
    output_dir_100: Path = Path("hw04/saves/cifar-100")


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
