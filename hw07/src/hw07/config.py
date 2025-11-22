from pathlib import Path
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DataSettings(BaseModel):
    """Settings for data generation."""

    num_samples_per_spiral: int = 200
    sigma_noise: float = 0.1


class ModelSettings(BaseModel):
    """Settings for model architecture."""

    num_inputs: int = 2
    num_outputs: int = 1
    num_hidden_layers: int = 4
    hidden_layer_width: int = 128


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 64
    num_iters: int = 20000
    learning_rate: float = 0.01
    l2_reg: float = 0.01


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 200
    output_dir: Path = Path("hw02/artifacts")


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    plotting: PlottingSettings = PlottingSettings()


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
