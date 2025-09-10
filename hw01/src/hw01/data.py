from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Data:
    """Handles generation of synthetic data for Gaussian regression."""

    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data based on y = sin(2*pi*x)."""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0.1, 0.9, size=(self.num_samples, 1))
        clean_y = np.sin(self.x * 2 * np.pi)
        noise = rng.normal(0, self.sigma, size=clean_y.shape)
        noisy_y = clean_y + noise
        self.y = noisy_y

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
