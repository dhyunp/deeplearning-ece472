from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Data:
    """Handles generation of synthetic data for MLP."""

    rng: InitVar[np.random.Generator]
    num_samples_per_spiral: int
    sigma: float
    x0: np.ndarray = field(init=False)
    y0: np.ndarray = field(init=False)
    x1: np.ndarray = field(init=False)
    y1: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)
    data: np.ndarray = field(init=False)
    spiral_coordinates: np.ndarray = field(init=False)
    spiral_labels: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate two spiral data."""
        self.index = np.arange(self.num_samples_per_spiral * 2)

        spiral = np.linspace(
            0, 4 * np.pi, self.num_samples_per_spiral
        )  # 2 2pi rotations

        # spiral * np.foo(spiral) makes the spiral radially grow outwards
        self.x0 = spiral * np.cos(spiral) + rng.normal(
            0, self.sigma, size=(self.num_samples_per_spiral)
        )
        self.y0 = spiral * np.sin(spiral) + rng.normal(
            0, self.sigma, size=(self.num_samples_per_spiral)
        )
        self.labels0 = np.zeros((self.num_samples_per_spiral, 1))
        self.x1 = -spiral * np.cos(spiral) + rng.normal(
            0, self.sigma, size=(self.num_samples_per_spiral)
        )
        self.y1 = -spiral * np.sin(spiral) + rng.normal(
            0, self.sigma, size=(self.num_samples_per_spiral)
        )
        self.labels1 = np.ones((self.num_samples_per_spiral, 1))

        spirals = np.vstack(
            [np.stack([self.x0, self.y0], axis=1), np.stack([self.x1, self.y1], axis=1)]
        )
        labels = np.vstack([self.labels0, self.labels1])
        self.data = np.hstack([spirals, labels])

        self.spiral_coordinates = self.data[:, :2]
        self.spiral_labels = self.data[:, 2]

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.spiral_coordinates[choices], self.spiral_labels[choices]
