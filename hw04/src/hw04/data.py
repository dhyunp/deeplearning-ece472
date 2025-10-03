from dataclasses import InitVar, dataclass, field

import numpy as np
import structlog


log = structlog.get_logger()


@dataclass
class Data:
    """Handles generation of synthetic data for CNN."""

    rng: InitVar[np.random.Generator]
    train_set: np.ndarray
    val_set: np.ndarray
    test_set: np.ndarray
    train_image_set: np.ndarray = field(init=False)
    train_label_set: np.ndarray = field(init=False)
    val_image_set: np.ndarray = field(init=False)
    val_label_set: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator) -> None:
        """Generate training and validating data."""

        self.train_image_set = np.stack([x for x, _ in self.train_set])
        self.train_label_set = np.stack([y for _, y in self.train_set])
        self.val_image_set = np.stack([x for x, _ in self.val_set])
        self.val_label_set = np.stack([y for _, y in self.val_set])
        self.test_image_set = np.stack([x for x, _ in self.test_set])
        self.test_label_set = np.stack([y for _, y in self.test_set])
        self.train_index = np.arange(self.train_image_set.shape[0])

        log.debug(
            "Data initialized",
            train_images=self.train_image_set.shape,
            train_labels=self.train_label_set.shape,
            val_images=self.val_image_set.shape,
            val_labels=self.val_label_set.shape,
        )

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.train_index, size=batch_size)

        return self.train_image_set[choices], self.train_label_set[choices].flatten()
