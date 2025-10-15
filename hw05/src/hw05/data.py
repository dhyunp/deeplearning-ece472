from dataclasses import InitVar, dataclass, field

import numpy as np
import structlog
import tensorflow_datasets as tfds

from .embedding import create_embeddings

log = structlog.get_logger()


@dataclass
class Data:
    """Handles generation of synthetic data for CNN."""

    rng: InitVar[np.random.Generator]
    dataset_name: str
    train_text_set: np.ndarray = field(init=False)
    train_label_set: np.ndarray = field(init=False)
    val_text_set: np.ndarray = field(init=False)
    val_label_set: np.ndarray = field(init=False)
    test_text_set: np.ndarray = field(init=False)
    test_label_set: np.ndarray = field(init=False)
    train_index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator) -> None:
        """Generate training and validating data."""

        # load data from TF
        (ag_train, ag_test) = tfds.load(
            self.dataset_name,
            split=["train", "test"],
            shuffle_files=True,
        )

        def combine_text(x):
            text = x["title"] + " " + x["description"]
            return text, x["label"]

        ag_train = ag_train.map(combine_text)
        ag_test = ag_test.map(combine_text)

        train_text_set = np.stack([x for x, _ in ag_train])
        self.train_label_set = np.stack([y for _, y in ag_train])
        test_text_set = np.stack([x for x, _ in ag_test])
        self.test_label_set = np.stack([y for _, y in ag_test])

        log.debug(
            "Data initialized",
            train_text=train_text_set.shape,
            train_labels=self.train_label_set.shape,
            test_text=test_text_set.shape,
            test_labels=self.test_label_set.shape,
        )

        log.debug(
            "Sample data",
            train_text=train_text_set[0],
            train_labels=self.train_label_set[0],
            test_text=test_text_set[0],
            test_labels=self.test_label_set[0],
        )

        log.info("Creating word embeddings of the text")

        self.train_text_set = create_embeddings(train_text_set)
        self.test_text_set = create_embeddings(test_text_set)
        self.train_index = np.arange(self.train_text_set.shape[0])

        log.info("Word embeddings of the text created")

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.train_index, size=batch_size)

        texts = self.train_text_set[choices]
        labels = self.train_label_set[choices].flatten()

        return texts, labels
