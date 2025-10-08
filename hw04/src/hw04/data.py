from dataclasses import InitVar, dataclass, field

import numpy as np
import structlog
import tensorflow_datasets as tfds
import tensorflow as tf

log = structlog.get_logger()


@dataclass
class Data:
    """Handles generation of synthetic data for CNN."""

    rng: InitVar[np.random.Generator]
    dataset_name: str
    percent_train: int
    train_image_set: np.ndarray = field(init=False)
    train_label_set: np.ndarray = field(init=False)
    val_image_set: np.ndarray = field(init=False)
    val_label_set: np.ndarray = field(init=False)
    test_image_set: np.ndarray = field(init=False)
    test_label_set: np.ndarray = field(init=False)
    train_index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator) -> None:
        """Generate training and validating data."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        # load data from TF
        (cifar_train, cifar_val) = tfds.load(
            self.dataset_name,
            split=[
                "train[:{percent}%]".format(percent=self.percent_train),
                "train[{percent}%:]".format(percent=self.percent_train),
            ],
            shuffle_files=True,
            as_supervised=True,
        )

        cifar_train = cifar_train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        cifar_train = tfds.as_numpy(cifar_train)
        cifar_val = cifar_val.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        cifar_val = tfds.as_numpy(cifar_val)

        cifar_test = tfds.load(
            self.dataset_name, split="test", shuffle_files=True, as_supervised=True
        )
        cifar_test = tfds.as_numpy(cifar_test)

        self.train_image_set = np.stack([x for x, _ in cifar_train])
        self.train_label_set = np.stack([y for _, y in cifar_train])
        self.val_image_set = np.stack([x for x, _ in cifar_val])
        self.val_label_set = np.stack([y for _, y in cifar_val])
        self.test_image_set = np.stack([x for x, _ in cifar_test])
        self.test_label_set = np.stack([y for _, y in cifar_test])
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

        images = self.train_image_set[choices]
        labels = self.train_label_set[choices].flatten()

        images = self.augment(rng, images)

        return images, labels

    def augment(self, rng: np.random.Generator, images: np.ndarray):
        batch_size, height, width, _ = images.shape
        augmented_images = []

        for image in images:
            if rng.random() > 0.5:
                image = np.fliplr(image)

            padding = 4
            padded_image = np.pad(
                image, ((padding, padding), (padding, padding), (0, 0)), mode="reflect"
            )

            h_start = rng.integers(0, padding * 2)
            w_start = rng.integers(0, padding * 2)

            image = padded_image[
                h_start : h_start + height, w_start : w_start + width, :
            ]
            augmented_images.append(image)

        return np.stack(augmented_images)
