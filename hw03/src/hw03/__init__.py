import structlog
import jax
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax import nnx

from .logging import configure_logging
from .config import load_settings
from .data import Data
from .model import Classifier
from .training_testing import train, test


def main() -> None:
    """CLI entry point."""
    configure_logging()
    log = structlog.get_logger()
    log.info("Hello from hw03!")

    settings = load_settings()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    mnist_train_val = tfds.load(
        "mnist",
        split=[
            "train[:{percent}%]".format(percent=settings.data.percent_train),
            "train[{percent}%:]".format(percent=settings.data.percent_train),
        ],
        shuffle_files=True,
        as_supervised=True,
    )
    mnist_train, mnist_val = tfds.as_numpy(mnist_train_val)
    mnist_test = tfds.load(
        "mnist", split="test", shuffle_files=True, as_supervised=True
    )
    mnist_test = tfds.as_numpy(mnist_test)

    data = Data(
        rng=np_rng,
        train_set=mnist_train,
        val_set=mnist_val,
        test_set=mnist_test,
    )

    model = Classifier(
        key=model_key,
        input_height=settings.model.input_height,
        input_width=settings.model.input_width,
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_sizes=settings.model.layer_kernel_sizes,
        num_classes=settings.model.num_classes,
    )

    learning_rate_schedule = optax.exponential_decay(
        init_value=settings.training.learning_rate,
        transition_steps=5000,
        decay_rate=0.1,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=settings.training.l2_reg,
        ),
        wrt=nnx.Param,
    )

    val_accuracy = train(model, optimizer, data, settings.training, np_rng)
    log.debug("Finished Training model")

    if val_accuracy > 0.955:
        log.info("Achieved target accuracy on validation set", accuracy=val_accuracy)
        test(model, data)
    else:
        log.info("Did not achieve target accuracy on validation set", accuracy=val_accuracy)
