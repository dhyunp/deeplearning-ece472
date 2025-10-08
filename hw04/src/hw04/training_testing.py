import jax
import jax.numpy as jnp
import numpy as np
import structlog
import optax
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import Classifier

log = structlog.get_logger()


def calc_values(x, y):
    loss = optax.softmax_cross_entropy_with_integer_labels(x, y).mean()
    accuracy = jnp.mean(jnp.argmax(x, axis=-1) == y)
    _, top_5_indices = jax.lax.top_k(x, k=5)
    y = y.reshape(-1, 1)
    top5_accuracy = jnp.mean(jnp.any(top_5_indices == y, axis=-1))
    return [loss, accuracy, top5_accuracy]


@nnx.jit
def train_step(
    model: Classifier, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
):
    """Performs a single training step."""

    def loss_fn(model: Classifier):
        y_hat = model(x)
        return calc_values(y_hat, y)[0]

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place update of model parameters
    accuracy = calc_values(model(x), y)[1]
    return loss, accuracy


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> list[float]:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.epochs)
    for i in bar:
        train_image_batch, train_label_batch = data.get_batch(
            np_rng, settings.batch_size
        )

        loss, accuracy = train_step(
            model, optimizer, train_image_batch, train_label_batch
        )
        # log.debug("Training step", step=i, loss=loss)
        bar.set_description(f"Loss @ {i} => {loss:.6f}, Acc @ {accuracy:.6f}")
        bar.refresh()

        if i % 2500 == 0:
            log.info("Training step", step=i, loss=loss, accuracy=accuracy)

    log.info("Training step", step=settings.epochs, loss=loss, accuracy=accuracy)
    log.info("Training finished")

    # test on validation set
    results = calc_values(model(data.test_image_set), data.test_label_set)
    top1_accuracy = results[1]
    top5_accuracy = results[2]
    log.info("Top 1 accuracy", accuracy=top1_accuracy)
    log.info("Top 5 accuracy", accuracy=top5_accuracy)

    return [top1_accuracy, top5_accuracy]


def test(
    model: Classifier,
    data: Data,
) -> None:
    """Test the model using test dataset."""
    results = calc_values(model(data.test_image_set), data.test_label_set)
    top1_accuracy = results[1]
    top5_accuracy = results[2]
    log.info("Top 1 accuracy", accuracy=top1_accuracy)
    log.info("Top 5 accuracy", accuracy=top5_accuracy)
