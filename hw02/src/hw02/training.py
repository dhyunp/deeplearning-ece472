import jax.numpy as jnp
import numpy as np
import structlog
import optax
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import NNXMLPModel

log = structlog.get_logger()


@nnx.jit
def train_step(
    model: NNXMLPModel, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
):
    """Performs a single training step."""

    def loss_fn(model: NNXMLPModel):
        y_hat = model(x)
        return optax.sigmoid_binary_cross_entropy(y_hat, y.reshape(y_hat.shape)).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place update of model parameters
    return loss


def train(
    model: NNXMLPModel,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        coords_np, labels_np = data.get_batch(np_rng, settings.batch_size)
        coords, labels = jnp.asarray(coords_np), jnp.asarray(labels_np)
        # log.debug("Y value", y=y)

        loss = train_step(model, optimizer, coords, labels)
        # log.debug("Training step", step=i, loss=loss)
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training finished")
