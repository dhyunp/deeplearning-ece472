import jax.numpy as jnp
import numpy as np
import structlog
import optax
import matplotlib.pyplot as plt
from pathlib import Path
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import MLP

log = structlog.get_logger()


def calc_values(x, y):
    loss = optax.softmax_cross_entropy_with_integer_labels(x, y).mean()
    accuracy = jnp.mean(jnp.argmax(x, axis=-1) == y)
    return loss, accuracy


@nnx.jit
def train_step(model: MLP, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray):
    """Performs a single training step."""

    def loss_fn(model: MLP):
        y_hat = model(x)
        loss, accuracy = calc_values(y_hat, y)
        # loss = optax.softmax_cross_entropy_with_integer_labels(y_hat, y).mean()

        return loss, accuracy

    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)  # In-place update of model parameters
    return loss, accuracy


def train(
    model: MLP,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)

    losses = []
    accuracies = []
    for i in bar:
        train_text_batch, train_label_batch = data.get_batch(
            np_rng, settings.batch_size
        )

        loss, accuracy = train_step(
            model, optimizer, train_text_batch, train_label_batch
        )
        losses.append(loss)
        accuracies.append(accuracy)

        bar.set_description(f"Loss @ {i} => {loss:.6f}, Acc @ {accuracy:.6f}")
        bar.refresh()
        if i % 1 == 0:
            # _, accuracy = calc_values(model(train_text_batch), train_label_batch)
            log.info("Training step", step=i, loss=loss, accuracy=accuracy)

    log.info("Training step", step=settings.num_iters, loss=loss, accuracy=accuracy)
    log.info("Training finished")

    plot(losses, accuracies)

    # test on validation set
    results = calc_values(model(data.val_text_set), data.val_label_set)
    top1_accuracy = results[1]
    log.info("Validation Set Top 1 accuracy", accuracy=top1_accuracy)


def test(
    model: MLP,
    data: Data,
) -> None:
    """Test the model using test dataset."""
    results = calc_values(model(data.test_text_set), data.test_label_set)
    top1_accuracy = results[1]
    log.info("Test Set Top 1 accuracy", accuracy=top1_accuracy)


def plot(losses, accuracies):
    steps = np.arange(len(losses))
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 10), dpi=200)

    # Plot Loss on the primary y-axis (left)
    color = "tab:red"
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(steps, losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a secondary y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(steps, accuracies, color=color, label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)

    # Add title and layout adjustments
    plt.title("Training Loss and Accuracy over Iterations")
    fig.tight_layout()

    # Ensure the output directory exists and save the figure
    path = Path("hw05/artifacts")
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / "training_loss.pdf"
    plt.savefig(output_path)
    log.info("Saved loss plot", path=str(output_path))
