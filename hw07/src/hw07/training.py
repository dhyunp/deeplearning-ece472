import jax.numpy as jnp
import numpy as np
import structlog
import optax
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import NNXMLPModel, SparseAutoEncoder

log = structlog.get_logger()


@nnx.jit
def train_mlp_step(
    model: NNXMLPModel, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
):
    """Performs a single training step."""

    def loss_fn(model: NNXMLPModel):
        y_hat = model(x)
        return optax.sigmoid_binary_cross_entropy(y_hat, y.reshape(y_hat.shape)).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place update of model parameters
    return loss


def train_mlp(
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

        loss = train_mlp_step(model, optimizer, coords, labels)
        bar.set_description(f"MLP Loss @ {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training finished")


@nnx.jit
def train_sae_step(
    sae: SparseAutoEncoder,
    optimizer: nnx.Optimizer,
    z: jnp.ndarray,
    lambda_sparsity: float,
):
    """Performs a single training step for the Sparse Autoencoder."""

    def loss_fn(sae: SparseAutoEncoder):
        z_hat, h = sae(z)

        reconstruction_loss = jnp.mean((z - z_hat) ** 2)

        sparsity_loss = jnp.mean(jnp.abs(h))

        total_loss = reconstruction_loss + lambda_sparsity * sparsity_loss

        return total_loss, (reconstruction_loss, sparsity_loss)

    (total_loss, (recon_loss, sparse_loss)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(sae)

    optimizer.update(sae, grads)

    return total_loss, recon_loss, sparse_loss


def train_sae(
    mlp: NNXMLPModel,
    sae: SparseAutoEncoder,
    sae_optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    log.info("Starting SAE Training (Stage 1)", **settings.model_dump())

    all_coords = jnp.asarray(data.spiral_coordinates)
    all_z = mlp.extract_final_hidden_state(all_coords)

    bar = trange(settings.num_iters // 2)
    for i in bar:
        choices = np_rng.choice(all_z.shape[0], size=settings.batch_size)
        z_batch = all_z[choices]

        total_loss, recon_loss, sparse_loss = train_sae_step(
            sae, sae_optimizer, z_batch, settings.lambda_sparsity
        )

        bar.set_description(
            f"SAE Loss @ {i} => L_tot={total_loss:.4f} | L_recon={recon_loss:.4f} | L_sparse={sparse_loss:.4f}"
        )
    log.info("SAE Training finished")
