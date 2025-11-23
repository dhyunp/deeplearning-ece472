import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .logging import configure_logging
from .config import load_settings
from .data import Data
from .model import NNXMLPModel, SparseAutoEncoder
from .training import train_mlp, train_sae
from .plotting import plot_fit, plot_latent_features  # Updated import

"""
Discussion:
The Sparse Autoencoder has been trained to learn features about the Spirals MLP.
Experiments were done with SAE of dimension 1024 and 2048.
The features learned by the SAE seem to strongly correspond with particular spirals, as shown in the graphs.
For example, in the first figure page, the SAE was trained with 2048 dimensions, feature 1177 activates strongly on the blue spiral's overall shape.
In the second figure page, the SAE was trained with 1024 dimensions, feature 129 seems to strongly activate on the red spiral's overall shape.
Other features activate on smaller subsections of sprials, indicating that they have learned localized features of spirals.
These localized features do not cross over to the opposite spiral, indicating that the localized features are also spiral discrimiative.
"""


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Hello from hw07!")
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key, sae_key = jax.random.split(key, 3)
    log.debug("keys", key=key, data_key=data_key, model_key=model_key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        num_samples_per_spiral=settings.data.num_samples_per_spiral,
        sigma=settings.data.sigma_noise,
    )
    # log.debug("Data generated", x=data.x, y=data.y)

    model = NNXMLPModel(
        key=model_key,
        num_inputs=settings.model.num_inputs,
        num_outputs=settings.model.num_outputs,
        num_hidden_layers=settings.model.num_hidden_layers,
        hidden_layer_width=settings.model.hidden_layer_width,
        hidden_activation=nnx.relu,
        output_activation=nnx.identity,
    )

    learning_rate_schedule = optax.exponential_decay(
        init_value=settings.training.learning_rate,
        transition_steps=5000,
        decay_rate=0.1,
    )

    mlp_optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=settings.training.l2_reg,
        ),
        wrt=nnx.Param,
    )

    train_mlp(model, mlp_optimizer, data, settings.training, np_rng)

    sae = SparseAutoEncoder(
        key=sae_key,
        hidden_layer_width=settings.model.hidden_layer_width,
        latent_dim=settings.model.latent_dim,
    )

    sae_optimizer = nnx.Optimizer(
        sae,
        optax.adamw(
            learning_rate=settings.training.sae_learning_rate,
            weight_decay=settings.training.l2_reg,
        ),
        wrt=nnx.Param,
    )

    train_sae(model, sae, sae_optimizer, data, settings.training, np_rng)

    plot_fit(model, data, settings.plotting)

    plot_latent_features(model, sae, data, settings.plotting)
