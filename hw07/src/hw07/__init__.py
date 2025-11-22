import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .logging import configure_logging
from .config import load_settings
from .data import Data
from .model import NNXMLPModel
from .training import train
from .plotting import plot_fit

"""
Discussion:
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
    data_key, model_key = jax.random.split(key)
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

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=settings.training.l2_reg,
        ),
        wrt=nnx.Param,
    )

    train(model, optimizer, data, settings.training, np_rng)

    plot_fit(model, data, settings.plotting)
