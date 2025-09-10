import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .logging import configure_logging
from .config import load_settings
from .data import Data
from .model import NNXGaussianModel
from .training import train
from .plotting import plot_fit


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Hello from hw01!")
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        num_features=settings.data.num_features,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )
    # log.debug("Data generated", x=data.x, y=data.y)

    model = NNXGaussianModel(
        rngs=nnx.Rngs(params=model_key), num_features=settings.data.num_features
    )

    log.debug("Initial model", model=model.model)

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train(model, optimizer, data, settings.training, np_rng)
    log.debug("Trained model", model=model.model)

    plot_fit(model, data, settings.plotting)
