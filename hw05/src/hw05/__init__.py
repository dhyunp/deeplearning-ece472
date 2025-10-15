import structlog
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path

from .logging import configure_logging
from .config import load_settings
from .data import Data
from .model import MLP
from .training_testing import train, test


def main() -> None:
    """CLI entry point."""
    configure_logging()
    log = structlog.get_logger()
    log.info("Hello from hw05!")

    settings = load_settings()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        dataset_name=settings.data.dataset_name,
        percent_train=settings.data.percent_train,
    )

    log.info("Loaded dataset", dataset=settings.data.dataset_name)

    model = MLP(
        rngs=nnx.Rngs(params=model_key),
        input_depth=settings.model.input_depth,
        hidden_layer_depth=settings.model.layer_depths,
        num_hidden_layers=settings.model.num_hidden_layers,
        num_classes=settings.model.num_classes,
        output_activation=nnx.identity,
    )

    learning_rate_schedule = optax.exponential_decay(
        init_value=settings.training.learning_rate,
        transition_steps=2500,
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
    log.info("Finished Training model")

    save_dir = settings.saving.output_dir
    log.info("savedir", save_dir=save_dir)
    save_dir = Path.cwd() / save_dir
    ckpt_dir = ocp.test_utils.erase_and_create_empty(save_dir)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "state", state)
    checkpointer.wait_until_finished()
    log.info("Saved model")


def run_test() -> None:
    """CLI entry point."""
    configure_logging()
    log = structlog.get_logger()
    log.info("Running test!")

    settings = load_settings()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        dataset_name=settings.data.dataset_name,
        percent_train=settings.data.percent_train,
    )

    log.info("Loaded dataset", dataset=settings.data.dataset_name)

    model = MLP(
        rngs=nnx.Rngs(params=model_key),
        input_depth=settings.model.input_depth,
        hidden_layer_depth=settings.model.layer_depths,
        num_hidden_layers=settings.model.num_hidden_layers,
        num_classes=settings.model.num_classes,
    )

    # recreate model
    ckpt_dir = Path.cwd() / settings.saving.output_dir
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state_restored = checkpointer.restore(ckpt_dir / "state", state)
    model = nnx.merge(graphdef, state_restored)

    log.info("Loaded model")

    test(model, data)
