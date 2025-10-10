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
from .model import Classifier
from .training_testing import train, test

"""
Discussion:
According to Wikipedia, state of the art for CIFAR-10 and CIFAR-100 performance can be seen here:
https://en.wikipedia.org/wiki/CIFAR-10#Research_papers_claiming_state-of-the-art_results_on_CIFAR-10
From the linked papers based on the paper "Wide Residual Networks", 
as it seems to implement the ResNet architecture without too many modifications or even other approaches like transformers

ResNet-18 architecture was implemented, so ideally we would get better performance that 16 layer ResNet show on page 8,
where the accuracies were 95.44% and 78.41% for CIFAR-10, and CIFAR-100 respectively.
We will be benchmarking on these values for our system.

Different approaches like data augmentation, hyperparameter adjustments (learning rate, decay, iterations, layer depths etc) were attempted
but constrained by computational power+time was the biggest factor.
Certain hyperparamters would cause the loss value to blow up exponentially, causing the system to perform badly
In the end, numbers hyperparameters used to run training was settled on, achieving about 
72% top1-accuracy on CIFAR-10
50% top5-accuracy on CIFAR-100

"""


def main() -> None:
    """CLI entry point."""
    configure_logging()
    log = structlog.get_logger()
    log.info("Hello from hw04!")

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

    num_classes = (
        settings.model.num_classes if settings.data.dataset_name == "cifar10" else 100
    )
    model = Classifier(
        key=model_key,
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_sizes=settings.model.layer_kernel_sizes,
        num_classes=num_classes,
        num_groups=settings.model.num_groups,
        strides=settings.model.strides,
    )

    learning_rate_schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.epochs,
        alpha=0.001,
    )

    optimizer_chain = optax.chain(
        optax.sgd(
            learning_rate=learning_rate_schedule,
            momentum=settings.training.momentum,
        ),
        optax.add_decayed_weights(settings.training.l2_reg),
    )

    optimizer = nnx.Optimizer(
        model,
        optimizer_chain,
        wrt=nnx.Param,
    )

    top1_accuracy, top5_accuracy = train(
        model, optimizer, data, settings.training, np_rng
    )
    log.info("Finished Training model")

    save_dir = (
        settings.saving.output_dir_10
        if settings.data.dataset_name == "cifar10"
        else settings.saving.output_dir_100
    )
    log.info("savedir", save_dir=save_dir)
    save_dir = Path.cwd() / save_dir
    ckpt_dir = ocp.test_utils.erase_and_create_empty(save_dir)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "state", state)
    checkpointer.wait_until_finished()
    log.info("Saved model")

    if (
        settings.data.dataset_name == "cifar10"
        and (top1_accuracy > 0.9544 or top5_accuracy > 0.9)
    ) or (
        settings.data.dataset_name == "cifar100"
        and (top1_accuracy > 0.7841 or top5_accuracy > 0.9)
    ):
        log.info(
            "Achieved target accuracy on validation set",
            accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
        )
    else:
        log.info(
            "Did not achieve target accuracy on validation set",
            accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
        )


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

    num_classes = (
        settings.model.num_classes if settings.data.dataset_name == "cifar10" else 100
    )
    model = Classifier(
        key=model_key,
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_sizes=settings.model.layer_kernel_sizes,
        num_classes=num_classes,
        num_groups=settings.model.num_groups,
        strides=settings.model.strides,
    )

    # recreate model
    save_dir = (
        settings.saving.output_dir_10
        if settings.data.dataset_name == "cifar10"
        else settings.saving.output_dir_100
    )
    ckpt_dir = Path.cwd() / save_dir
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state_restored = checkpointer.restore(ckpt_dir / "state", state)
    model = nnx.merge(graphdef, state_restored)

    log.info("Loaded model")

    test(model, data)
