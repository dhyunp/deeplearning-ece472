import structlog
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
from sklearn.model_selection import KFold

from .logging import configure_logging
from .config import load_settings
from .data import Data
from .model import MLP
from .training_testing import train, test

"""
Discussion:
Implemented a MLP that intakes a word emebedding 384-width vector of the ag news article pass through HuggingFace's SentenceTransformer
MLP hyperparameters: 512 hidden layer nodes, 2 layers
Experimented with batch size, number of layers, number of iterations, optimizers, dropout, random batch vs iterative batch selection 
When trained on single batch, was able to reach 90% accuracy with 0.3 loss
Performance seemed to top out at 50% accuracy no matter the hyperparameter tuning
Performed 5-fold cross validations, mean-accuracy of 46-47%
Test set accuracy of 46.43%
"""


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
    )

    log.info("Loaded dataset", dataset=settings.data.dataset_name)

    kfold = KFold(n_splits=settings.training.k_folds, shuffle=True, random_state=settings.random_seed)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data.train_text_set)):
        log.info(f"Starting Fold {fold + 1}/5")

        train_embeddings = data.train_text_set[train_idx]
        train_labels = data.train_label_set[train_idx]
        val_embeddings = data.train_text_set[val_idx]
        val_labels = data.train_label_set[val_idx]

        log.info("Train/Val split", train_shape=train_embeddings.shape, val_shape=val_embeddings.shape)

        model = MLP(
            rngs=nnx.Rngs(params=model_key),
            input_depth=settings.model.input_depth,
            hidden_layer_depth=settings.model.layer_depths,
            num_hidden_layers=settings.model.num_hidden_layers,
            num_classes=settings.model.num_classes,
            output_activation=nnx.identity,
        )

        learning_rate_schedule = optax.cosine_decay_schedule(
            init_value=settings.training.learning_rate,
            decay_steps=settings.training.num_iters,
            alpha=0.001,
        )

        optimizer = nnx.Optimizer(
            model,
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=settings.training.l2_reg,
            ),
            wrt=nnx.Param,
        )

        val_accuracy = train(
            model,
            optimizer,
            [train_embeddings, train_labels, val_embeddings, val_labels],
            settings.training,
            fold,
            np_rng,
        )
        log.info(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")
        fold_accuracies.append(val_accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    log.info(
        "Cross-Validation Finished",
        mean_accuracy=f"{mean_accuracy:.4f}",
        std_deviation=f"{std_accuracy:.4f}",
    )

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

    # recreate model
    ckpt_dir = Path.cwd() / settings.saving.output_dir
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state_restored = checkpointer.restore(ckpt_dir / "state", state)
    model = nnx.merge(graphdef, state_restored)

    log.info("Loaded model")

    test(model, data)
