import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import structlog
from sklearn.inspection import DecisionBoundaryDisplay
from tqdm import tqdm

from .config import PlottingSettings
from .data import Data
from .model import NNXMLPModel, SparseAutoEncoder

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(
    model: NNXMLPModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots the graph and saves it to a file."""
    log.info("Plotting Graph")
    fig, ax = plt.subplots(figsize=settings.figsize, dpi=settings.dpi)

    x_min, x_max = (
        data.spiral_coordinates[:, 0].min() - 1,
        data.spiral_coordinates[:, 0].max() + 1,
    )
    y_min, y_max = (
        data.spiral_coordinates[:, 1].min() - 1,
        data.spiral_coordinates[:, 1].max() + 1,
    )
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    z = model(jnp.column_stack([xx.ravel(), yy.ravel()]))
    predicts = (jax.nn.sigmoid(z) > 0.5).astype(int)
    predicts = predicts.reshape(xx.shape)

    display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=predicts)
    display.plot(ax=ax, cmap=plt.cm.RdBu, alpha=0.5)

    ax.scatter(
        data.spiral_coordinates[:, 0],
        data.spiral_coordinates[:, 1],
        c=data.spiral_labels,
        cmap=plt.cm.RdBu,
        edgecolors="k",
        s=100,
    )

    ax.set_title("Decision Boundary of Spirals")
    ax.set_xlabel("x")
    ax.set_ylabel("y", labelpad=10).set_rotation(0)

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "fit.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def plot_latent_features(
    mlp: NNXMLPModel,
    sae: SparseAutoEncoder,
    data: Data,
    settings: PlottingSettings,
    num_features_to_plot: int = 9,
):
    log.info("Plotting Sparse Latent Features")

    x_min, x_max = (
        data.spiral_coordinates[:, 0].min() - 1,
        data.spiral_coordinates[:, 0].max() + 1,
    )
    y_min, y_max = (
        data.spiral_coordinates[:, 1].min() - 1,
        data.spiral_coordinates[:, 1].max() + 1,
    )
    mesh_step = 0.05
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step), np.arange(y_min, y_max, mesh_step)
    )
    coords = jnp.column_stack([xx.ravel(), yy.ravel()])

    dense_hidden_state_z = mlp.extract_final_hidden_state(coords)

    latent_activations = sae.encode(dense_hidden_state_z)
    mean_activations = jnp.mean(latent_activations, axis=0)
    top_indices = jnp.argsort(mean_activations)[::-1][:num_features_to_plot]

    if jnp.all(mean_activations == 0):
        log.warning(
            "All latent features have zero mean activation. No features to plot."
        )
        return

    nrows_cols = int(np.ceil(np.sqrt(num_features_to_plot)))
    fig, axes = plt.subplots(
        nrows=nrows_cols, ncols=nrows_cols, figsize=(12, 12), dpi=settings.dpi
    )
    axes = axes.flatten()

    for i, ax in enumerate(tqdm(axes, desc="Generating feature plots")):
        if i >= len(top_indices):
            ax.axis("off")
            continue

        feature_index = top_indices[i]

        feature_activations_1d = latent_activations[:, feature_index]
        feature_map = feature_activations_1d.reshape(xx.shape)

        max_val = jnp.max(feature_activations_1d)
        max_idx = jnp.argmax(feature_activations_1d)
        max_coords = coords[max_idx, :]
        x_at_max = max_coords[0]
        y_at_max = max_coords[1]

        ax.set_title(
            f"N={feature_index} (Max={max_val:.2f} @ ({x_at_max:.1f}, {y_at_max:.1f}))",
            fontsize=10,
        )

        c = ax.pcolormesh(xx, yy, feature_map, cmap="viridis", shading="auto")

        ax.scatter(
            data.spiral_coordinates[:, 0],
            data.spiral_coordinates[:, 1],
            c=data.spiral_labels,
            cmap=plt.cm.RdBu,
            edgecolors="k",
            s=5,
            alpha=0.2,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        plt.colorbar(c, ax=ax, orientation="vertical", shrink=0.6)

    plt.suptitle(
        f"Activation Maps of Top {num_features_to_plot} Sparse Latent Features",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "latent_features.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
