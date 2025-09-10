import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import structlog

from .config import PlottingSettings
from .data import Data
from .model import NNXGaussianModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(
    model: NNXGaussianModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots the fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 2, figsize=settings.figsize, dpi=settings.dpi)

    ax[0].set_title("Fit")
    ax[0].set_xlabel("x")
    ax[0].set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    ax[1].set_title("Bases for Fit")
    ax[1].set_xlabel("x")
    ax[1].set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax[1].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 10)
    xs = xs[:, np.newaxis]
    ax[0].plot(
        xs, np.squeeze(model(jnp.asarray(xs))), "-", np.squeeze(data.x), data.y, "o"
    )

    for i in range(model.num_features):
        phi = np.exp(-((xs - model.mu.value[i]) ** 2) / (model.sigma.value[i] ** 2))
        ax[1].plot(xs, phi, label=f"Basis {i + 1}")
    ax[1].legend()

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "fit.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
