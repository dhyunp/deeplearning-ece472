import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import structlog
from sklearn.inspection import DecisionBoundaryDisplay

from .config import PlottingSettings
from .data import Data
from .model import NNXMLPModel

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
