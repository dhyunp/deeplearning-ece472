import jax
import structlog
from flax import nnx
from jax import numpy as jnp
import numpy as np

log = structlog.get_logger()


class Conv2d(nnx.Module):
    """A Flax NNX module for a 2D convolutional layer."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
    ):
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.avg_pool(nnx.relu(self.conv(x)), window_shape=(2, 2), strides=(2, 2))


class Classifier(nnx.Module):
    """A Flax NNX module for an CNN model."""

    def __init__(
        self,
        *,
        key: jax.random.PRNGKey,
        input_height: int,
        input_width: int,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
    ):
        keys = jax.random.split(key, len(layer_depths) + 2)
        # layer_depths and layer_kernel_sizes must have the same length
        assert len(layer_depths) == len(layer_kernel_sizes)
        self.conv_layers = []
        current_depth = input_depth

        for depth, kernel_size, rng_key in zip(
            layer_depths, layer_kernel_sizes, keys[:-2]
        ):
            self.conv_layers.append(
                Conv2d(
                    rngs=nnx.Rngs(params=rng_key),
                    in_features=current_depth,
                    out_features=depth,
                    kernel_size=kernel_size,
                )
            )
            current_depth = depth

        self.dropout = nnx.Dropout(rate=0.5, rngs=nnx.Rngs(dropout=keys[-2]))

        # Compute the number of features after the conv layers
        dummy_input = jnp.ones((1, input_height, input_width, input_depth))
        dummy_output = dummy_input
        for layer in self.conv_layers:
            dummy_output = layer(dummy_output)
        dense_in_features = np.prod(dummy_output.shape[1:])

        self.dense = nnx.Linear(
            in_features=dense_in_features,
            out_features=num_classes,
            rngs=nnx.Rngs(params=keys[-1]),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # """Iterates through the CNN layers."""
        for layer in self.conv_layers:
            x = layer(x)

        x = x.reshape((x.shape[0], -1))
        x = self.dropout(x)
        x = self.dense(x)

        return x
