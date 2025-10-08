import jax
import structlog
from flax import nnx
from jax import numpy as jnp

log = structlog.get_logger()


class GroupNorm(nnx.Module):
    """A Flax NNX module for Group Normalization."""

    def __init__(self, *, rngs: nnx.Rngs, num_groups: int, num_features: int):
        self.norm = nnx.GroupNorm(
            num_groups=num_groups, num_features=num_features, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x)


class Conv2d(nnx.Module):
    """A Flax NNX module for a 2D convolutional layer."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        strides: int,
    ):
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class ResidualBlock(nnx.Module):
    """A Flax NNX module for a residual block."""

    def __init__(
        self,
        *,
        key: jax.random.PRNGKey,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        num_groups: int,
        strides: int,
    ):
        res_keys = jax.random.split(key, 5)

        self.conv1 = Conv2d(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            rngs=nnx.Rngs(params=res_keys[0]),
        )
        self.gn1 = GroupNorm(
            num_groups=num_groups,
            num_features=in_features,
            rngs=nnx.Rngs(params=res_keys[1]),
        )
        self.conv2 = Conv2d(
            in_features=out_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=1,
            rngs=nnx.Rngs(params=res_keys[2]),
        )
        self.gn2 = GroupNorm(
            num_groups=num_groups,
            num_features=out_features,
            rngs=nnx.Rngs(params=res_keys[3]),
        )

        if (in_features != out_features) or (strides != 1):
            self.shortcut = Conv2d(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                rngs=nnx.Rngs(params=res_keys[4]),
            )
        else:
            self.shortcut = lambda x: x  # Identity

        self.activation = jax.nn.relu

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = self.shortcut(x)
        fx = self.gn1(x)
        fx = self.activation(fx)
        fx = self.conv1(fx)
        fx = self.gn2(fx)
        fx = self.activation(fx)
        fx = self.conv2(fx)

        return residual + fx


class Classifier(nnx.Module):
    """A Flax NNX module for an CNN model."""

    def __init__(
        self,
        *,
        key: jax.random.PRNGKey,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
        num_groups: int,
        strides: list[int],
    ):
        keys = jax.random.split(key, len(layer_depths) + 2)
        # layer_depths and layer_kernel_sizes must have the same length
        assert len(layer_depths) == len(layer_kernel_sizes)

        self.start_conv = Conv2d(
            rngs=nnx.Rngs(params=keys[0]),
            in_features=input_depth,
            out_features=layer_depths[0],
            kernel_size=layer_kernel_sizes[0],
            strides=strides[0],
        )

        self.res_layers = nnx.List([])

        for i in range(1, len(layer_depths)):
            self.res_layers.append(
                ResidualBlock(
                    key=keys[i],
                    in_features=layer_depths[i - 1],
                    out_features=layer_depths[i],
                    kernel_size=layer_kernel_sizes[i],
                    num_groups=num_groups[i],
                    strides=strides[i],
                )
            )

        self.dense = nnx.Linear(
            in_features=layer_depths[-1],
            out_features=num_classes,
            rngs=nnx.Rngs(params=keys[-1]),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # """Iterates through the CNN layers."""
        # x = jax.nn.relu(self.start_gn(self.start_conv(x)))
        x = jax.nn.relu(self.start_conv(x))
        for layer in self.res_layers:
            x = layer(x)

        x = jnp.mean(x, axis=(1, 2))
        x = self.dense(x)

        return x
