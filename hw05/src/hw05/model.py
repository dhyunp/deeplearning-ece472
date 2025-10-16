import jax
import structlog
from flax import nnx
from jax import numpy as jnp

log = structlog.get_logger()


class GLU(nnx.Module):
    def __init__(
        self,
        *,
        input_layer_depth: int,
        output_layer_depth: int,
        rngs: nnx.Rngs,
    ):
        self.glu = nnx.Linear(input_layer_depth, output_layer_depth * 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate, activation = jnp.split(self.glu(x), 2, axis=-1)
        return gate * nnx.sigmoid(activation)


class HiddenLayer(nnx.Module):
    def __init__(
        self,
        *,
        layer_depth: int,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.GroupNorm(num_features=layer_depth, rngs=rngs)
        self.hidden_layer = GLU(
            input_layer_depth=layer_depth, output_layer_depth=layer_depth, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.hidden_layer(self.norm(x))


class HiddenLayers(nnx.Module):
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        layer_depth: int,
        rngs: nnx.Rngs,
    ):
        @nnx.split_rngs(splits=num_hidden_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layers(rngs: nnx.Rngs):
            return HiddenLayer(layer_depth=layer_depth, rngs=rngs)

        self.layers = create_layers(rngs)
        self.num_hidden_layers = num_hidden_layers

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.num_hidden_layers > 0:

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
            def forward_body(carry, layer):
                # skip connection
                return nnx.relu(carry + layer(carry))

            x = forward_body(x, self.layers)
        return x


class MLP(nnx.Module):
    """A Flax NNX module for an MLP model."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        vocab_size: int,
        embedding_dim: int,        
        hidden_layer_depth: int,
        num_hidden_layers: int,
        num_classes: int,
    ):
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embedding_dim,
            rngs=rngs,
        )

        self.input_layer = GLU(
            input_layer_depth=embedding_dim,
            output_layer_depth=hidden_layer_depth,
            rngs=rngs,
        )

        self.hidden_layers = HiddenLayers(
            num_hidden_layers=num_hidden_layers,
            layer_depth=hidden_layer_depth,
            rngs=rngs,
        )

        self.output_layer = nnx.Linear(hidden_layer_depth, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Iterates through the MLP layers."""
        x = self.embedding(x)
        x = jnp.mean(x, axis=1)
        
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
