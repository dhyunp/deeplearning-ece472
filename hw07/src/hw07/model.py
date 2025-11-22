import jax
import structlog
from flax import nnx


log = structlog.get_logger()


class NNXMLPModel(nnx.Module):
    """A Flax NNX module for an MLP model."""

    def __init__(
        self,
        *,
        key: jax.random.PRNGKey,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int,
        hidden_layer_width: int,
        hidden_activation=nnx.identity,
        output_activation=nnx.identity,
    ):
        keys = jax.random.split(key, num_hidden_layers + 2)
        log.debug("RNG keys generated", original_key=key, keys=keys)
        input_rngs = nnx.Rngs(params=keys[0])
        output_rngs = nnx.Rngs(params=keys[-1])
        hidden_keys = keys[1:-1]

        self.input_layer = nnx.Linear(num_inputs, hidden_layer_width, rngs=input_rngs)
        self.hidden_layers = []
        if num_hidden_layers > 0:

            def create_hidden_layer(key: jax.random.PRNGKey) -> nnx.Linear:
                hidden_rngs = nnx.Rngs(params=key)
                return nnx.Linear(
                    hidden_layer_width, hidden_layer_width, rngs=hidden_rngs
                )

            vmap_create_layer = jax.vmap(create_hidden_layer)
            self.hidden_layers = vmap_create_layer(hidden_keys)

        self.output_layer = nnx.Linear(
            hidden_layer_width, num_outputs, rngs=output_rngs
        )
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, coords: jax.Array) -> jax.Array:
        # """Iterates through the MLP layers."""
        coords = self.hidden_activation(self.input_layer(coords))

        def forward(carry, hidden_layer):
            output_coords = self.hidden_activation(hidden_layer(carry))
            return output_coords, None

        coords, _ = jax.lax.scan(f=forward, init=coords, xs=self.hidden_layers)

        return self.output_activation(self.output_layer(coords))
