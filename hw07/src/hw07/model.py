import jax
import structlog
from flax import nnx
import jax.numpy as jnp


log = structlog.get_logger()


class NNXMLPModel(nnx.Module):
    """A Flax NNX module for an MLP-based Sparse Autoencoder with a classification head."""

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

    def extract_final_hidden_state(self, coords: jax.Array) -> jnp.ndarray:
        coords = self.hidden_activation(self.input_layer(coords))

        def forward(carry, hidden_layer):
            output_coords = self.hidden_activation(hidden_layer(carry))
            return output_coords, None

        final_hidden_state, _ = jax.lax.scan(
            f=forward, init=coords, xs=self.hidden_layers
        )
        return final_hidden_state

    def __call__(self, coords: jax.Array) -> jax.Array:
        final_hidden_state = self.extract_final_hidden_state(coords)

        return self.output_activation(self.output_layer(final_hidden_state))


class SparseAutoEncoder(nnx.Module):
    def __init__(
        self,
        *,
        key: jax.random.PRNGKey,
        hidden_layer_width: int,
        latent_dim: int,
    ):
        keys = jax.random.split(key, 3)

        self.encoder_layer = nnx.Linear(
            hidden_layer_width, latent_dim, rngs=nnx.Rngs(params=keys[0])
        )
        self.decoder_layer = nnx.Linear(
            latent_dim, hidden_layer_width, rngs=nnx.Rngs(params=keys[1])
        )
        self.latent_activation = nnx.relu
        self.latent_dim = latent_dim

    def encode(self, final_hidden_state: jnp.ndarray) -> jnp.ndarray:
        latent_vector = self.latent_activation(self.encoder_layer(final_hidden_state))
        return latent_vector

    def decode(self, latent_vector: jnp.ndarray) -> jnp.ndarray:
        return self.decoder_layer(latent_vector)

    def __call__(
        self, final_hidden_state: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        latent_vector = self.encode(final_hidden_state)
        reconstruction = self.decode(latent_vector)
        return reconstruction, latent_vector
