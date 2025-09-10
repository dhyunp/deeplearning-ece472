from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class GaussianModel:
    """Represents a simple gaussian model."""

    weights: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    bias: float


class NNXGaussianModel(nnx.Module):
    """A Flax NNX module for a gaussian regression model."""

    def __init__(self, *, rngs: nnx.Rngs, num_features: int):
        self.num_features = num_features
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (self.num_features, 1)))  # Weights
        self.mu = nnx.Param(jax.random.normal(key, (1, self.num_features)))  # Mean
        self.sigma = nnx.Param(
            jnp.array([0.1] * self.num_features).reshape(1, -1)
        )  # SD
        self.b = nnx.Param(jnp.zeros((1, 1)))  # Bias

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output array y_hat for given input array x."""
        phi = jnp.exp(
            -((x - self.mu.value) ** 2) / (self.sigma.value**2)
        )  # exp((-(x - mu)^2) / sigma^2)
        return phi @ self.w.value + self.b.value

    @property
    def model(self) -> GaussianModel:
        """Returns the underlying simple gaussian model."""
        return GaussianModel(
            weights=np.array(self.w.value).reshape([self.num_features]),
            mean=np.array(self.mu.value).reshape([self.num_features]),
            sd=np.array(self.sigma.value).reshape([self.num_features]),
            bias=np.array(self.b.value).squeeze(),
        )
