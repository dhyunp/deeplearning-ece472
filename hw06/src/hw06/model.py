import jax
import structlog
from flax import nnx
from jax import numpy as jnp

log = structlog.get_logger()


class SingleHeadAttention(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        embedding_dim: int,
        head_dim: int,
        testing: bool = False,
    ):
        self.head_dim = head_dim
        self.query = nnx.Linear(
            in_features=embedding_dim, out_features=head_dim, use_bias=False, rngs=rngs
        )
        self.key = nnx.Linear(
            in_features=embedding_dim, out_features=head_dim, use_bias=False, rngs=rngs
        )
        self.value = nnx.Linear(
            in_features=embedding_dim, out_features=head_dim, use_bias=False, rngs=rngs
        )
        self.testing = testing

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        tril = jnp.tril(jnp.ones(shape=(T, T), dtype=bool))
        tril = jnp.expand_dims(tril, axis=0)

        weights = query @ key.transpose((0, -1, -2)) * (self.head_dim**-0.5)
        weights = jnp.where(tril, weights, -jnp.inf)
        weights = nnx.softmax(weights, axis=-1)
        out = weights @ value

        return (out, weights) if self.testing else out


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
    ):
        self.heads = nnx.List(
            SingleHeadAttention(
                embedding_dim=embedding_dim, head_dim=head_dim, rngs=rngs
            )
            for _ in range(num_heads)
        )
        self.proj = nnx.Linear(
            in_features=embedding_dim, out_features=embedding_dim, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.concatenate([head(x) for head in self.heads], axis=-1)
        x = self.proj(x)

        return x


class Block(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        embedding_dim: int,
        num_heads: int,
    ):
        head_dim = embedding_dim // num_heads
        self.self_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            rngs=rngs,
        )
        self.ff = FeedForward(
            embedding_dim=embedding_dim,
            rngs=rngs,
        )
        self.ln1 = nnx.LayerNorm(num_features=embedding_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class FeedForward(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        embedding_dim: int,
    ):
        self.ff = nnx.Sequential(
            nnx.Linear(
                in_features=embedding_dim, out_features=embedding_dim * 4, rngs=rngs
            ),
            nnx.relu,
            nnx.Linear(
                in_features=embedding_dim * 4, out_features=embedding_dim, rngs=rngs
            ),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.ff(x)


class Transformer(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        context_size: int,
        num_layers: int,
    ):
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size, features=embedding_dim, rngs=rngs
        )
        self.pos_embedding = nnx.Embed(
            num_embeddings=context_size, features=embedding_dim, rngs=rngs
        )
        blocks = [
            Block(embedding_dim=embedding_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.blocks = nnx.Sequential(*blocks)
        self.ln = nnx.LayerNorm(num_features=embedding_dim, rngs=rngs)
        self.output = nnx.Linear(
            in_features=embedding_dim, out_features=vocab_size, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T = x.shape
        token_embeddings = self.token_embedding(x)
        pos_embeddings = self.pos_embedding(jnp.arange(T))

        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.ln(x)
        x = self.output(x)

        return x
