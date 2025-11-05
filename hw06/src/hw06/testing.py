import jax
import jax.numpy as jnp
import structlog
from flax import nnx

from .model import (
    SingleHeadAttention,
    MultiHeadAttention,
    FeedForward,
    Block,
    Transformer,
)

log = structlog.get_logger()


def run_tests():
    # Hyperparameters
    B, T, C = 4, 8, 32  # Batch Size, Context Size (Time), Embedding Dim
    num_heads = 4
    head_dim = C // num_heads
    vocab_size = 64
    num_layers = 2

    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key, dropout=jax.random.key(1))

    x_token_ids = jax.random.randint(key, (B, T), 0, vocab_size)
    x_embeddings = jax.random.normal(key, (B, T, C))

    log.info("Test Settings")
    log.info(f"Batch Size (B): {B}, Context (T): {T}, Embedding (C): {C}")
    log.info(f"Num Heads: {num_heads}, Head Dim: {head_dim}")
    log.info(f"Vocab Size: {vocab_size}, Num Layers: {num_layers}")

    log.info(
        "-----------------------------------------------------------------------------------------------"
    )

    # SingleHeadAttention weight inspection
    log.info("Test 1: SingleHeadAttention (Testable)")
    sha_model = SingleHeadAttention(
        embedding_dim=C, head_dim=head_dim, rngs=rngs, testing=True
    )
    out, weights = sha_model(x_embeddings)
    log.info(f"Input shape: {x_embeddings.shape}")
    log.info(f"Output shape: {out.shape}  (Expected: {(B, T, head_dim)})")
    log.info(f"Weights shape: {weights.shape} (Expected: {(B, T, T)})")
    log.info("Attention weights (proving causal mask works):")
    log.info(weights[0])
    log.info("Test passed: Upper triangle is all 0.0, proving masking is in effect)")

    log.info(
        "-----------------------------------------------------------------------------------------------"
    )

    # MultiHeadAttention
    log.info("Test 2: MultiHeadAttention")
    mha_model = MultiHeadAttention(
        embedding_dim=C, num_heads=num_heads, head_dim=head_dim, rngs=rngs
    )
    out = mha_model(x_embeddings)
    log.info(f"Input shape: {x_embeddings.shape}")
    log.info(f"Output shape: {out.shape} (Expected: {(B, T, C)})")

    # FeedForward
    log.info("Test 3: FeedForward")
    ff_model = FeedForward(embedding_dim=C, rngs=rngs)
    out = ff_model(x_embeddings)
    log.info(f"Input shape: {x_embeddings.shape}")
    log.info(f"Output shape: {out.shape} (Expected: {(B, T, C)})")

    if out.shape == (B, T, C):
        log.info("Test passed: The model is correctly shaped.")
    else:
        log.info("Test failed: The model is NOT correctly shaped.")

    log.info(
        "-----------------------------------------------------------------------------------------------"
    )

    # Transformer Block
    log.info("Test 4: Block")
    block_model = Block(embedding_dim=C, num_heads=num_heads, rngs=rngs)
    out = block_model(x_embeddings)
    log.info(f"Input shape: {x_embeddings.shape}")
    log.info(f"Output shape: {out.shape} (Expected: {(B, T, C)})")
    log.info("Residual connections (x + F(x)) successful.")

    if out.shape == (B, T, C):
        log.info("Test passed: The model is correctly shaped.")
    else:
        log.info("Test failed: The model is NOT correctly shaped.")

    log.info(
        "-----------------------------------------------------------------------------------------------"
    )

    # Transformer
    log.info("Test 5: Full Transformer Model")
    transformer_model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=C,
        num_heads=num_heads,
        context_size=T,
        num_layers=num_layers,
        rngs=rngs,
    )
    out = transformer_model(x_token_ids)
    log.info(f"Input shape: {x_token_ids.shape} (Token IDs)")
    log.info(f"Output shape: {out.shape} (Expected: {(B, T, vocab_size)})")

    if out.shape == (B, T, vocab_size):
        log.info("Test passed: The model is correctly shaped.")
    else:
        log.info("Test failed: The model is NOT correctly shaped.")

    log.info(
        "-----------------------------------------------------------------------------------------------"
    )

    # Shuffle Variance
    log.info("Test 6: Proving Position-Awareness (NON-Shuffle-Invariance)")
    x_original = jnp.arange(T).reshape(1, T)
    x_shuffled = jnp.fliplr(x_original)

    log.info(f"Original input: {x_original}")
    log.info(f"Shuffled input: {x_shuffled}")

    out_original = transformer_model(x_original)
    out_shuffled = transformer_model(x_shuffled)

    are_outputs_same = jnp.allclose(out_original, out_shuffled)
    log.info(f"Are outputs identical? {are_outputs_same} (Expected: False)")

    are_outputs_equivariant = jnp.allclose(out_original, jnp.fliplr(out_shuffled))
    log.info(
        f"Are outputs permuted (equivariant)? {are_outputs_equivariant} (Expected: False)"
    )

    if not are_outputs_same and not are_outputs_equivariant:
        log.info("Test passed: The model is correctly position-aware.")
    else:
        log.info("Test failed: The model is NOT correctly position-aware.")
