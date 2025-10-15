import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


def create_embeddings(text_batch: np.ndarray):
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    embeddings = model.encode(text_batch)

    log.debug("Embedding shape", embedding_shape=embeddings.shape)
    log.debug("Sample word embedding", word_embedding=embeddings[0])

    return embeddings
