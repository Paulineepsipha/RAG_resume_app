from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load a pre-trained model (small and free)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: List[str]) -> List[np.ndarray]:
    """
    Embeds a list of text chunks into vectors using a SentenceTransformer model.

    Args:
        chunks (List[str]): The text chunks to embed.

    Returns:
        List[np.ndarray]: The embeddings for each chunk.
    """
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings
