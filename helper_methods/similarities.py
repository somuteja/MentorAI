import numpy as np

def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Calculate the cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
