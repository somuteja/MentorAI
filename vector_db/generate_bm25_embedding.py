from vector_db.embedding_models import BM25_EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_bm25_embedding(text: str):
    """
    Generate a BM25 embedding for the given text
    
    Args:
        text: Text to generate embedding for
    """
    try:
        embedding = next(BM25_EMBEDDING_MODEL.passage_embed(text))
        return embedding
    except Exception as e:
        logger.error(f"Error generating BM25 embedding: {e}")
        return None
    
def generate_bm25_query_embedding(query: str):
    """
    Generate a BM25 query embedding for the given query
    
    Args:
        query: Query to generate embedding for
    """
    try:
        embedding = next(BM25_EMBEDDING_MODEL.query_embed(query))
        return embedding
    except Exception as e:
        logger.error(f"Error generating BM25 query embedding: {e}")
        return None
    
