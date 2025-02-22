"""Methods for splitting text into chunks based on token count of cl100k_base encoding scheme."""

import logging
from tiktoken import get_encoding
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ENCODING_NAME = "cl100k_base"

#text splitting method which has two parameters, chunk_overlap and tokens_per_chunk
def text_splitter(text: str, chunk_overlap: int, tokens_per_chunk: int) -> list[str]:
    """
    Args:
        text (str): The text to split into chunks.
        chunk_overlap (int): The number of tokens to overlap between chunks.
        tokens_per_chunk (int): The number of tokens per chunk.

    Returns:
        list[str]: A list of chunks of text.
    """
    try:
        if tokens_per_chunk <= 0:
            raise ValueError("tokens_per_chunk must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be positive")
        if chunk_overlap >= tokens_per_chunk:
            raise ValueError("chunk_overlap must be less than tokens_per_chunk")
        
        encoding = get_encoding(DEFAULT_ENCODING_NAME)
        encoded_tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(encoded_tokens), tokens_per_chunk - chunk_overlap):
            chunk = encoded_tokens[i:i+tokens_per_chunk]
            chunks.append(encoding.decode(chunk))
        return chunks
    except Exception as e:
        logger.error(f"Error in split_text: {str(e)}")
        return []
    
