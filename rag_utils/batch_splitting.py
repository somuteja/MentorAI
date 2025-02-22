"""
This file contains method for splitting a list of texts into batches based on the number of tokens in the text.
We will use the cl100k_base encoding scheme to count the number of tokens in the text and we split into batches of texts
with a maximum of BATCH_SIZE_IN_TOKENS.
"""

import logging
from helper_methods.tokens import num_tokens_from_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ENCODING_NAME = "cl100k_base"

def batch_splitter(texts: list[str], batch_size_in_tokens: int) -> list[list[str]]:
    """
    Args:
        texts (list[str]): A list of texts to split into batches.
        batch_size_in_tokens (int): The maximum number of tokens in a batch.

    Returns:
        list[list[str]]: A list of batches of texts.
    """
    try:
        batches = []
        current_batch = []
        current_batch_size = 0
        for text in texts:
            num_tokens = num_tokens_from_string(text, encoding_name=ENCODING_NAME)
            if current_batch_size + num_tokens <= batch_size_in_tokens:
                current_batch.append(text)
                current_batch_size += num_tokens
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_batch_size = num_tokens
        if current_batch:
            batches.append(current_batch)
        return batches
    except Exception as e:
        logger.error(f"Error splitting batches: {e}")
        return []   
