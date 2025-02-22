"""Methods for counting and converting tokens. Note that this is not a comprehensive list of all tokenization methods.
 Only few encoding schemes are supported."""

import logging
import tiktoken

DEFAULT_ENCODING_NAME = "cl100k_base"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    if model is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            msg = f"Failed to get encoding for {model} when getting num_tokens_from_string. Fall back to default encoding {DEFAULT_ENCODING_NAME}"
            logger.warning(msg)
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
    else:
        encoding = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)
    return len(encoding.encode(string))

def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    if model is not None:
        encoding = tiktoken.encoding_for_model(model)
    elif encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        msg = "Either model or encoding_name must be specified."
        raise ValueError(msg)
    return encoding.decode(tokens)
