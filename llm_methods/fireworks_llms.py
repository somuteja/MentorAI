import logging
from llm_methods.config_files.fireworks_config import (FIREWORKS_CLIENT, FIREWORKS_MODELS_URL_DICT)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response_from_fireworks_models(model: str,
                                       prompt: str,
                                       temperature: float = 0,
                                       max_tokens: int = 4096,
                                       context_length_exceeded_behavior: str = "truncate") -> str:
    """Get output from fireworks models

    Args:
        model (str): The name of the model to use.
        prompt (str): The prompt to send to the model.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
        context_length_exceeded_behavior (str, optional): Behavior when context length is exceeded. Defaults to 'truncate'.

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model is not found or the response structure is invalid.
        RuntimeError: If the API call fails.
    """

    if model not in FIREWORKS_MODELS_URL_DICT:
        logger.error(f"Model '{model}' not found in FIREWORKS_MODELS_URL_DICT.")
        raise ValueError(f"Model '{model}' not found in FIREWORKS_MODELS_URL_DICT."
                         f" Currently the model is not supported.")

    try:
        response = FIREWORKS_CLIENT.chat.completions.create(
            model=FIREWORKS_MODELS_URL_DICT[model],
            messages=[{"role": "user", "content": prompt, }],
            temperature=temperature,
            max_tokens=max_tokens,
            context_length_exceeded_behavior=context_length_exceeded_behavior)

        if not response.choices or not response.choices[0].message:
            logger.error("Invalid response structure received from the model")
            raise ValueError("Invalid response structure received from the model")

        return response.choices[0].message.content
    except Exception as e:
        logger.exception(f"Failed to get response from model '{model}'")
        raise RuntimeError(f"Failed to get response from model '{model}': {e}")
    
