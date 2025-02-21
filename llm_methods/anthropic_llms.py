import logging
from llm_methods.config_files.anthropic_config import (ANTHROPIC_CLIENT, ANTHROPIC_MODELS_DICT)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response_from_anthropic_models(model: str,
                                       prompt: str,
                                       system_prompt: str = None,
                                       temperature: float = 0,
                                       max_tokens: int = 4096) -> str:
    """Get output from anthropic models

    Args:
        model (str): The name of the model to use.
        prompt (str): The prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model is not found or the response structure is invalid.
        RuntimeError: If the API call fails.
    """

    if model not in ANTHROPIC_MODELS_DICT:
        logger.error(f"Model '{model}' not found in ANTHROPIC_MODELS_DICT.")
        raise ValueError(f"Model '{model}' not found in ANTHROPIC_MODELS_DICT."
                         f" Currently the model is not supported.")

    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] }]
        
        if system_prompt:
            response = ANTHROPIC_CLIENT.messages.create(
            model=ANTHROPIC_MODELS_DICT[model],
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=temperature
            )
        else:
            response = ANTHROPIC_CLIENT.messages.create(
            model=ANTHROPIC_MODELS_DICT[model],
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature
            )
        

        if not response.content:
            logger.error("Invalid response structure received from the model")
            raise ValueError("Invalid response structure received from the model")

        return response.content[0].text
    except Exception as e:
        logger.exception(f"Failed to get response from model '{model}'")
        raise RuntimeError(f"Failed to get response from model '{model}': {e}")
