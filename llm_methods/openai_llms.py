import logging
from llm_methods.config_files.openai_config import (OPENAI_CLIENT, OPENAI_MODELS_DICT, OPENAI_EMBEDDING_MODELS_DICT)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response_from_openai_models(model: str,
                                    prompt: str,
                                    system_prompt: str = None,
                                    temperature: float = 0,
                                    max_tokens: int = 4096) -> str:
    """Get output from openai models

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

    if model not in OPENAI_MODELS_DICT:
        logger.error(f"Model '{model}' not found in OPENAI_MODELS_DICT.")
        raise ValueError(f"Model '{model}' not found in OPENAI_MODELS_DICT."
                         f" Currently the model is not supported.")

    try:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})
        
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODELS_DICT[model],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens)

        if not response.choices or not response.choices[0].message:
            logger.error("Invalid response structure received from the model")
            raise ValueError("Invalid response structure received from the model")

        return response.choices[0].message.content
    except Exception as e:
        logger.exception(f"Failed to get response from model '{model}'")
        raise RuntimeError(f"Failed to get response from model '{model}': {e}")

def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding from openai models

    Args:
        text (str): The text to get embedding from.
        model (str, optional): The name of the model to use. Defaults to "text-embedding-3-small".

    Returns:
        list[float]: The embedding from the model.

    Raises:
        ValueError: If the model is not found or the response structure is invalid.
        RuntimeError: If the API call fails.
    """
    try:
        if model not in OPENAI_EMBEDDING_MODELS_DICT:
            logger.error(f"Model '{model}' not found in OPENAI_EMBEDDING_MODELS_DICT.")
            raise ValueError(f"Model '{model}' not found in OPENAI_EMBEDDING_MODELS_DICT."
                         f" Currently the model is not supported.")
        text = text.replace("\n", " ")
        return OPENAI_CLIENT.embeddings.create(input = [text], model=model).data[0].embedding
    except Exception as e:
        logger.exception(f"Failed to get embedding from model '{model}'")
        raise RuntimeError(f"Failed to get embedding from model '{model}': {e}")    
