import logging
from llm_methods.config_files.google_config import (GOOGLE_MODELS_DICT, SAFETY_SETTINGS, DIFFERENT_TOP_K_MODELS)
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response_from_google_models(model: str,
                                    prompt: str,
                                    system_prompt: str = None,
                                    temperature: float = 0,
                                    max_tokens: int = 4096) -> str:
    """Get output from google models

    Args:
        model (str): The name of the model to use.
        prompt (str): The prompt to send to the model.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to None.
    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model is not found.
        RuntimeError: If the API call fails.
    """

    if model not in GOOGLE_MODELS_DICT:
        logger.error(f"Model '{model}' not found in GOOGLE_MODELS_DICT.")
        raise ValueError(f"Model '{model}' not found in GOOGLE_MODELS_DICT."
                         f" Currently the model is not supported.")

    try:
        if model in DIFFERENT_TOP_K_MODELS:
            generation_config = {"temperature": temperature,
                             "top_p": 0.95,
                             "top_k": 40,
                             "max_output_tokens": max_tokens,
                             "response_mime_type": "text/plain",
                             }
        else:
            generation_config = {"temperature": temperature,
                             "top_p": 0.95,
                             "top_k": 64,
                             "max_output_tokens": max_tokens,
                             "response_mime_type": "text/plain",
                             }
        
        if system_prompt:
            model_google = genai.GenerativeModel(
                model_name=GOOGLE_MODELS_DICT[model],
                safety_settings=SAFETY_SETTINGS,
                generation_config=generation_config,
                system_instruction=system_prompt,
            )
        else:
            model_google = genai.GenerativeModel(
                model_name=GOOGLE_MODELS_DICT[model],
                safety_settings=SAFETY_SETTINGS,
                generation_config=generation_config,
            )
        
        chat_session = model_google.start_chat(history=[])

        response = chat_session.send_message(prompt)

        return response.text
    except Exception as e:
        logger.exception(f"Failed to get response from model '{model}'")
        raise RuntimeError(f"Failed to get response from model '{model}': {e}")
    

def get_response_for_multimodal_prompts(model: str,
                                        files: list,
                                        prompt: str,
                                        system_prompt: str = None,
                                        temperature: float = 0,
                                        max_tokens: int = 4096) -> str:
    """Get output from google models for multimodal prompts

    Args:
        model (str): The name of the model to use.
        files (list): The list of files that have been uploaded to google drive.
        prompt (str): The prompt to send to the model.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to None.
    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model is not found.
        RuntimeError: If the API call fails.
    """

    if model not in GOOGLE_MODELS_DICT:
        logger.error(f"Model '{model}' not found in GOOGLE_MODELS_DICT.")
        raise ValueError(f"Model '{model}' not found in GOOGLE_MODELS_DICT."
                         f" Currently the model is not supported.")

    try:
        if model in DIFFERENT_TOP_K_MODELS:
            generation_config = {"temperature": temperature,
                             "top_p": 0.95,
                             "top_k": 40,
                             "max_output_tokens": max_tokens,
                             "response_mime_type": "text/plain",
                             }
        else:
            generation_config = {"temperature": temperature,
                             "top_p": 0.95,
                             "top_k": 64,
                             "max_output_tokens": max_tokens,
                             "response_mime_type": "text/plain",
                             }
        
        if system_prompt:
            model_google = genai.GenerativeModel(
                model_name=GOOGLE_MODELS_DICT[model],
                safety_settings=SAFETY_SETTINGS,
                generation_config=generation_config,
                system_instruction=system_prompt,
            )
        else:
            model_google = genai.GenerativeModel(
                model_name=GOOGLE_MODELS_DICT[model],
                safety_settings=SAFETY_SETTINGS,
                generation_config=generation_config,
            )

        history = [{"role": "user", "parts": files,},]

        chat_session = model_google.start_chat(history=history)

        response = chat_session.send_message(prompt)

        return response.text
    except Exception as e:
        logger.exception(f"Failed to get response from model '{model}'")
        raise RuntimeError(f"Failed to get response from model '{model}': {e}")
    
