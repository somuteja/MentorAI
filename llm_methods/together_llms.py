import logging
from llm_methods.config_files.together_config import (TOGETHER_CLIENT, TOGETHER_MODELS_URL_DICT, RERANK_MODELS_DICT)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_response_from_together_models(model: str,
                                      prompt: str,
                                      temperature: float = 0,
                                      max_tokens: int = 4096) -> str:
    """Get output from together models

    Args:
        model (str): The name of the model to use.
        prompt (str): The prompt to send to the model.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model is not found or the response structure is invalid.
        RuntimeError: If the API call fails.
    """

    if model not in TOGETHER_MODELS_URL_DICT:
        logger.error(f"Model '{model}' not found in TOGETHER_MODELS_URL_DICT.")
        raise ValueError(f"Model '{model}' not found in TOGETHER_MODELS_URL_DICT."
                         f" Currently the model is not supported.")
                         

    try:
        response = TOGETHER_CLIENT.chat.completions.create(
            model=TOGETHER_MODELS_URL_DICT[model],
            messages=[{"role": "user", "content": prompt, }],
            temperature=temperature,
            max_tokens=max_tokens)

        if not response.choices or not response.choices[0].message:
            logger.error("Invalid response structure received from the model")
            raise ValueError("Invalid response structure received from the model")

        return response.choices[0].message.content
    except Exception as e:
        logger.exception(f"Failed to get response from model '{model}'")
        raise RuntimeError(f"Failed to get response from model '{model}': {e}")
     

def get_rerank_from_together_models(model: str,
                                    query: str,
                                    documents: list[str],
                                    top_n: int,
                                    minimum_relevance_score: float = 0):
    """
    Rerank documents using Together AI models based on a given query.

    Args:
        model (str): The name of the reranking model to use.
        query (str): The query to use for reranking documents.
        documents (list[str]): A list of documents to be reranked.
        top_n (int): The number of top-ranked documents to return.
        minimum_relevance_score (float, optional): The minimum relevance score for a document to be included in the results. Defaults to 0.

    Raises:
        ValueError: If the specified model is not found in RERANK_MODELS_DICT.
        RuntimeError: If the API call to the reranking model fails.
    """
    try:
        if model not in RERANK_MODELS_DICT:
            logger.error(f"Model '{model}' not found in RERANK_MODELS_DICT.")
            raise ValueError(f"Model '{model}' not found in RERANK_MODELS_DICT."
                             f" Currently the model is not supported.")
        
        if not documents:
            logger.error(f"There are no documents to re-rank.")
            raise ValueError(f"documents cannot be empty")

        response = TOGETHER_CLIENT.rerank.create(
            model=RERANK_MODELS_DICT[model],
            query=query,
            documents=documents,
            top_n=top_n
        )

        return [result for result in response.results if result.relevance_score > minimum_relevance_score]
    except Exception as e:
        logger.exception(f"Failed to get rerank from model '{model}'")
        raise RuntimeError(f"Failed to get rerank from model '{model}': {e}")
