from llama_parse import LlamaParse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
DEFAULT_NUMBER_OF_WORKERS = 8

def get_markdown_text(file_path: str, number_of_workers: int = DEFAULT_NUMBER_OF_WORKERS) -> str:
    """
    Parse the markdown text from the given file path.
    Args:
        file_path (str): The path to the file to parse.
        number_of_workers (int): The number of workers to use for the parsing.
    Returns:
        str: The markdown text from the given file path.
    """
    try:
        parser = LlamaParse(
            result_type="markdown",
            api_key=LLAMA_PARSE_API_KEY,
            verbose=False,
            num_workers=number_of_workers
        )
        result = parser.load_data(file_path)
        return result[0].text
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        return ""

def get_markdown_texts_in_batches(file_paths: list[str], number_of_workers: int = DEFAULT_NUMBER_OF_WORKERS) -> list[str]:
    """
    Parse the markdown text from the given file paths. We are using synchronous batch parsing here.
    Args:
        file_paths (list[str]): The paths to the files to parse.
        number_of_workers (int): The number of workers to use for the parsing.
    Returns:
        list[str]: The markdown texts from the given file paths.
    """
    try:
        parser = LlamaParse(
            result_type="markdown",
            api_key=LLAMA_PARSE_API_KEY,
            verbose=False,
            num_workers=number_of_workers
        )
        results = parser.load_data(file_paths)
        return [result.text for result in results]
    except Exception as e:
        logger.error(f"Error parsing files {file_paths}: {e}")
        return []
