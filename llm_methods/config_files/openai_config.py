from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OPENAI_MODELS_DICT = {"gpt-4o": "gpt-4o-2024-08-06",
                      "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
                      "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
                      "chatgpt-4o-latest": "chatgpt-4o-latest"}

OPENAI_EMBEDDING_MODELS_DICT = {"text-embedding-3-small": "text-embedding-3-small",
                                "text-embedding-3-large": "text-embedding-3-large"}
