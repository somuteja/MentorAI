from together import Together
from dotenv import load_dotenv
import os


load_dotenv()

TOGETHER_CLIENT = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

TOGETHER_MODELS_URL_DICT = {"llama3-70b-chat": "meta-llama/Llama-3-70b-chat-hf",
                            "qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct",
                            "qwen15-110b-chat": "Qwen/Qwen1.5-110B-Chat",
                            "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"}

RERANK_MODELS_DICT = {"Llama-Rank-V1": "Salesforce/Llama-Rank-V1"}
