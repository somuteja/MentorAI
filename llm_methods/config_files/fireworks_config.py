from fireworks.client import Fireworks
from dotenv import load_dotenv
import os


load_dotenv()

FIREWORKS_CLIENT = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

FIREWORKS_MODELS_URL_DICT = {"llama3-70b-instruct": "accounts/fireworks/models/llama-v3-70b-instruct",
                             "qwen2-72b-instruct": "accounts/fireworks/models/qwen2-72b-instruct"}

