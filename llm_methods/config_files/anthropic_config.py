import anthropic
from dotenv import load_dotenv
import os


load_dotenv()

ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ANTHROPIC_MODELS_DICT = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet_old": "claude-3-5-sonnet-20240620",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-opus": "claude-3-opus-20240229"
}
