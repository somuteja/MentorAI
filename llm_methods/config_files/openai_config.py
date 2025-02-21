import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()


genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

SAFETY_SETTINGS = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

GOOGLE_MODELS_DICT =  {"gemini-1.5-flash": "gemini-1.5-flash",
                       "gemini-1.5-pro": "gemini-1.5-pro",
                       "gemini-1.5-pro-experimental": "gemini-1.5-pro-exp-0827",
                       "gemini-1.5-flash-experimental": "gemini-1.5-flash-exp-0827",
                       "gemini-1.5-pro-002": "gemini-1.5-pro-002",
                       "gemini-1.5-flash-002": "gemini-1.5-flash-002"}

DIFFERENT_TOP_K_MODELS = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]
