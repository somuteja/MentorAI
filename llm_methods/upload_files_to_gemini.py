import logging
import google.generativeai as genai
import time
import tempfile
from dotenv import load_dotenv
import requests
import os

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_image_to_gemini(image_path: str):
    """Upload image to gemini

    Args:
        image_path (str): path to the image
    Returns:
        file: file object
    """
    try:
        file = genai.upload_file(image_path, mime_type="image/jpeg")
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logger.error(f"Error uploading image to gemini: {e}")
        return None

def upload_video_to_gemini(video_path: str):
    """Upload video to gemini

    Args:
        video_path (str): path to the video
    Returns:
        file: file object
    """
    try:
        file = genai.upload_file(video_path, mime_type="video/mp4")
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logger.error(f"Error uploading video to gemini: {e}")
        return None

def upload_audio_to_gemini(audio_path: str):
    """Upload audio to gemini

    Args:
        audio_path (str): path to the audio
    Returns:
        file: file object
    """
    try:
        file = genai.upload_file(audio_path, mime_type="audio/wav")
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logger.error(f"Error uploading audio to gemini: {e}")
        return None


def wait_for_file_to_be_active(file, wait_time=10):
  """Waits for the given file to be active.

  Some file uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  Args:
    file: file object
    wait_time: time to wait between checks (default is 10 seconds)
  """
  try:
    file = genai.get_file(file.name)
    while file.state.name == "PROCESSING":
      time.sleep(wait_time)
      file = genai.get_file(file.name)
    
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
    
    logger.info(f"File {file.name} is active")
  except Exception as e:
    logger.error(f"Error waiting for file to be active: {e}")
    return None
  
def upload_file_to_gemini(file_path: str, mime_type: str = None):
  """Upload file to gemini

  Args:
    file_path (str): path to the file
    mime_type (str): mime type of the file
  Returns:
    file: file object
  """
  try:
    file = genai.upload_file(file_path, mime_type=mime_type)
    logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file
  except Exception as e:
    logger.error(f"Error uploading file to gemini: {e}")
    return None

def upload_screenshot_to_gemini_from_supabase(screenshot_url: str):
  """Upload screenshot to gemini from supabase

  Args:
    screenshot_url (str): url of the screenshot
  Returns:
    file: file object
  """
  try:
    response = requests.get(screenshot_url)
    response.raise_for_status()
    image_data = response.content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_data)
        temp_file_path = temp_file.name

    file = upload_image_to_gemini(temp_file_path)
    
    wait_for_file_to_be_active(file)
    
    os.remove(temp_file_path)
    
    logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file
  except Exception as e:
    logger.error(f"Error uploading screenshot to gemini: {e}")
    return None
