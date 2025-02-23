import bs4
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_html_output(html_output: str, tag_to_extract: str) -> str:
    """Parse the HTML output and return the text.

    Args:
        html_output (str): The HTML output.
        tag_to_extract (str): The tag to extract.
    Returns:
        str: The text.
    """
    try:
        soup = bs4.BeautifulSoup(html_output, 'html.parser')
        tag = soup.find(tag_to_extract)
        return tag.text.strip()
    except Exception as e:
        logger.error(f"Error parsing HTML output: {e}. Could not parse {tag_to_extract} from {html_output}.")
        return html_output
    

    
def parse_html_output_with_multiple_tags(html_output: str,
                                        main_tag: str,
                                        list_of_tags_to_extract: list[str]) -> list[dict[str, str]]:
    """General method for parsing html outputs with multiple tags.

    Args:
        html_output (str): The HTML output.
        main_tag (str): The main tag to extract.
        list_of_tags_to_extract (list[str]): The list of tags to extract.
    Returns:
        list[dict[str, str]]: A list of dictionaries with the tag as the key and the text as the value.
    """
    try:
        soup = bs4.BeautifulSoup(html_output, 'html.parser')
        list_of_dicts = []
        for main_instance in soup.find_all(main_tag):
            dict_to_add = {}
            for tag in list_of_tags_to_extract:
                dict_to_add[tag] = main_instance.find(tag).text.strip() if main_instance.find(tag) else ""
            list_of_dicts.append(dict_to_add)
        return list_of_dicts
    except Exception as e:
        logger.error(f"Error parsing HTML output: {e}.")
        return []
    
def parse_html_outputs(html_output: str, tag_to_extract: str) -> list[str]:
    """Parse the HTML output and return the list of texts.

    Args:
        html_output (str): The HTML output.
        tag_to_extract (str): The tag to extract.
    Returns:
        list[str]: The list of texts.
    """
    try:
        soup = bs4.BeautifulSoup(html_output, 'html.parser')
        list_of_texts = []
        for tag in soup.find_all(tag_to_extract):
            list_of_texts.append(tag.text.strip())
        return list_of_texts
    except Exception as e:
        logger.error(f"Error parsing HTML output: {e}.")
        return []
    
