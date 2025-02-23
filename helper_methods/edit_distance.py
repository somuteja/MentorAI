"""Contains method for calculating edit distance between two strings."""

import Levenshtein

def calculate_edit_distance(str1: str, str2: str) -> int:
    """Calculate the edit distance between two strings.
    Args:
        str1 (str): The first string.
        str2 (str): The second string.
    Returns:
        int: The edit distance between the two strings.
    """
    return Levenshtein.distance(str1, str2)
