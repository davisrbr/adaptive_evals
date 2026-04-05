import re


def find_phone_numbers(text):
    """Find all US phone numbers in text. Formats: (123) 456-7890 or 123-456-7890."""
    # BUG: only matches format with parentheses, misses 123-456-7890
    pattern = r'\(\d{3}\)\s*\d{3}-\d{4}'
    return re.findall(pattern, text)
