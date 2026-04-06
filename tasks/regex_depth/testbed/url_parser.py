import re


def extract_urls(text):
    """Extract all URLs from text, including those with query params."""
    # BUG: pattern doesn't match URLs with query strings (?key=value&key2=value2)
    # or fragment identifiers (#section)
    pattern = r'https?://[a-zA-Z0-9.-]+/[a-zA-Z0-9/_-]*'
    return re.findall(pattern, text)
