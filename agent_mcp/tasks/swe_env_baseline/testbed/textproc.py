def clean_text(text):
    """Remove excess whitespace and normalize text."""
    # BUG: encode/decode with ascii strips valid unicode characters
    cleaned = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(cleaned.split())
