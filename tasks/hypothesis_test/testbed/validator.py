import re


def validate_email(email):
    """Validate an email address. Returns True if valid, False otherwise."""
    # BUG: pattern rejects valid emails with + in local part (e.g. user+tag@example.com)
    # and with subdomains (e.g. user@mail.example.com)
    pattern = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
