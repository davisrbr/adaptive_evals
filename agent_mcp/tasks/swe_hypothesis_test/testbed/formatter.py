import json


def format_report(data):
    """Format a data dict into a human-readable report string.

    Should produce lines like:
        Name: Alice
        Score: 95
        Tags: python, coding
    """
    lines = []
    for key, value in data.items():
        # BUG: lists should be comma-joined, but this just calls str()
        # which produces "['python', 'coding']" instead of "python, coding"
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return "\n".join(lines)
