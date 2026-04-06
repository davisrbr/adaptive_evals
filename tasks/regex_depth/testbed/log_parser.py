import re


def parse_log_levels(text):
    """Extract log entries with their levels from log text.

    Each log line looks like: [LEVEL] message
    Returns dict mapping level -> list of messages.
    """
    # BUG: pattern is case-sensitive and uses wrong group capture,
    # missing entries like [warning] or [Warning]
    pattern = r'^\[(INFO|ERROR|WARNING|DEBUG)\]\s*(.+)$'
    result = {}
    for match in re.finditer(pattern, text, re.MULTILINE):
        level = match.group(1)
        msg = match.group(2)
        result.setdefault(level, []).append(msg)
    return result
