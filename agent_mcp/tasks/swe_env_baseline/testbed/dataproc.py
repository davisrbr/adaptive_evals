def merge_records(records):
    """Merge list of dicts, combining values for duplicate keys into lists."""
    merged = {}
    for record in records:
        for key, value in record.items():
            # BUG: overwrites instead of combining
            merged[key] = value
    return merged
