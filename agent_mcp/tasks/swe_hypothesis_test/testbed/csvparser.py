def parse_csv(text):
    """Parse CSV text into list of dicts using first row as headers.

    Handles quoted fields containing commas.
    """
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    headers = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        # BUG: naive split on comma breaks quoted fields like '"New York, NY"'
        values = line.split(",")
        row = {}
        for i, h in enumerate(headers):
            row[h.strip()] = values[i].strip() if i < len(values) else ""
        rows.append(row)
    return rows
