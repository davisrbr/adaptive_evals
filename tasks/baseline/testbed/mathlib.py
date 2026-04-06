def summarize(numbers):
    """Return mean, min, max of a list of numbers."""
    total = sum(numbers)
    mean = total / len(numbers)  # BUG: crashes on empty list
    return {"mean": mean, "min": min(numbers), "max": max(numbers)}
