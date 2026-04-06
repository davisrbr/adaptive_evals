def count_words(text):
    """Count occurrences of each word in text (case-insensitive)."""
    words = text.split()
    counts = {}
    for word in words:
        # BUG: doesn't lowercase, so "Hello" and "hello" are counted separately
        counts[word] = counts.get(word, 0) + 1
    return counts
