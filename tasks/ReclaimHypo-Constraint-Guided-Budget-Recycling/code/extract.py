def extract(raw):
    """Remove a single pair of wrapping inline-code backticks only."""
    text=raw.strip()
    if len(text)>2 and text[0]=='`' and text[-1]=='`' and '`' not in text[1:-1]:
        return text[1:-1].strip()
    return text
