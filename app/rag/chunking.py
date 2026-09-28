DEFAULT_CHUNK_SIZE_WORDS = 250
DEFAULT_OVERLAP_WORDS = 40


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) < 20:  # drop tiny trailing scraps, not useful for retrieval
            break
        chunks.append(" ".join(chunk_words))

    return chunks
