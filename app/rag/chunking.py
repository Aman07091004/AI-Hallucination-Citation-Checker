"""
Splits a long document's text into overlapping chunks suitable for
embedding. We chunk by word count rather than characters or sentences
because it's simple, fast, and gives predictable chunk sizes regardless of
punctuation quirks from OCR'd text (OCR output sometimes has broken
sentence boundaries, which would make sentence-based chunking unreliable).

Overlap matters because a key sentence can otherwise get split across two
chunks and lose context on both sides - e.g. "the court held that..." at
the end of one chunk with the actual holding at the start of the next.
"""

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
