from pathlib import Path

import chromadb

from app import config
from app.ingestion.loader import load_document
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts

COLLECTION_NAME = "case_law"

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
    return _client


def get_collection():
    return get_client().get_or_create_collection(COLLECTION_NAME)


def ingest_case(file_path: Path, case_name: str) -> int:
    """Chunks + embeds a case file and adds it to the store. Returns the
    number of NEW chunks added (0 if this file was already ingested - the
    cache-hit case)."""
    collection = get_collection()

    already_present = collection.get(where={"source_file": str(file_path)}, limit=1)
    if already_present["ids"]:
        return 0

    doc = load_document(file_path)
    chunks = chunk_text(doc.text)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    ids = [f"{file_path.stem}__{i}" for i in range(len(chunks))]
    metadatas = [
        {"source_file": str(file_path), "case_name": case_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def query_case(file_path: Path, query_text: str, top_k: int = 3) -> list[dict]:
    """Semantic search restricted to chunks from one specific case file -
    we're not searching the whole corpus, just checking what THIS case
    actually says relative to the claim being made about it."""
    collection = get_collection()
    query_embedding = embed_texts([query_text])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source_file": str(file_path)},
    )

    hits = []
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        hits.append({"text": doc, "similarity": 1 - distance})  # cosine distance -> similarity
    return hits


if __name__ == "__main__":
    from app.ingestion.citations import dedupe_citations, extract_citations, get_citation_context
    from app.rag.corpus_index import build_registry, find_case_in_corpus

    brief_path = config.RAW_CORPUS_DIR / "White and Case.pdf"
    brief_doc = load_document(brief_path)
    citations = dedupe_citations(extract_citations(brief_doc.text))
    registry = build_registry()

    for c in citations:
        match = find_case_in_corpus(c.case_name, registry)
        if match.status != "MATCHED":
            print(f"\n[SKIP - not in corpus] {c.case_name} ({c.citation})")
            continue

        added = ingest_case(match.matched_file, c.case_name)
        cache_note = f"embedded {added} new chunks" if added else "already cached"
        print(f"\n=== {c.case_name} ({c.citation}) - {cache_note} ===")

        claimed_context = get_citation_context(brief_doc.text, c.char_offset)
        print(f"Brief claims: \"{claimed_context}\"")

        top_hits = query_case(match.matched_file, claimed_context, top_k=2)
        for i, hit in enumerate(top_hits, 1):
            print(f"  Retrieved passage {i} (similarity {hit['similarity']:.3f}):")
            print(f"    {hit['text'][:220]}...")
