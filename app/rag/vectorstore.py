from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from app import config
from app.ingestion.loader import load_document
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts

COLLECTION_NAME = "case_law"

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        # anonymized_telemetry=False avoids a known noisy (but harmless)
        # compatibility warning between chromadb and its telemetry library -
        # not a functional issue, just console noise we don't need.
        _client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
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


def hybrid_query_case(
    file_path: Path, query_text: str, top_k: int = 4, embedding_weight: float = 0.6
) -> list[dict]:
    """Combines semantic (embedding) search with BM25 (keyword/lexical)
    search over one case's chunks, then blends the two scores.

    Why this matters: pure embedding search on real output showed a real
    weakness - for OBG v Allan, it surfaced passages about earlier judges'
    commentary instead of the actual holding the brief was citing, and
    scores across the board were mediocre (0.47-0.66). Embeddings are good
    at "similar meaning" but can miss passages that share exact legal
    terminology with the query but are phrased differently overall. BM25
    is the opposite: strong at exact/near-exact term overlap, weak at
    paraphrase. Combining them is standard practice in production RAG
    systems for exactly this reason - each covers the other's blind spot.
    """
    collection = get_collection()
    all_chunks = collection.get(where={"source_file": str(file_path)}, include=["documents"])
    documents = all_chunks["documents"]
    if not documents:
        return []

    # --- BM25 (lexical) scoring ---
    tokenized_corpus = [d.lower().split() for d in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_text.lower().split())
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    normalized_bm25 = [score / max_bm25 for score in bm25_scores]

    # --- Embedding (semantic) scoring, over the same full set of chunks ---
    query_embedding = embed_texts([query_text])[0]
    emb_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents),
        where={"source_file": str(file_path)},
    )
    doc_to_similarity = {
        doc: 1 - dist for doc, dist in zip(emb_results["documents"][0], emb_results["distances"][0])
    }

    # --- Blend and rank ---
    combined = []
    for doc, bm25_norm in zip(documents, normalized_bm25):
        semantic_score = doc_to_similarity.get(doc, 0.0)
        blended = embedding_weight * semantic_score + (1 - embedding_weight) * bm25_norm
        combined.append(
            {"text": doc, "blended_score": blended, "semantic_score": semantic_score, "bm25_score": bm25_norm}
        )

    combined.sort(key=lambda hit: hit["blended_score"], reverse=True)
    return combined[:top_k]


if __name__ == "__main__":
    from app.ingestion.citations import (
        dedupe_citations,
        extract_citations,
        get_citation_context,
        strip_self_reference,
    )
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
        query_text = strip_self_reference(claimed_context, c.case_name, c.citation)
        print(f"Brief claims: \"{claimed_context}\"")
        print(f"Query (self-reference stripped): \"{query_text}\"")

        top_hits = hybrid_query_case(match.matched_file, query_text, top_k=3)
        for i, hit in enumerate(top_hits, 1):
            print(
                f"  Passage {i} - blended {hit['blended_score']:.3f} "
                f"(semantic {hit['semantic_score']:.3f}, bm25 {hit['bm25_score']:.3f}):"
            )
            print(f"    {hit['text'][:220]}...")
