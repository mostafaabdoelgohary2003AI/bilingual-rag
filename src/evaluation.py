"""Retrieval evaluation: hit rate and mean reciprocal rank (MRR).

Given a small labelled set of (question, relevant_doc_id) pairs, measure
whether the retriever surfaces a passage from the right document, and how
high it ranks it. This is the minimal honest evaluation loop for a RAG
system: change one thing, re-run, compare.
"""

from __future__ import annotations

from .pipeline import RAGPipeline


def evaluate_retrieval(pipeline: RAGPipeline, qa_pairs: list[dict], k: int = 4) -> dict:
    """Return hit-rate@k and MRR@k over labelled QA pairs.

    Each item in *qa_pairs* is {"question": str, "relevant_doc_id": str}.
    """
    if not qa_pairs:
        raise ValueError("qa_pairs must not be empty")
    hits = 0
    reciprocal_rank_sum = 0.0
    per_question = []
    for item in qa_pairs:
        passages = pipeline.retriever.retrieve(item["question"], k=k)
        rank = next(
            (i for i, p in enumerate(passages, start=1) if p.chunk.doc_id == item["relevant_doc_id"]),
            None,
        )
        hit = rank is not None
        hits += int(hit)
        reciprocal_rank_sum += (1.0 / rank) if hit else 0.0
        per_question.append({"question": item["question"], "rank": rank, "hit": hit})
    n = len(qa_pairs)
    return {
        "k": k,
        "questions": n,
        "hit_rate": round(hits / n, 4),
        "mrr": round(reciprocal_rank_sum / n, 4),
        "per_question": per_question,
    }
