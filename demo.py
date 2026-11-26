"""End-to-end demo: index the sample docs and answer questions offline.

Usage:
    python demo.py                 # extractive answers, no API key needed
    LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... python demo.py --llm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import evaluate_retrieval
from src.pipeline import RAGPipeline

QUESTIONS = [
    "What was Omar's role in the FireGuard project?",
    "كم عدد جلسات الشحن التي حللها مشروع وارسو؟",
    "What problem does retrieval-augmented generation solve?",
]

QA_SET = [
    {"question": "What was Omar's role in FireGuard?", "relevant_doc_id": "fireguard-overview.md"},
    {"question": "How many charging sessions were analyzed in Warsaw?", "relevant_doc_id": "warsaw-ev-analysis.md"},
    {"question": "Why does RAG reduce hallucination?", "relevant_doc_id": "rag-concepts.md"},
    {"question": "كم عدد جلسات الشحن في مشروع وارسو؟", "relevant_doc_id": "warsaw-ev-analysis-ar.md"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="use an OpenAI-compatible LLM endpoint")
    args = parser.parse_args()

    generator = None
    if args.llm:
        from src.generator import OpenAICompatibleClient

        generator = OpenAICompatibleClient(model=os.environ.get("LLM_MODEL", "meta/llama-3.1-8b-instruct"))

    pipeline = RAGPipeline(generator=generator)
    docs = sorted(Path(__file__).parent.glob("sample_docs/*.md"))
    added = pipeline.ingest_paths(docs)
    print(f"Indexed {added} chunks from {len(docs)} documents.\n")

    for question in QUESTIONS:
        result = pipeline.query(question, k=3)
        print(f"Q: {result['question']}")
        print(result["answer"])
        print("-" * 72)

    report = evaluate_retrieval(pipeline, QA_SET, k=3)
    print(f"Retrieval evaluation over {report['questions']} labelled questions:")
    print(f"  hit-rate@{report['k']}: {report['hit_rate']}")
    print(f"  MRR@{report['k']}:       {report['mrr']}")


if __name__ == "__main__":
    main()
