# bilingual-rag

Retrieval-augmented generation over Arabic and English documents, with
citation-grounded answers and a built-in retrieval evaluation loop.
Runs on a laptop CPU with zero downloads and no API key; plugs into any
OpenAI-compatible LLM endpoint (OpenAI, NVIDIA NIM, Ollama) when you want
generated answers.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Why

Most RAG demos are English-only, need a paid API key to do anything, and
never measure whether retrieval actually works. This project is the
opposite: bilingual from the first commit, fully runnable offline, and it
ships with a labelled evaluation set so every change is measured, not
vibes.

## Features

- **Arabic-aware chunking** — sentence splitting handles both `?` and `؟`;
  recursive paragraph → sentence packing never cuts a sentence in half.
- **Two embedding backends** — TF-IDF (default, CPU-only, no downloads) or
  dense multilingual embeddings via `sentence-transformers` (optional).
- **Cited answers** — every answer carries its source passages with scores;
  zero-overlap passages are dropped instead of padded in.
- **Two generators** — an extractive offline fallback that quotes evidence,
  and a minimal stdlib-only client for any OpenAI-compatible chat API.
- **Honest evaluation** — hit-rate@k and MRR@k over labelled QA pairs;
  change one thing, re-run, compare.

## Architecture

```text
documents ──► chunking ──► embeddings ──► vector store
                                              │
question ──► embed query ──► top-k retrieval ─┤
                                              ▼
                              prompt with numbered passages
                                              ▼
                              LLM (or extractive fallback)
                                              ▼
                              answer + citations [1] [2] ...
```

## Quickstart

```bash
pip install -r requirements.txt
python demo.py          # fully offline: indexes sample_docs/, answers, evaluates
```

With a real LLM (any OpenAI-compatible endpoint):

```bash
export LLM_BASE_URL="https://integrate.api.nvidia.com/v1"
export LLM_API_KEY="nvapi-..."
export LLM_MODEL="meta/llama-3.1-8b-instruct"
python demo.py --llm
```

Use it as a library:

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_paths(["my-docs/contract.md", "my-docs/ملخص.md"])
result = pipeline.query("ما هي شروط الإلغاء؟", k=4)
print(result["answer"])
print(result["citations"])
```

## What the demo prints (real output)

```text
Indexed 6 chunks from 4 documents.

Q: كم عدد جلسات الشحن التي حللها مشروع وارسو؟
[1] ... حوالي 114 ألف جلسة شحن للسيارات الكهربائية ...  (source: warsaw-ev-analysis-ar.md, score 0.381)

Retrieval evaluation over 4 labelled questions:
  hit-rate@3: 1.0
  MRR@3:       1.0
```

The labelled set is tiny on purpose — it is a template. Add your own
questions to `QA_SET` in `demo.py` (or load them from JSON) as your corpus
grows.

## Project structure

```text
src/
  chunking.py      # Arabic-aware recursive chunking
  embeddings.py    # TF-IDF (default) + optional dense backend
  vectorstore.py   # in-memory cosine store with save/load
  retriever.py     # top-k retrieval, zero-evidence filtering
  generator.py     # extractive fallback + OpenAI-compatible client
  pipeline.py      # RAGPipeline: ingest -> query -> answer + citations
  evaluation.py    # hit-rate@k, MRR@k over labelled QA pairs
tests/             # pytest, no network needed
sample_docs/       # demo corpus (English + Arabic)
```

## Roadmap

- [ ] Hybrid retrieval (dense + sparse with reciprocal-rank fusion)
- [ ] Cross-encoder reranking stage
- [ ] Streamlit UI with per-answer citation highlighting
- [ ] Answer-quality rubric (see my `llm-eval-kit` project)

## Notes

- Developed with AI pair-programming assistance; architecture, evaluation
  design and testing decisions are my own.
- License: MIT.
