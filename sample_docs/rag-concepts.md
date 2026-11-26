# Retrieval-augmented generation, briefly

Large language models answer from what they memorised during training, so
they can state false facts confidently — a failure mode called hallucination.
Retrieval-augmented generation (RAG) reduces hallucination by grounding every
answer in documents retrieved at question time: the system fetches the most
relevant passages from a document collection and asks the model to answer
using only those passages, citing them.

A RAG pipeline has five stages: loading documents, chunking them into
passages, embedding passages into vectors, retrieving the top-k passages for
a question, and generating an answer conditioned on the retrieved passages.
Retrieval quality is measured with labelled questions using hit rate and
mean reciprocal rank; answer quality is measured separately, often with an
LLM-as-judge rubric.
