import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_text
from src.embeddings import TfidfEmbedder
from src.retriever import Retriever
from src.vectorstore import NumpyVectorStore

DOCS = {
    "fire.md": "FireGuard detects fire and smoke with a YOLO model and alerts a mobile app.",
    "ev.md": "The Warsaw project analysed 114,000 electric vehicle charging sessions.",
    "rag.md": "Retrieval augmented generation grounds answers in retrieved documents.",
    "ev-ar.md": "حلل مشروع وارسو حوالي 114 ألف جلسة شحن للسيارات الكهربائية في محطات الشحن العامة.",
}


def _build():
    chunks = []
    for doc_id, text in DOCS.items():
        chunks.extend(chunk_text(text, doc_id=doc_id, chunk_size=400, overlap=50))
    embedder = TfidfEmbedder().fit([c.text for c in chunks])
    store = NumpyVectorStore()
    store.add(embedder.transform([c.text for c in chunks]), chunks)
    return Retriever(embedder, store)


def test_retrieves_correct_doc_english():
    retriever = _build()
    top = retriever.retrieve("How many charging sessions were analyzed?", k=1)[0]
    assert top.chunk.doc_id == "ev.md"
    assert top.score > 0


def test_retrieves_correct_doc_arabic():
    retriever = _build()
    top = retriever.retrieve("كم عدد جلسات الشحن؟", k=1)[0]
    assert top.chunk.doc_id == "ev-ar.md"


def test_store_roundtrip(tmp_path=None):
    import tempfile

    retriever = _build()
    with tempfile.TemporaryDirectory() as tmp:
        retriever.store.save(tmp)
        loaded = NumpyVectorStore.load(tmp)
        assert len(loaded) == len(retriever.store)


if __name__ == "__main__":
    test_retrieves_correct_doc_english()
    test_retrieves_correct_doc_arabic()
    test_store_roundtrip()
    print("retrieval tests passed")
