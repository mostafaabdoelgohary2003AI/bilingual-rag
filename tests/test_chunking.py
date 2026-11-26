import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_text, split_sentences


def test_arabic_sentence_split():
    text = "ما هو المشروع؟ هو نظام كشف الحريق. تم تطويره في الجامعة."
    sentences = split_sentences(text)
    assert sentences == ["ما هو المشروع؟", "هو نظام كشف الحريق.", "تم تطويره في الجامعة."]


def test_chunk_size_respected_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, doc_id="t", chunk_size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 220 for c in chunks)  # small slack for tail seeding
    assert chunks[0].doc_id == "t"
    # overlap: consecutive chunks share some trailing/leading words
    shared = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert shared


def test_metadata_propagates():
    chunks = chunk_text("Hello world. " * 50, doc_id="doc1", chunk_size=100, overlap=20,
                        metadata={"lang": "en"})
    assert all(c.metadata["lang"] == "en" and c.metadata["doc_id"] == "doc1" for c in chunks)


if __name__ == "__main__":
    test_arabic_sentence_split()
    test_chunk_size_respected_and_overlap()
    test_metadata_propagates()
    print("chunking tests passed")
