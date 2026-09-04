from rag.chunking import RecursiveChunker
from rag.models import Document


def test_recursive_chunker_creates_chunks():
    document = Document(
        id="doc-1",
        content="abcdefghij",
    )

    chunker = RecursiveChunker(
        chunk_size=4,
        overlap=1,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 3

    assert chunks[0].content == "abcd"
    assert chunks[1].content == "defg"
    assert chunks[2].content == "ghij"

    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2


def test_empty_document_returns_no_chunks():
    document = Document(
        id="doc-empty",
        content="",
    )

    chunker = RecursiveChunker()

    assert chunker.chunk(document) == []
