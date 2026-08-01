"""Load and chunk source documents for the RAG and summarization tasks.

Drop your own English source documents (plain text or Markdown) into
`documents/corpus/` and use `load_document()` / `load_corpus()`. Chunking
is paragraph-aware: it packs whole paragraphs up to `max_words_per_chunk`
words, and only hard-splits a paragraph if it alone exceeds that budget.
Chunk numbering is stable (0-indexed, source order) since generators and
the RAG postprocessing step key off `chunk_id`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config.settings import DOCUMENTS_DIR


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source_path: str
    chunks: list[Chunk]

    def chunk_text(self, chunk_id: int) -> str:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c.text
        raise KeyError(f"No chunk_id={chunk_id} in document {self.doc_id!r}")

    def as_numbered_block(self) -> str:
        """Text block used in prompts: '[chunk 0]\\n...\\n\\n[chunk 1]\\n...'"""
        return "\n\n".join(f"[chunk {c.chunk_id}]\n{c.text}" for c in self.chunks)


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]
    sentences = _SENTENCE_SPLIT.split(paragraph)
    if len(sentences) <= 1:
        # No sentence boundaries either -- hard-split by word count.
        return [
            " ".join(words[i : i + max_words])
            for i in range(0, len(words), max_words)
        ]
    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_words + n > max_words:
            parts.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += n
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(raw_text: str, *, max_words_per_chunk: int = 180) -> list[Chunk]:
    raw_text = raw_text.strip()
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(raw_text) if p.strip()]

    normalized: list[str] = []
    for p in paragraphs:
        normalized.extend(_split_long_paragraph(p, max_words_per_chunk))

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0
    for para in normalized:
        n = len(para.split())
        if buffer and buffer_words + n > max_words_per_chunk:
            chunks.append(Chunk(chunk_id=len(chunks), text="\n".join(buffer)))
            buffer, buffer_words = [], 0
        buffer.append(para)
        buffer_words += n
    if buffer:
        chunks.append(Chunk(chunk_id=len(chunks), text="\n".join(buffer)))
    return chunks


def load_document(
    path: str | Path,
    *,
    doc_id: str | None = None,
    title: str | None = None,
    max_words_per_chunk: int = 180,
) -> Document:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    first_line = raw_text.strip().splitlines()[0].lstrip("# ").strip() if raw_text.strip() else path.stem
    return Document(
        doc_id=doc_id or path.stem,
        title=title or first_line,
        source_path=str(path),
        chunks=chunk_text(raw_text, max_words_per_chunk=max_words_per_chunk),
    )


def load_corpus(corpus_dir: str | Path | None = None) -> list[Document]:
    corpus_dir = Path(corpus_dir) if corpus_dir else (DOCUMENTS_DIR / "corpus")
    if not corpus_dir.exists():
        return []
    docs = []
    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() in (".txt", ".md") and path.is_file():
            docs.append(load_document(path))
    return docs


_SAMPLE_DOC_PATHS = [
    "documents/sample/malaria_prevention.md",
    "documents/sample/farming_crop_rotation.md",
]


def load_default_documents() -> list[Document]:
    """Sample docs (always available) + anything dropped in documents/corpus/.

    This is the single registry both generators and postprocessing use, so
    a `doc_id` recorded during generation can always be resolved back to
    its real chunk text later, regardless of which module needs it.
    """
    sample_docs = [load_document(p) for p in _SAMPLE_DOC_PATHS]
    return sample_docs + load_corpus()


def documents_by_id(docs: list[Document]) -> dict[str, Document]:
    by_id = {d.doc_id: d for d in docs}
    if len(by_id) != len(docs):
        raise ValueError("Duplicate doc_id among documents -- rename files in documents/corpus/.")
    return by_id
