import re
from bisect import bisect_right

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, HEADING_PATTERN, MAX_SECTION_CHARS

HEADING_RE = re.compile(HEADING_PATTERN)


def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def split_by_headings(pages):
    """Cut one PDF at every heading (e.g. 'Module 8.5:'), even across page breaks.
    Returns None when the PDF has no clear headings."""
    text, page_starts = "", []
    for p in pages:
        page_starts.append(len(text))   # where each page begins in the joined text
        text += p.page_content + "\n"

    cuts = [m.start() for m in HEADING_RE.finditer(text)]
    if len(cuts) < 3:
        return None  # not a document with headings: use normal splitting
    if cuts[0] != 0:
        cuts.insert(0, 0)  # the title part before the first heading
    cuts.append(len(text))

    source = pages[0].metadata.get("source")
    chunks = []
    for start, end in zip(cuts, cuts[1:]):
        piece = text[start:end].strip()
        if not piece:
            continue
        page = bisect_right(page_starts, start) - 1  # page where this section starts
        meta = {"source": source, "page": page}
        if len(piece) <= MAX_SECTION_CHARS:
            chunks.append(Document(page_content=piece, metadata=meta))
        else:  # a very long section: cut it further, keep the start page
            for part in _splitter().split_text(piece):
                chunks.append(Document(page_content=part, metadata=meta))
    return chunks


def ingest_files(paths):
    """Take a list of PDF paths and return a list of chunks."""
    chunks = []
    for path in paths:
        pages = PyPDFLoader(str(path)).load()
        if not pages:
            continue
        by_heading = split_by_headings(pages)
        if by_heading is None:
            by_heading = _splitter().split_documents(pages)
        chunks.extend(by_heading)
    return chunks