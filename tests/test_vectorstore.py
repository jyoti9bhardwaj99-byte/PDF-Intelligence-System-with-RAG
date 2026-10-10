from langchain_core.documents import Document

from app.vectorstore import _chunk_id


def doc(source, page, text):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_same_chunk_from_a_different_folder_gets_the_same_id():
    # Regression test: IDs used the full path, so re-uploading doubled the index
    a = doc("app/doc_files/syllabus.pdf", 5, "Module 8.5 text")
    b = doc("data/uploads/syllabus.pdf", 5, "Module 8.5 text")
    assert _chunk_id(a) == _chunk_id(b)


def test_different_page_or_text_gets_a_different_id():
    base = doc("syllabus.pdf", 5, "Module 8.5 text")
    assert _chunk_id(base) != _chunk_id(doc("syllabus.pdf", 6, "Module 8.5 text"))
    assert _chunk_id(base) != _chunk_id(doc("syllabus.pdf", 5, "other text"))