from langchain_core.documents import Document

from app.ingestion import split_by_headings


def make_pages(texts):
    """Pretend PDF pages: one Document per text."""
    return [
        Document(page_content=t, metadata={"source": "fake.pdf", "page": i})
        for i, t in enumerate(texts)
    ]


def test_sections_are_cut_at_headings_with_start_page():
    pages = make_pages([
        "Course title\nModule 1: Python (4 weeks)\nTopics: loops",
        "Module 2: Pandas (4 weeks)\nTopics: dataframes",
        "Module 3: SQL (3 weeks)\nTopics: joins",
    ])
    chunks = split_by_headings(pages)

    assert len(chunks) == 4  # title part + 3 modules
    assert chunks[1].page_content.startswith("Module 1:")
    assert [c.metadata["page"] for c in chunks[1:]] == [0, 1, 2]


def test_section_crossing_a_page_break_stays_in_one_chunk():
    # Regression test: Module 2's mini project used to be cut off on the next page
    pages = make_pages([
        "Module 1: Python\nTopics: loops\nModule 2: PySpark (2 weeks)\nTopics: spark",
        "Mini Project: NYC Taxi\nMock Interview: Big Data\nModule 3: NLP\nTopics: text",
        "Module 4: GenAI\nTopics: llm",
    ])
    chunks = split_by_headings(pages)
    pyspark = [c for c in chunks if c.page_content.startswith("Module 2:")][0]

    assert "NYC Taxi" in pyspark.page_content
    assert "Big Data" in pyspark.page_content
    assert pyspark.metadata["page"] == 0  # the page where the section starts


def test_document_without_headings_returns_none():
    pages = make_pages(["Just some plain text.", "More plain text without headings."])
    assert split_by_headings(pages) is None


def test_very_long_section_is_split_further():
    long_text = "Module 1: Big\n" + "lorem ipsum " * 400 + "\nModule 2: B\nx\nModule 3: C\ny"
    chunks = split_by_headings(make_pages([long_text]))

    assert len(chunks) > 3  # Module 1 became several chunks
    assert chunks[0].page_content.startswith("Module 1:")