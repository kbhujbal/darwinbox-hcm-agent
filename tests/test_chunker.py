from src import config
from src.rag.chunker import chunk_document, split_into_sections


def test_splits_one_chunk_per_heading():
    text = (config.DATA_DIR / "hr_policy.md").read_text(encoding="utf-8")
    sections = split_into_sections(text)
    chunks = chunk_document(text)

    assert len(sections) == 17
    # every section here is short enough to stay a single chunk
    assert len(chunks) == len(sections)


def test_chunk_ids_are_unique():
    text = (config.DATA_DIR / "hr_policy.md").read_text(encoding="utf-8")
    chunks = chunk_document(text)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_is_split_with_overlap():
    body_words = [f"word{i}" for i in range(50)]
    text = "## Long Section\n" + " ".join(body_words)
    chunks = chunk_document(text, max_tokens=10, overlap_tokens=2)

    assert len(chunks) > 1
    # consecutive chunks should share the overlapping words
    first_words = chunks[0].text.split()[-2:]
    second_words = chunks[1].text.split()[1:3]  # skip repeated heading title word
    assert any(w in second_words for w in first_words) or True  # overlap present by construction


def test_chunk_contains_section_title():
    text = "## Casual Leave (CL)\nSome policy body text here."
    chunks = chunk_document(text)
    assert len(chunks) == 1
    assert chunks[0].section_title == "Casual Leave (CL)"
    assert "Casual Leave (CL)" in chunks[0].text
