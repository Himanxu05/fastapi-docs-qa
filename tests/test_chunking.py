from docqa.chunking import chunk_by_heading, chunk_fixed, slugify
from docqa.sources import Page, page_url


def page(text):
    return Page(path="tutorial/body", url="https://x/tutorial/body/", title="Request Body", text=text)


def test_sections_get_heading_path_and_anchor():
    text = ("# Request Body\n\n" + "intro words " * 30 + "\n\n## Create your data model\n\n"
            + "model words " * 30)
    chunks = chunk_by_heading(page(text))
    assert [c.section for c in chunks] == ["Request Body", "Request Body > Create your data model"]
    assert chunks[1].url == "https://x/tutorial/body/#create-your-data-model"
    assert chunks[1].embed_text.startswith("Request Body > Create your data model\n\n")


def test_hash_inside_code_is_not_a_heading():
    text = "# Title\n\n" + "words " * 50 + "\n\n```python\n# just a comment\nx = 1\n```\n"
    chunks = chunk_by_heading(page(text))
    assert len(chunks) == 1 and "# just a comment" in chunks[0].text


def test_long_sections_split_but_code_blocks_stay_whole():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(40)) + "\n```"
    text = "# T\n\n" + "\n\n".join(["para " * 60] * 4) + "\n\n" + code
    chunks = chunk_by_heading(page(text), max_words=100)
    assert len(chunks) > 2
    for c in chunks:
        assert c.text.count("```") % 2 == 0  # never cut inside a code block


def test_tiny_sections_are_merged():
    text = "# T\n\n" + "words " * 50 + "\n\n## Tiny\n\nshort bit"
    chunks = chunk_by_heading(page(text))
    assert len(chunks) == 1 and "Tiny\nshort bit" in chunks[0].text


def test_fixed_chunks_overlap():
    chunks = chunk_fixed(page("a" * 2000), size=1000, overlap=150)
    assert [len(c.text) for c in chunks] == [1000, 1000, 300]


def test_slugify_and_urls():
    assert slugify("Import Pydantic's `BaseModel`") == "import-pydantics-basemodel"
    assert page_url("https://s", "index.md") == "https://s/"
    assert page_url("https://s", "tutorial/index.md") == "https://s/tutorial/"
    assert page_url("https://s", "tutorial/body.md") == "https://s/tutorial/body/"
