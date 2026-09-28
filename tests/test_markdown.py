from docqa.markdown import clean_markdown, page_title


def test_resolves_code_includes_with_line_ranges(tmp_path):
    src = tmp_path / "docs_src" / "body"
    src.mkdir(parents=True)
    (src / "tutorial001.py").write_text("line1\nline2\nline3\nline4\n")
    raw = "Intro\n\n{* ../../docs_src/body/tutorial001.py ln[2:3] hl[2] *}\n"
    out = clean_markdown(raw, tmp_path)
    assert "```python\nline2\nline3\n```" in out
    assert "line1" not in out


def test_old_include_syntax_and_missing_file(tmp_path):
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "a.js").write_text("let x = 1;")
    out = clean_markdown("{!> ../../docs_src/a.js!}\n{* ../../docs_src/missing.py *}", tmp_path)
    assert "```javascript\nlet x = 1;\n```" in out
    assert "missing" not in out


def test_admonitions_tabs_links_and_anchors(tmp_path):
    raw = """---
hide: toc
---
# Request Body { #request-body }

/// tip | Technical Details

See [the guide](https://example.com) ![img](a.png)

///

//// tab | Python 3.10+

text

////
"""
    out = clean_markdown(raw, tmp_path)
    assert out.startswith("# Request Body\n")
    assert "Tip (Technical Details):" in out
    assert "See the guide" in out and "https://example.com" not in out and "a.png" not in out
    assert "Python 3.10+:" in out and "////" not in out and "///" not in out


def test_code_blocks_are_left_alone(tmp_path):
    raw = "```python\n/// not an admonition\n[x](y)\n```"
    assert clean_markdown(raw, tmp_path) == raw


def test_page_title():
    assert page_title("# First Steps { #first-steps }\nbody", "x") == "First Steps"
    assert page_title("no heading", "request-files") == "Request Files"
