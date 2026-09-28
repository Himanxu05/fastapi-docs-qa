"""Download the FastAPI docs and turn them into clean pages."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .markdown import clean_markdown, page_title

DOCS_PATH = "docs/en/docs"

# pages that are noise for Q&A (changelogs, people lists, marketing, API reference stubs)
SKIP = {
    "release-notes.md", "fastapi-people.md", "management.md", "management-tasks.md",
    "newsletter.md", "translation-banner.md", "_llm-test.md", "external-links.md",
    "contributing.md", "help-fastapi.md", "benchmarks.md",
}
SKIP_DIRS = {"reference", "img", "css", "js", "about"}


@dataclass
class Page:
    path: str  # e.g. "tutorial/body"
    url: str
    title: str
    text: str


def fetch_docs(repo: str, ref: str, dest: Path) -> Path:
    """Sparse, shallow clone of just the docs + code examples. Skips if already there."""
    if (dest / DOCS_PATH).is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1",
                    "--branch", ref, "--filter=blob:none", "--sparse", repo, str(dest)],
                   check=True)
    subprocess.run(["git", "sparse-checkout", "set", DOCS_PATH, "docs_src"], cwd=dest, check=True)
    return dest


def page_url(site: str, rel: str) -> str:
    # mkdocs: tutorial/body.md -> /tutorial/body/, tutorial/index.md -> /tutorial/
    rel = rel.removesuffix(".md")
    if rel == "index":
        return site + "/"
    if rel.endswith("/index"):
        rel = rel[: -len("/index")]
    return f"{site}/{rel}/"


def load_pages(repo_root: Path, site: str) -> list[Page]:
    docs = repo_root / DOCS_PATH
    pages = []
    for md in sorted(docs.rglob("*.md")):
        rel = md.relative_to(docs).as_posix()
        if rel in SKIP or rel.split("/")[0] in SKIP_DIRS:
            continue
        raw = md.read_text(encoding="utf-8")
        text = clean_markdown(raw, repo_root)
        if len(text.split()) < 30:
            continue
        pages.append(Page(path=rel.removesuffix(".md"), url=page_url(site, rel),
                          title=page_title(raw, fallback=md.stem), text=text))
    return pages
