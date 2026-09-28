"""Split pages into chunks.

Two strategies so they can be compared in the eval:

- by_heading: one chunk per section, prefixed with its heading path
  ("Request Body > Create your data model"). Long sections are split on
  paragraph/code-block boundaries, never inside a code block.
- fixed: plain sliding window over characters, the usual baseline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .sources import Page

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    page: str
    url: str
    title: str
    section: str  # heading path, e.g. "Request Body > Import Pydantic's BaseModel"
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embed_text(self) -> str:
        # the heading path carries a lot of meaning for short sections
        return f"{self.section}\n\n{self.text}"


def slugify(heading: str) -> str:
    # close enough to mkdocs' default so #anchors mostly work
    s = re.sub(r"[`*_]", "", heading.lower())
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s]+", "-", s).strip("-")


def _blocks(text: str) -> list[str]:
    """Paragraphs and whole code blocks."""
    blocks, cur, in_code = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _sections(page: Page) -> list[tuple[list[str], str]]:
    """[(heading path, body)] in page order."""
    sections: list[tuple[list[str], str]] = []
    path: list[str] = []
    body: list[str] = []
    in_code = False

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append((list(path) or [page.title], text))

    for line in page.text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            body = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            continue
        body.append(line)
    flush()
    return sections


def _split(blocks: list[str], max_words: int) -> list[str]:
    parts, cur, n = [], [], 0
    for b in blocks:
        w = len(b.split())
        if cur and n + w > max_words:
            parts.append("\n\n".join(cur))
            # carry the last block over if it's short, keeps some context
            cur = [cur[-1]] if len(cur[-1].split()) < max_words // 4 else []
            n = sum(len(x.split()) for x in cur)
        cur.append(b)
        n += w
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def chunk_by_heading(page: Page, max_words: int = 220, min_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, body in _sections(page):
        section = " > ".join(heading_path)
        anchor = slugify(heading_path[-1]) if len(heading_path) > 1 else ""
        url = page.url + (f"#{anchor}" if anchor else "")
        for part in _split(_blocks(body), max_words):
            # tiny sections read better glued onto the previous chunk of the same page
            if chunks and len(part.split()) < min_words and \
                    len(chunks[-1].text.split()) + len(part.split()) <= max_words:
                chunks[-1].text += f"\n\n{heading_path[-1]}\n{part}"
                continue
            chunks.append(Chunk(id=f"{page.path}#{len(chunks)}", page=page.path, url=url,
                                title=page.title, section=section, text=part))
    return chunks


def chunk_fixed(page: Page, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    text = page.text
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size]
        chunks.append(Chunk(id=f"{page.path}#{len(chunks)}", page=page.path, url=page.url,
                            title=page.title, section=page.title, text=piece))
        start += size - overlap
    return chunks
