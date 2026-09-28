"""Turn mkdocs-flavoured markdown into plain markdown we can index.

The FastAPI docs pull their code examples in from docs_src/ with lines like

    {* ../../docs_src/body/tutorial001_py310.py hl[4] *}
    {* ../../docs_src/sql_databases/tutorial001_an_py310.py ln[1:11] hl[7:11] *}

Without resolving these the index would contain almost no code, which is most
of what people ask about.
"""

from __future__ import annotations

import re
from pathlib import Path

INCLUDE_RE = re.compile(r"\{\*\s*(\S+)(.*?)\*\}")
OLD_INCLUDE_RE = re.compile(r"\{!>?\s*(\S+?)\s*!\}")
LINES_RE = re.compile(r"ln\[([\d:,\s]+)\]")
ADMONITION_RE = re.compile(r"^/// ?(\w+)(?:\s*\|\s*(.*))?$")
TAB_RE = re.compile(r"^//// ?tab\s*\|\s*(.*)$")
TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)

LANGS = {".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json",
         ".toml": "toml", ".sh": "bash", ".yml": "yaml", ".yaml": "yaml", ".txt": ""}


def _pick_lines(code: str, spec: str) -> str:
    """ln[1:11] or ln[1:5,9:12] -> only those (1-based, inclusive) lines."""
    lines = code.splitlines()
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            out.extend(lines[int(start) - 1:int(end)])
        elif part:
            out.append(lines[int(part) - 1])
    return "\n".join(out)


def _include(path: str, options: str, repo_root: Path) -> str:
    rel = re.sub(r"^(\.\./)+", "", path)
    file = repo_root / rel
    if not file.is_file():
        return ""
    code = file.read_text(encoding="utf-8").rstrip()
    m = LINES_RE.search(options or "")
    if m:
        code = _pick_lines(code, m.group(1))
    lang = LANGS.get(file.suffix, "")
    return f"```{lang}\n{code}\n```"


def clean_markdown(raw: str, repo_root: Path) -> str:
    text = raw
    # front matter
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    text = INCLUDE_RE.sub(lambda m: _include(m.group(1), m.group(2), repo_root), text)
    text = OLD_INCLUDE_RE.sub(lambda m: _include(m.group(1), "", repo_root), text)

    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        tab = TAB_RE.match(line.strip())
        if tab:
            # code tabs, e.g. "//// tab | Python 3.10+"
            out.append(f"{tab.group(1).strip()}:")
            continue
        m = ADMONITION_RE.match(line.strip())
        if m:
            # "/// tip | Technical Details" -> "Tip (Technical Details):", closing "///" dropped
            kind, label = m.group(1).capitalize(), (m.group(2) or "").strip()
            out.append(f"{kind} ({label}):" if label else f"{kind}:")
            continue
        if line.strip() in ("///", "////"):
            continue
        line = re.sub(r"<img[^>]*>", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # keep link text only
        line = re.sub(r"</?(abbr|a|span|div|details|summary)[^>]*>", "", line)
        line = re.sub(r"\s*\{\s*#[\w-]+\s*\}\s*$", "", line)  # "## Title { #anchor }"
        out.append(line)

    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def page_title(raw: str, fallback: str) -> str:
    m = TITLE_RE.search(raw)
    if not m:
        return fallback.replace("-", " ").title()
    return re.sub(r"\s*\{\s*#[\w-]+\s*\}\s*$", "", m.group(1)).strip()
