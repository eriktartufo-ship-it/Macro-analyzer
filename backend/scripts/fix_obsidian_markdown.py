"""Fix idempotente per cosmetic markdown lint warnings nei file Obsidian.

Fix automatici:
- MD060/table-column-style: spazi attorno alle pipe nelle tabelle compatte (|a|b| → | a | b |)
- MD004/ul-style: liste unordered usano `-` invece di `+` o `*`
- MD032/blanks-around-lists: blank line prima/dopo liste
- MD040/fenced-code-language: code fence senza language → ```text
- MD047/single-trailing-newline: esattamente una newline finale
- MD037/no-space-in-emphasis: `* text *` → `*text*`
- MD009/no-trailing-spaces: rimuove trailing spaces (preserva line break Markdown intenzionale a 2+ spazi)
- MD041/first-line-heading: skip se il file è frontmatter o snippet
- MD031/blanks-around-fences: blank line prima/dopo ```

Run:
    python scripts/fix_obsidian_markdown.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET_DIR = Path("d:/Antigravity/obsidian/Erik/02_Progetti/Macro_Analyzer/")


def fix_md004_ul_style(text: str) -> str:
    """`+ item` o `* item` (top-level e nested) → `- item`. Preserva indent."""
    # Match indented list markers (not in code blocks)
    out_lines = []
    in_code_block = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue
        # Match `  + ` or `  * ` (not `**` bold markers)
        m = re.match(r"^(\s*)([+*])(\s+)(.*)$", line)
        if m and m.group(2) in ("+", "*"):
            indent, marker, space, content = m.groups()
            # Skip se è bullet inline come `*text*` (emphasis)
            if marker == "*" and content.startswith("*"):
                out_lines.append(line)
                continue
            out_lines.append(f"{indent}-{space}{content}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def fix_md060_table_compact(text: str) -> str:
    """Tabelle compatte `|a|b|` → `| a | b |`. Preserva contenuto cella."""
    out_lines = []
    in_code_block = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue
        # Una riga tabella deve iniziare con `|` e contenere almeno 2 pipe
        stripped = line.rstrip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            # Split per pipe, strip ogni cella, rejoint con padding
            cells = stripped.split("|")
            # Primo e ultimo sono empty (per | iniziale/finale)
            # Strip cells interne
            new_cells = []
            for i, c in enumerate(cells):
                if i == 0 or i == len(cells) - 1:
                    new_cells.append(c)  # preserve edge
                else:
                    new_cells.append(f" {c.strip()} ")
            new_line = "|".join(new_cells).rstrip()
            out_lines.append(new_line)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


_LIST_ITEM_RE = re.compile(r"^\s*[-*+]\s+|^\s*\d+\.\s+")


def _is_list_item(line: str) -> bool:
    return _LIST_ITEM_RE.match(line) is not None


def _is_list_continuation(line: str) -> bool:
    """Continuation: indent >= 2 spaces, content non-empty, NON list marker."""
    if not line.strip():
        return False
    if _is_list_item(line):
        return False
    # Almeno 2 spazi di indent
    return line.startswith("  ") or line.startswith("\t")


def fix_collapse_continuation_blanks(text: str) -> str:
    """Rimuove blank lines spurie INTRA-lista.

    Casi target:
    1. Tra list item e sua continuation indented (`- item\\n\\n  cont` → `- item\\n  cont`).
    2. Tra continuation di un item e il prossimo item della stessa lista
       (`  cont\\n\\n- next` → `  cont\\n- next`).
    3. Tra due list items consecutivi (`- a\\n\\n- b` → `- a\\n- b`).
    """
    lines = text.split("\n")
    out: list[str] = []
    in_code_block = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            i += 1
            continue
        out.append(line)
        if in_code_block:
            i += 1
            continue
        # Se questo è list-related, prova a collapse blanks verso il prossimo list-related
        if _is_list_item(line) or _is_list_continuation(line):
            j = i + 1
            blanks_idx: list[int] = []
            while j < len(lines) and lines[j].strip() == "":
                blanks_idx.append(j)
                j += 1
            if j < len(lines):
                nxt = lines[j]
                nxt_is_list_related = _is_list_item(nxt) or _is_list_continuation(nxt)
                if nxt_is_list_related and blanks_idx:
                    out.append(nxt)
                    i = j + 1
                    continue
        i += 1
    return "\n".join(out)


def fix_md032_blanks_around_lists(text: str) -> str:
    """Garantisce blank line prima/dopo blocchi lista. NON triggera su continuation."""
    lines = text.split("\n")
    out = []
    in_code_block = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            continue
        if in_code_block:
            out.append(line)
            continue
        is_list = _is_list_item(line)
        prev = lines[i - 1] if i > 0 else ""
        prev_is_list = _is_list_item(prev)
        prev_is_continuation = _is_list_continuation(prev)
        prev_is_blank = prev.strip() == ""
        prev_is_heading = prev.lstrip().startswith("#")
        # Inserisci blank line PRIMA solo se è inizio nuovo blocco lista
        # (prev è prosa, non lista, non continuation di lista, non heading)
        if (
            is_list
            and not prev_is_list
            and not prev_is_continuation
            and not prev_is_blank
            and prev.strip()
            and not prev_is_heading
        ):
            out.append("")
        out.append(line)
    # Garantisce blank line DOPO l'ultimo elemento lista
    # (curr è lista, nxt è prosa non-lista non-continuation non-blank)
    final = []
    for i, line in enumerate(out):
        final.append(line)
        if i + 1 < len(out):
            curr_is_list = _is_list_item(line) or _is_list_continuation(line)
            nxt = out[i + 1]
            nxt_is_list = _is_list_item(nxt)
            nxt_is_continuation = _is_list_continuation(nxt)
            nxt_is_blank = nxt.strip() == ""
            if (
                curr_is_list
                and not nxt_is_list
                and not nxt_is_continuation
                and not nxt_is_blank
            ):
                final.append("")
    return "\n".join(final)


def fix_md040_fenced_code_language(text: str) -> str:
    """\\`\\`\\` senza language → \\`\\`\\`text."""
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "```":
            # Distinguere apertura vs chiusura non triviale, lascio come è per ora
            # Solo se sicuro che è apertura (basato su context)
            out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def fix_md040_smart(text: str) -> str:
    """\\`\\`\\` da solo come apertura → \\`\\`\\`text. Tracking open/close."""
    out = []
    in_code_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "```":
            if not in_code_block:
                # apertura senza lang
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}```text")
                in_code_block = True
            else:
                out.append(line)
                in_code_block = False
        elif stripped.startswith("```") and not in_code_block:
            in_code_block = True
            out.append(line)
        elif stripped.startswith("```") and in_code_block:
            in_code_block = False
            out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def fix_md047_trailing_newline(text: str) -> str:
    """Esattamente una newline finale."""
    return text.rstrip("\n") + "\n"


def fix_md037_emphasis_spaces(text: str) -> str:
    """`* text *` → `*text*`, `** text **` → `**text**`. Solo single-line."""
    out_lines = []
    in_code_block = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue
        # Bold double-asterisk con spazi interni
        new_line = re.sub(r"\*\* ([^\n*]+?) \*\*", r"**\1**", line)
        # Italic single-asterisk con spazi interni
        new_line = re.sub(r"(?<![*\w])\* ([^\n*]+?) \*(?![*\w])", r"*\1*", new_line)
        out_lines.append(new_line)
    return "\n".join(out_lines)


def fix_md009_trailing_spaces(text: str) -> str:
    """Rimuove trailing spaces. Preserva line break a 2 spazi."""
    out = []
    for line in text.split("\n"):
        # Conta trailing spaces
        stripped = line.rstrip(" ")
        n_trailing = len(line) - len(stripped)
        if n_trailing >= 2:
            # Linebreak Markdown intenzionale, preserva 2 spazi esatti
            out.append(stripped + "  " if line != stripped else line)
        else:
            out.append(stripped)
    return "\n".join(out)


def fix_md031_blanks_around_fences(text: str) -> str:
    """Blank line prima/dopo `\\`\\`\\``."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_fence = stripped.startswith("```")
        prev = lines[i - 1] if i > 0 else ""
        prev_blank = prev.strip() == ""
        if is_fence and not prev_blank and prev.strip():
            out.append("")
        out.append(line)
    final = []
    for i, line in enumerate(out):
        final.append(line)
        if i + 1 < len(out):
            stripped = line.strip()
            is_fence = stripped.startswith("```")
            nxt = out[i + 1]
            nxt_blank = nxt.strip() == ""
            if is_fence and not nxt_blank and nxt.strip():
                final.append("")
    return "\n".join(final)


def fix_file(path: Path, dry_run: bool = False) -> bool:
    """Applica tutti i fix idempotenti. Returns True se modificato."""
    if not path.exists() or path.suffix != ".md":
        return False
    original = path.read_text(encoding="utf-8")
    text = original

    # Skip file con frontmatter complesso (MEMORY.md, agent files)
    if text.startswith("---\n") and "name:" in text[:200]:
        return False

    text = fix_md004_ul_style(text)
    text = fix_md060_table_compact(text)
    text = fix_md037_emphasis_spaces(text)
    text = fix_md009_trailing_spaces(text)
    # Recupera dal danno del MD032 v1: collapse blanks spurie tra item e continuation
    text = fix_collapse_continuation_blanks(text)
    text = fix_md032_blanks_around_lists(text)
    text = fix_md031_blanks_around_fences(text)
    text = fix_md040_smart(text)
    # Collapse multiple blank lines a max 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = fix_md047_trailing_newline(text)

    if text != original:
        if not dry_run:
            path.write_text(text, encoding="utf-8")
        return True
    return False


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    if not TARGET_DIR.exists():
        print(f"ERROR: target dir not found: {TARGET_DIR}")
        return 1
    print(f"Scanning {TARGET_DIR}")
    modified = []
    skipped = []
    for md in TARGET_DIR.rglob("*.md"):
        try:
            changed = fix_file(md, dry_run=dry_run)
            if changed:
                modified.append(md)
            else:
                skipped.append(md)
        except Exception as e:
            print(f"  ERROR {md.name}: {e}")
    print(f"\nModified: {len(modified)} files (dry-run={dry_run})")
    for p in modified:
        print(f"  - {p.relative_to(TARGET_DIR)}")
    print(f"Unchanged/skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
