#!/usr/bin/env python3
"""Generate a starter Anki deck from the brick library.

Each brick contributes up to two cards: a "define/recall" card from its
**One-liner:**, and a "common mistake" card from its `## Common mistake`
section. Output is a tab-separated file ready for Anki's File -> Import.

Usage: python3 scripts/anki-bricks.py            # -> anki/decks/bricks-starter.tsv
"""
import re, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().replace("\t", " ")


def main() -> int:
    cards = []
    for f in sorted(glob.glob(os.path.join(ROOT, "library/bricks/*.md"))):
        base = os.path.basename(f)
        if base in ("README.md", "topic-map.md"):
            continue
        txt = open(f, encoding="utf-8").read()
        m = re.search(r"^#\s+(.+)$", txt, re.M)
        title = m.group(1).strip() if m else base[:-3]
        ol = re.search(r"\*\*One-liner:\*\*\s*(.+)", txt)
        if ol:
            cards.append((f"Define / recall: <b>{title}</b>", clean(ol.group(1)), "nthexp::bricks"))
        cm = re.search(r"##\s*Common mistake\s*\n+(.+?)(?:\n##|\Z)", txt, re.S)
        if cm:
            body = clean(cm.group(1))
            if len(body) > 350:
                body = body[:350].rsplit(" ", 1)[0] + "…"
            cards.append((f"Common mistake — {title}?", body, "nthexp::bricks::mistakes"))

    out = os.path.join(ROOT, "anki/decks/bricks-starter.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("#separator:tab\n#html:true\n#tags column:3\n")
        for front, back, tags in cards:
            fh.write(f"{front}\t{back}\t{tags}\n")
    print(f"wrote {out}: {len(cards)} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
