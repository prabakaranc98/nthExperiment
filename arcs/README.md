# Arcs

Weekly build units from the [30-Builds Roadmap](../seminars/30-builds-roadmap.md). Each arc is a focused, from-scratch build — the thing you ship at the end of the week. Bricks become arcs when they're running.

**FAIRE context:** Arcs are the 1-credit learning units within each subject (501–505). → [Program Handbook](../faire-program.md)

---

## Log

| Arc # | Name | FAIRE Subject | Ships when | Status | Link |
|-------|------|---------------|-----------|--------|------|
| — | — | — | — | — | — |

---

## Structure

```
arcs/
└── 001-bpe-tokenizer/
    ├── arc.md          ← the build spec (copy from _templates/arc.md)
    ├── log.md          ← running notes while building
    ├── reflection.md   ← fill when shipped
    └── src/            ← code lives here
        └── main.py
```

## Start a new arc

```bash
bash scripts/new-experiment.sh 001 bpe-tokenizer arcs
```
