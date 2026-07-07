# Arcs

Build units — the thing you actually ship. An arc is a focused, from-scratch build that **emerges from a live question** inside a [~100-hour block](../now.md) — an experiment that tests *"is this actually true?"* and leaves a canal. The [30-Builds Roadmap](../seminars/30-builds-roadmap.md) is a scaffold of good candidates, **not a weekly quota**; the real trigger is your question, never a résumé/JD. *First commit before first paper.*

**FAIRE context:** Arcs are the learn-by-building units within each subject. → [Program Handbook](../faire-program.md)

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
