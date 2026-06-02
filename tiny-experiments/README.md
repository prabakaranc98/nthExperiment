# Tiny Experiments

Fast, focused, single-file experiments. The bar is low — get something running in under an hour. Break it. Learn one thing.

Spawned from papers, lectures, or curiosity. No pressure to be clean.

---

## Log

| # | Title | Framework | Spawned From | Status | Date |
|---|-------|-----------|-------------|--------|------|
| — | — | — | — | — | — |

---

## Structure

```
tiny-experiments/
└── 001-attention-from-scratch/
    ├── main.py        ← or notebook.ipynb
    └── notes.md       ← what you found
```

## Scaffold a new one

```bash
bash scripts/new-experiment.sh 001 attention-from-scratch tiny-experiments
```

## Run anything

```bash
uv run python tiny-experiments/001-attention-from-scratch/main.py
# or
uv run jupyter lab
```
