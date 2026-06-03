# Now

**Program:** MS FAIRE — Frontiers in AI & Research Engineering
**Timeline:** June 2026 → January/February 2027
**Handbook:** [faire-program.md](faire-program.md)

*Update this file whenever something changes. It is the single source of truth for what's active.*

---

## Current Term

| Term | Weeks | Subjects | Status |
|------|-------|----------|--------|
| **Term 1** | Weeks 1–12 | FAIRE 501 + start 502 | **← you are here** |
| **Term 2** | Weeks 13–24 | Finish 502 · 503 · start 504 | — |
| **Term 3** | Weeks 25–36 | Finish 504 · 505 · Thesis (599) | — |

---

## Active Right Now

| FAIRE Subject | Active Arc / Build | Folder | Status |
|---------------|-------------------|--------|--------|
| — | — | — | — |

## Active Capstones

| Capstone ID | Name | Folder | Status |
|-------------|------|--------|--------|
| — | — | — | — |

---

## The Question I'm Sitting With This Week

> [One question — from a paper, a lecture, or something that won't leave.]

---

## Qualifying Milestones (FAIRE 500)

- [ ] **Q1 — Reproduce a paper cold** *(~1 week, paper TBD)*
- [ ] **Q2 — Statistical defense** *(valid CIs + paired test + calibration check on a model comparison)*

---

## Horizon Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| Q1 qual — paper reproduced | End of Term 1 | — |
| Q2 qual — statistical defense | End of Term 1 | — |
| **503.1 — nanoLM pretraining** *(thesis spine begins)* | Mid Term 2 | — |
| 504.1 — Mini-R1 | End of Term 2 | — |
| 505.1 — Multi-GPU training stack | Mid Term 3 | — |
| **FAIRE 599 — Thesis defense** | January/February 2027 | — |

---

## How to Start an Arc or Build

```bash
# Scaffold a new experiment
bash scripts/new-experiment.sh 001 my-experiment-name experiments

# Or for a quick build
bash scripts/new-experiment.sh 001 my-build tiny-experiments

# Activate environment
source .venv/bin/activate   # or: uv run python script.py
```

---

## FAIRE Calendar → 30-Build Roadmap

| Month | FAIRE | Builds | Key deliverable |
|-------|-------|--------|----------------|
| June | 501 + 502 start | 1–5 (nanoLM base) | Q1 + Q2 quals |
| July | 502 | 6–10 (architecture + systems) | 502 capstones |
| August | 503 | 11–15 (LMs + serving) | **503.1 nanoLM** |
| September | 503 + 504 start | 16–20 (RL/reasoning) | Mini-R1 |
| October | 504 | 21–25 (interp + safety) | 504 capstones |
| November | 505 | 26–28 (systems) | 505 capstones |
| December | 599 thesis | 29–30 (capstone + synthesis) | Thesis draft |
| January | 599 defense | — | **Thesis defense** |
