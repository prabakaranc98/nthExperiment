# Now

**Program:** MS FAIRE — Frontiers in AI & Research Engineering
**Timeline:** June 2026 → January/February 2027
**Handbook:** [faire-program.md](faire-program.md)

*Update this file whenever something changes. It is the single source of truth for what's active.*

---

> **Status: PRE-FLIGHT** — curriculum + the 334-card library are authored, but **0 arcs / 0 capstones / 0 quals shipped.** The program *starts* when build #1 commits. Don't let content volume read as progress — *authored ≠ shipped.*

## Progress at a glance

| Track | Shipped | Target |
|-------|---------|--------|
| Arcs / modules (builds shipped) | 0 | ~30 |
| Capstones (defended) | 0 | 13 |
| Qualifying milestones (Q1, Q2) | 0 | 2 |
| Reviews (reconciliation) | 0 *(week-01 scaffold only)* | — |
| 100-hour blocks completed | 0 | — |
| *Concept library (authored, not "done")* | *334 bricks · 2 foundations docs* | — |

**Daily ritual:** 15–20 min spaced-repetition review ([anki/](anki/README.md)) · advance one paper through the loop (read → [Cornell note](_templates/cornell-note.md) → flag `## Anki` cards → build).

---

## Current 100-hour block  *(the operating unit)*

*One module/concept at a time, question-driven — [a living system, not a checklist](faire-program.md). Order **emerges** from the work; the term/calendar tables below are **maps for blind-spot awareness, not a schedule to walk**. Learning here is holistic and systemic, not reductionist.*

| Block # | Module / concept | Hours | Status |
|---------|------------------|-------|--------|
| — | — | 0 / ~100 | not started |

**Live question(s) I'm sitting with** — *the thing that won't leave; explore freely, then reconcile back into notes/bricks:*
> [your live question here — a paper, a wall you hit, a genuine curiosity]

**Loop state** (Learn ⇄ Experiment ⇄ Interact): —
**Canalization watch** — *ideas that keep recurring across notes/questions = grooves forming; deepen them into bricks:* —

---

## Terms & calendar  *(scaffold — a map, not a schedule)*

*The real operating unit is the 100-hour block above. This is a loose route for orientation and blind-spot awareness, not a timetable to walk.*

| Term | ~Weeks | Subjects (loose scaffold) | |
|------|--------|---------------------------|---|
| **Term 1** | 1–12 | 501 · 506 + start 502 | entry |
| **Term 2** | 13–24 | finish 502 · 503 · start 504 · tracks | — |
| **Term 3** | 25–36 | finish 504 · 505 · 507 · tracks · 599 | — |

---

## Active Right Now

| FAIRE Subject | Active Arc / Build | Folder | Status |
|---------------|-------------------|--------|--------|
| — | — | — | — |

## Active Capstones

| Capstone ID | Name | Folder | Status |
|-------------|------|--------|--------|
| R1 | Causal Foundation Model (cFM / Do-FM) | [causal-foundation-model](capstones/causal-foundation-model/capstone.md) | spec'd · not started |

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

## FAIRE Calendar → 30-Build Roadmap  *(a possible route, not a timetable)*

*Follow the live question, not the month. This shows one coherent path through the material — use it to spot what you're skipping, not to pace yourself.*

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
