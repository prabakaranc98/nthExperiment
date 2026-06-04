# Capstones

Portfolio-grade, defended builds — one per FAIRE subject, two per subject. Each capstone is a complete artifact: reproducible code + a written report that meets the FAIRE rubric. These are the degree deliverables.

**FAIRE context:** Capstones are the 1.5-credit defended projects (501.1, 501.2, 502.1, ...). → [Program Handbook](../faire-program.md)

---

## Log

| Capstone | Name | FAIRE Subject | Status | Score | Link |
|----------|------|---------------|--------|-------|------|
| 500-Q1 | Reproduce a paper cold | FAIRE 500 | — | — | — |
| 500-Q2 | Statistical defense | FAIRE 500 | — | — | — |
| 501-1 | An Honest Eval | FAIRE 501 | — | — | — |
| 501-2 | Math + Stats for Frontier AI | FAIRE 501 | — | — | — |
| 502-1 | Flow-matching generative model | FAIRE 502 | — | — | — |
| 502-2 | I-JEPA-lite + probing study | FAIRE 502 | — | — | — |
| 503-1 | nanoLM pretraining *(thesis spine)* | FAIRE 503 | — | — | — |
| 503-2 | Frontier component reproduction | FAIRE 503 | — | — | — |
| 504-1 | Mini-R1 | FAIRE 504 | — | — | — |
| 504-2 | Alignment & safety study | FAIRE 504 | — | — | — |
| 505-1 | Multi-GPU nanoLM training stack | FAIRE 505 | — | — | — |
| 505-2 | Mini-vLLM serving system | FAIRE 505 | — | — | — |
| 599 | Thesis: nanoLM end-to-end | FAIRE 599 | — | — | — |
| **R1** | Causal Foundation Model (cFM / Do-FM, 300–500M) — in-context causal inference | 501 / 502 *(research elective)* | not started | — | [capstone](causal-foundation-model/capstone.md) |

> **R-track = research / exploratory capstones** — self-initiated bets from the research thread (causal ML × world models × foundation models), cross-listed against the subjects they exercise. Spec → [causal-foundation-model/capstone.md](causal-foundation-model/capstone.md).

---

## FAIRE Rubric (pass = ≥4 on Correctness + Communication, ≥3 on rest)

| Axis | What it means |
|------|--------------|
| **Correctness & reproducibility** | Runs from a clean clone and produces the claimed result |
| **Statistical rigor** | Claims backed by valid uncertainty (CIs, paired tests, calibration) |
| **Depth / from-scratch-ness** | Implemented from first principles, not wired together |
| **Engineering quality** | Efficient, profiled, clean code |
| **Communication** | A write-up a lab reviewer could read and trust |

---

## Structure

```
capstones/
└── 501-1-honest-eval/
    ├── capstone.md     ← spec + rubric self-assessment (copy from _templates/capstone.md)
    ├── report.md       ← the written report (public-ready)
    ├── log.md          ← running notes
    └── src/            ← reproducible code
        ├── main.py
        └── README.md   ← "run this to reproduce"
```

## Start a new capstone

```bash
mkdir -p capstones/501-1-honest-eval/src
cp _templates/capstone.md capstones/501-1-honest-eval/capstone.md
touch capstones/501-1-honest-eval/report.md capstones/501-1-honest-eval/log.md
```
