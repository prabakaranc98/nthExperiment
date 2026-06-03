# Foundation Models — A Cross-Domain Survey Seminar

*How the foundation model paradigm spreads across every domain of science and engineering — the design patterns, the efforts, the breakthroughs, and the open questions.*

**The central question:** What changes when you apply large-scale pretraining + fine-tuning to a new domain? What transfers, what fails, what has to be reinvented? This seminar tracks the FM-ification of 15 domains — from language to virtual cells — and builds a synthetic understanding of what it takes to push a domain toward the frontier with foundation modeling.

---

## Why this seminar exists

Language models showed that scale + pretraining + adaptation is a general recipe. The frontier question is now: *how general?* Each domain below has researchers trying to answer it — and the answers reveal something deep about what foundation models actually are.

**Design patterns to look for across domains:**
- What is the "token"? (patch, amino acid, nucleotide, time step, atom, pixel...)
- What is the pretraining objective? (masked, contrastive, next-step, denoising, generative...)
- What does "fine-tuning" mean here? (supervised, few-shot, RLHF, domain-specific loss...)
- Where does transfer succeed and where does it fail?
- What is the right inductive bias for this domain?
- What is the "verification signal"? (protein folds or doesn't, code passes tests, climate forecast verifiable...)

---

## Domain Index

| # | Domain | Key question | Folders |
|---|--------|-------------|---------|
| 01 | [Language](01-language/README.md) | The reference domain | 01-language |
| 02 | [Vision](02-vision/README.md) | Patches as tokens; what SAM changed | 02-vision |
| 03 | [Time Series](03-time-series/README.md) | Zero-shot forecasting; is there a GPT-3 moment? | 03-time-series |
| 04 | [Audio & Speech](04-audio-and-speech/README.md) | Self-supervised speech; generative audio | 04-audio-and-speech |
| 05 | [Video](05-video/README.md) | Temporal modeling; world simulator hypothesis | 05-video |
| 06 | [Protein & Biology](06-protein-and-biology/README.md) | Structure prediction → design; AlphaFold3 era | 06-protein-and-biology |
| 07 | [Omics & Genomics](07-omics-and-genomics/README.md) | DNA/RNA as language; single-cell FMs | 07-omics-and-genomics |
| 08 | [Physical Sciences](08-physical-sciences/README.md) | Neural operators; materials discovery | 08-physical-sciences |
| 09 | [Tabular](09-tabular/README.md) | TabPFN; in-context Bayesian prediction | 09-tabular |
| 10 | [Multimodal](10-multimodal/README.md) | Cross-modal alignment; any-to-any | 10-multimodal |
| 11 | [Design Patterns & Theory](11-design-patterns/README.md) | The meta-level: what makes FM transfer work? | 11-design-patterns |
| 12 | [Virtual Cells](12-virtual-cells/README.md) | Simulating cellular biology at scale | 12-virtual-cells |
| 13 | [Climate & Earth](13-climate-and-earth/README.md) | Weather forecasting; earth system emulation | 13-climate-and-earth |
| 14 | [Code](14-code/README.md) | Code generation, execution, and reasoning | 14-code |
| 15 | [Voice](15-voice/README.md) | TTS, voice cloning, real-time generation | 15-voice |

---

## Reading approach

For each domain: read the **landmark paper** first (the one that cracked open the domain), then the **current frontier**, then one **critical/failure-mode paper** that shows where it breaks.

**Synthesis question per domain:** "What had to be invented fresh vs. what transferred directly from language modeling?"

Log notes using `_templates/cornell-note.md`. When a design pattern repeats across domains, write a Zettel.
