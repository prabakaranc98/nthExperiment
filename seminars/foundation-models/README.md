# Foundation Models — A Cross-Domain Survey Seminar

*How the foundation-model paradigm spreads across every domain of science and engineering — the patterns, the breakthroughs, and the open questions.*

**The central question:** What changes when you apply large-scale pretraining + adaptation to a new domain? What transfers, what fails, what has to be reinvented? This seminar tracks the FM-ification of 15 domains — from language to virtual cells — and builds a synthetic view of what it takes to push a domain to the frontier.

**What changed recently (2026):** the frontier has shifted from *can we pretrain here?* to *does it scale, generalize zero-shot, and verify?* — reasoning/RL post-training, agentic and any-to-any systems, and test-time compute are now the live questions across nearly every domain.

---

## Why this seminar exists

Language models showed that scale + pretraining + adaptation is a general recipe. The frontier question is now: *how general?* Each domain below has researchers testing it — and their answers reveal what foundation models actually are.

**Design patterns to look for across domains:**
- **The "token":** patch, amino acid, nucleotide, time step, atom, pixel?
- **Pretraining objective:** masked, contrastive, next-step, denoising, generative?
- **Adaptation:** supervised, few-shot/in-context, RL post-training, domain loss?
- **Transfer:** where does it succeed, and where does it break?
- **Inductive bias:** what must be baked in vs. learned from scale?
- **Verification signal:** what makes an output checkable (folds, passes tests, forecast verifies)?

---

## Domain Index

| # | Domain | Key question | Folders |
|---|--------|-------------|---------|
| 01 | [Language](01-language/README.md) | The reference domain — has RL/reasoning post-training replaced the pretraining story? | 01-language |
| 02 | [Vision](02-vision/README.md) | Beyond patches and SAM — do self-supervised backbones still beat VLM features? | 02-vision |
| 03 | [Time Series](03-time-series/README.md) | Did the zero-shot forecaster's "GPT-3 moment" actually arrive, or plateau? | 03-time-series |
| 04 | [Audio & Speech](04-audio-and-speech/README.md) | Unified speech-LLM stacks — is the encoder/codec divide closing? | 04-audio-and-speech |
| 05 | [Video](05-video/README.md) | World models as simulators — do they learn physics or just appearance? | 05-video |
| 06 | [Protein & Biology](06-protein-and-biology/README.md) | Post-AlphaFold3: from structure prediction to generative design and binders | 06-protein-and-biology |
| 07 | [Omics & Genomics](07-omics-and-genomics/README.md) | Long-context DNA + single-cell FMs — do they beat strong task-specific baselines? | 07-omics-and-genomics |
| 08 | [Physical Sciences](08-physical-sciences/README.md) | Neural operators + ML interatomic potentials — universal models for simulation? | 08-physical-sciences |
| 09 | [Tabular](09-tabular/README.md) | TabPFN-scale in-context learning — does it now win on real, large tables? | 09-tabular |
| 10 | [Multimodal](10-multimodal/README.md) | Any-to-any and native multimodal — is late-fusion alignment obsolete? | 10-multimodal |
| 11 | [Design Patterns & Theory](11-design-patterns/README.md) | The meta-level: what actually makes FM transfer and scaling work? | 11-design-patterns |
| 12 | [Virtual Cells](12-virtual-cells/README.md) | Can a learned cell model predict unseen perturbations in silico? | 12-virtual-cells |
| 13 | [Climate & Earth](13-climate-and-earth/README.md) | ML weather now beats numerics — how far do earth-system emulators generalize? | 13-climate-and-earth |
| 14 | [Code](14-code/README.md) | From generation to agentic execution — does verification/RL close the reliability gap? | 14-code |
| 15 | [Voice](15-voice/README.md) | Real-time, full-duplex speech-to-speech — has streaming dissolved the TTS pipeline? | 15-voice |

---

## Reading approach

Per domain: read the **landmark paper** (what cracked the domain open), then the **current frontier**, then one **critical/failure-mode paper** showing where it breaks.

**Synthesis question:** "What had to be invented fresh vs. what transferred directly from language modeling?"

Log notes with `_templates/cornell-note.md`. When a pattern repeats across domains, write a Zettel.
