# Research Engineering — what it means here

*The **"& RE"** in FAIRE. Where [research-philosophy.md](research-philosophy.md) is the *craft of asking* — taste, questions, rigor — this is the **craft of building the evidence**: turning a frontier idea into a system that runs, scales, reproduces, and can be defended with a number.*

**FAIRE context:** the mission is to get better at the *foundations and frontiers of AI research **and** engineering* ([handbook](faire-program.md)). This doc is the "engineering" half of that identity — not a subject (505 is the systems *subject*), but the **stance** that runs through every subject.

---

## The definition (for this repo)

A **research engineer** is neither the researcher who has ideas but never ships, nor the software engineer who ships but doesn't understand the science. It's the fusion:

> the person who can **derive the idea** *and* **build it, scale it, measure it, and defend it** — who makes research *real, reproducible, and fast*.

At frontier labs (Anthropic, DeepMind, OpenAI) the Research Engineer is the role that reproduces the paper, builds and scales the training/inference stack, designs and runs the experiments, and owns the empirical loop that turns a hypothesis into evidence. This program is a self-directed apprenticeship to *that* profile.

## What it is *not*

- **Not** a SWE wiring APIs around a model you don't understand — that's a demo, and the [résumé trap](https://pracha.me/musings/posts/2026_07_03_curriculum-as-a-living-system.html#resume-trap).
- **Not** a researcher who reads papers but never touches the cluster or the code.
- **Not** "MLOps plumbing" divorced from the science, and **not** "prompt-and-ship" divorced from the systems.
- **Not** volume of projects. It's **residue** — working systems and reproduced results that changed what you understand.

## The five axes *are* research engineering

The [FAIRE rubric](faire-program.md) isn't a grading gimmick — it's the definition of the craft, decomposed:

| Axis | The research-engineering skill |
|------|-------------------------------|
| **Correctness & reproducibility** | it runs from a clean clone and produces the claimed result — the RE's core discipline |
| **Statistical rigor** | you attach *valid uncertainty* to every claim — the rare skill that reads as seniority |
| **Depth / from-scratch-ness** | you implemented it from first principles, not wired it together |
| **Engineering quality** | efficient, profiled, tested, clean — it survives contact with scale |
| **Communication** | a write-up a lab reviewer could read and *trust* |

A research engineer is someone who hits all five *at once*, on the same artifact.

## What a research engineer can *do* (what this program builds)

- **Build & scale a frontier model end to end** — data → tokenizer → pretrain → post-train (SFT → RLVR) → serve (503 + 505 + thesis).
- **Reproduce a paper cold**, fast, with ablations and an honest account of where you diverged (Q1).
- **Go down to the metal** — write a fused Triton/CUDA kernel, reason from the roofline, train multi-GPU with FSDP/TP/PP, diagnose a loss spike (505).
- **Run the frontier post-training loop** — RLHF/DPO/GRPO/RLVR, reasoning RL, and reverse-engineer it with an SAE (504).
- **Validate anything statistically** — a confidence interval, a paired test, a calibration check on every comparison (501).
- **Turn a live question into an experiment that leaves a canal** — the smallest build that answers *"is this actually true?"*

## Research engineering in the living system

Research engineering *is* the **Experiment** and **Interact** in the Learn ⇄ Experiment ⇄ Interact loop — the building that closes the loop and carves the groove. Two consequences:

- **First commit before first paper.** You build to *understand*, not after understanding. The system you make is how the theory becomes yours.
- **It's the antidote to the résumé trap.** RE done right produces *residue* — a working artifact, a reproduced result, sharpened taste — not a demo reverse-engineered from a job description. The residue is also, not coincidentally, the far better story.

## The bar — what it certifies

When the work is public and reproducible, "research engineer" here means you can, with evidence:

- **Build and scale the pipeline, not just use it.**
- **Reproduce, derive, and extend from first principles.**
- **Make any empirical claim defensible** — the number comes with an interval.
- **Do the 2024–2026 frontier work** — reasoning RL, post-training, interpretability, efficient systems — and explain *why* each choice is right.

That is the whole point of the program: not to collect credits, but to *become* the person who can do that — and prove it by shipping.

## See also

- [research-philosophy.md](research-philosophy.md) — the other half: the craft of the question
- [faire-program.md](faire-program.md) — the standard, the rubric, the subjects (esp. 505)
- [CLAUDE.md](CLAUDE.md) — how the adviser holds this bar without imposing a fixed path
