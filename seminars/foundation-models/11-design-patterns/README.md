# 11 · Design Patterns & Theory of Foundation Models

*The meta-level: what makes FM transfer work across domains? What are the common patterns and the failure modes?*

## The recurring design decisions

Every domain faces the same five questions:

1. **Tokenization** — what is the atomic unit? (subword, patch, amino acid, nucleotide, time step, atom, mesh node...)
2. **Pretraining objective** — masked, contrastive, next-step prediction, denoising, generative?
3. **Inductive bias** — what architecture constraints encode domain knowledge? (equivariance for physics, positional encoding for sequence, causal masking for generation...)
4. **Adaptation** — fine-tuning, few-shot, prompt engineering, RLHF — what works here?
5. **Verification** — what's the ground truth? (fold/don't fold, code passes tests, forecast verifiable...)

## Paper Log

| Paper | Authors | Year | Link | Note |
|-------|---------|------|------|------|
| On the Opportunities and Risks of Foundation Models | Bommasani et al., Stanford | 2021 | [arXiv 2108.07258](https://arxiv.org/abs/2108.07258) | The manifesto; coined "foundation model" |
| A Path Towards Autonomous Machine Intelligence | LeCun | 2022 | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | JEPA as the right FM architecture for prediction |
| Scaling Laws for Neural Language Models | Kaplan et al. | 2020 | [arXiv 2001.08361](https://arxiv.org/abs/2001.08361) | Do scaling laws transfer across domains? |
| Pre-trained Models: Past, Present and Future | Han et al. | 2021 | [arXiv 2106.07139](https://arxiv.org/abs/2106.07139) | Survey of pretraining across NLP, vision, multimodal |
| Toward a Unified Theory of Transfer Learning | – | ongoing | – | Active research: when does transfer work? |

## The pattern that holds across all domains

The domains where FMs work best share:
1. **Large unlabeled data** exists (web text, protein sequences, DNA, satellite imagery, audio...)
2. **A natural pretraining objective** exists (next token, masked token, denoising...)
3. **A verification signal** exists at fine-tuning time (labels, simulation, ground truth...)
4. **The domain has compositional structure** — complex things are made of simpler reusable pieces

Where FMs struggle: scarce data, no natural self-supervised objective, no compositionality, hard evaluation.
