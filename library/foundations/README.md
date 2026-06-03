# Foundations

The mathematical substrate that frontier ML papers assume you already know. Not an intro course — a precision calibration. Each topic is framed around *exactly* the version of it that appears in frontier work, with the frontier connection made explicit.

If you can't follow a derivation in a paper, the gap is probably in one of these. Come here, fill the gap, go back to the paper.

---

## Topics

| Topic | File | What frontier papers assume you know |
|-------|------|--------------------------------------|
| Linear Algebra for ML | [linear-algebra.md](linear-algebra.md) | Eigendecomposition, SVD, matrix calculus — why LoRA works, what NTK is, how attention is low-rank |
| Probability & Information Theory | [probability.md](probability.md) | KL divergence cold (RLHF/DPO/VAEs), mutual information (contrastive learning), Bayesian inference (ICL, PFNs) |
| Calculus & Automatic Differentiation | [calculus-and-autodiff.md](calculus-and-autodiff.md) | Jacobians, chain rule in vector form, score functions — backprop, diffusion score matching, RL policy gradients |
| Optimization: Deep Theory | [optimization.md](optimization.md) | Loss landscape geometry, implicit bias of SGD, edge of stability, μP, Adam's adaptive moments |
| Statistical Learning Theory | [statistical-learning.md](statistical-learning.md) | Generalization for overparameterized nets, double descent, PAC-Bayes — why interpolation works |
| Measure Theory & Probability Limits | [measure-theory.md](measure-theory.md) | Concentration inequalities, CLT and its failure modes — evals, generalization bounds, conformal prediction |
