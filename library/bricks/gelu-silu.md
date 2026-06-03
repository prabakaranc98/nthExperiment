# GELU / SiLU (Smooth Activations)

**One-liner:** Smooth, non-monotonic gating-style nonlinearities — GELU multiplies x by its Gaussian CDF, SiLU/Swish multiplies x by its sigmoid — used as the base activation inside transformer FFNs in place of ReLU because they pass small gradients on the negative side.

## The formula / definition

**GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016):
GELU(x) = x · Φ(x) = x · (1/2)[1 + erf(x / √2)]

Tanh approximation (what most kernels actually compute):
GELU(x) ≈ 0.5x · (1 + tanh[ √(2/π) · (x + 0.044715 x³) ])

**SiLU / Swish** (Sigmoid Linear Unit; Elfwing 2017 / Ramachandran 2017):
SiLU(x) = x · σ(x) = x / (1 + e^{−x})

Swish-β = x · σ(βx) generalizes it; β=1 is SiLU. As β→∞ Swish → ReLU; as β→0 it → linear/2.

Both are smooth (C^∞), non-monotonic (dip below 0 near x≈−1 to −2), self-gated, and approximately equal to each other — GELU and SiLU curves nearly overlap. Both ≈ ReLU = max(0,x) for large |x| but with a soft, differentiable knee.

## Where it appears

- **GELU** — BERT, GPT-2/3, ViT, most pre-2022 transformer FFNs as the inner activation `W₂·GELU(W₁x)`.
- **SiLU/Swish** — EfficientNet, YOLO backbones; the activation inside **SwiGLU** (`SiLU(W₁x) ⊙ (W₃x)`) used in LLaMA, PaLM, Mistral, Qwen, Gemma.
- **GEGLU** — GELU-gated FFN variant (T5 v1.1, some Gemma configs).
- Diffusion U-Nets / DiT blocks routinely default to SiLU.

## Common mistake

Conflating the activation with the gating scheme. SwiGLU/GeGLU are GLU *variants* (an extra elementwise multiply by a learned linear branch, 3 weight matrices, hidden dim shrunk to ~2/3 d_ff for param parity) — the "Swi"/"Ge" only names which base activation gates the branch. SwiGLU ≠ SiLU. Also: the tanh-GELU approximation and exact erf-GELU differ slightly (~1e-3); mixing them between training and inference frameworks can cause subtle numerical mismatch.

## See also
- [[swiglu-geglu]] — the GLU-gated FFN variants that wrap these base activations
- [[mlp]] — the FFN block where the activation lives
- [[layer-norm]] — the other smooth component paired with the FFN in a transformer block
