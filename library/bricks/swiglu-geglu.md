# SwiGLU / GeGLU

**One-liner:** Gated FFN variants that split the up-projection into a gate branch (passed through Swish/GELU) and a value branch, multiply them elementwise, then down-project — three weight matrices instead of two, with hidden dim shrunk to ~2/3 to keep params constant; the default transformer FFN since ~2023.

## The formula

Standard FFN (two matrices):

FFN(x) = W_down · σ(W_up · x),  σ = ReLU or GELU

Gated GLU variant (three matrices W_gate, W_up, W_down):

FFN_GLU(x) = W_down · ( φ(W_gate · x) ⊙ (W_up · x) )

- **SwiGLU:** φ = Swish/SiLU, i.e. φ(z) = z · sigmoid(βz) (β=1 in practice)
- **GeGLU:** φ = GELU
- **ReGLU/Bilinear:** φ = ReLU / identity (ablation baselines)

⊙ is elementwise. W_gate, W_up ∈ ℝ^{d_ff × d_model}; W_down ∈ ℝ^{d_model × d_ff}. No bias terms in modern impls.

## The 2/3 convention

GLU adds a third matrix, so to match the ~2·d_model·d_ff param count of a vanilla FFN you set d_ff ≈ (2/3)·d_ff_dense. Llama uses d_ff ≈ (8/3)·d_model rounded to a hardware-friendly multiple (e.g. 11008 for 4096).

## Where it appears

- **Shazeer (2020), "GLU Variants Improve Transformer"** — origin; SwiGLU/GeGLU win on perplexity, with the famous "we offer no explanation... divine benevolence" line
- **Llama / Llama 2 / 3, PaLM, Mistral, Qwen, Gemma** — SwiGLU is the de facto standard FFN block
- **T5 v1.1 / mT5** — GeGLU variant; many encoder-decoders use GeGLU

## Common mistake

Forgetting to shrink d_ff: comparing a 3-matrix SwiGLU at full d_ff against a 2-matrix FFN is a 1.5x param/FLOP increase, not a fair architecture comparison. The whole point is iso-parameter (2/3 hidden) quality gains.

## See also
- [[gelu-silu]] — the activation functions (GELU, SiLU/Swish) used in the gate branch
- [[mlp]] — the standard two-matrix FFN that GLU variants replace
- [[moe-routing]] — MoE experts are themselves SwiGLU FFNs, just routed
