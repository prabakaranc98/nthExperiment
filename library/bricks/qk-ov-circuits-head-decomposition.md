# QK / OV Circuits & Head Decomposition

**One-liner:** Factor an attention head into a query-key circuit W_QK = W_Q W_K^T (which positions attend to which) and an output-value circuit W_OV = W_O W_V (what information moves from source to destination), turning every head-level mechanistic claim into linear algebra on the residual stream.

## The formula / definition

A head reads from and writes to the residual stream. Its attention pattern is governed by the **QK circuit** and its written output by the **OV circuit**:

- Attention logits: `A_ij = (x_i W_Q)(x_j W_K)^T = x_i (W_Q W_K^T) x_j^T = x_i W_QK x_j^T`
- Output written back: `h = A · (x W_V) W_O = A · x W_OV`, with `W_OV = W_V W_O`

So one head's contribution to the residual stream is the bilinear/low-rank form `h_i = sum_j softmax_j(x_i W_QK x_j^T) · (x_j W_OV)`. The two circuits factor cleanly: W_QK (shape d×d, rank ≤ d_head) decides the pattern; W_OV (shape d×d, rank ≤ d_head) decides the content. Both are low-rank (d_head ≪ d_model) and read/write the full-dimensional stream — the W_Q/W_K and W_V/W_O splits are *not individually meaningful*, only the products are. Eigen/SVD analysis of W_QK and W_OV (e.g. eigenvalues of W_E W_QK W_E^T in token space) reveals what a head "copies" or "matches."

## Where it appears

- *A Mathematical Framework for Transformer Circuits* (Anthropic, Elhage et al. 2021) — origin of the QK/OV decomposition; full-rank one-layer attention-only models become products of these two circuits.
- **Induction heads** — a prev-token head's OV writes position info, then the induction head's QK matches "same token" and OV copies the next token; the canonical two-head composition argument.
- **Activation patching / attribution** — head-level interventions interpret deltas via which circuit (QK pattern vs OV content) was perturbed.
- **Logit lens & direct logit attribution** — W_OV W_U projects a head's write directly to vocab logits to read off "what it copies."

## Common mistake

Reading W_Q, W_K, W_V, W_O as separately meaningful. Only the *products* W_QK and W_OV are basis-independent; you can insert any invertible R between W_Q and W_K^T without changing the head. Also: forgetting QK and OV are independent — a head can attend somewhere (QK) yet move uninformative content (OV), or vice versa.

## See also
- [[induction-heads]] — the canonical multi-head circuit built from QK matching + OV copying
- [[residual-stream]] — the shared linear space both circuits read from and write to
- [[multi-head-attention]] — the architecture this decomposition factors
