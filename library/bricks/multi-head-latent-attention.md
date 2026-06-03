# Multi-Head Latent Attention (MLA)

**One-liner:** Compress K/V into one low-rank latent vector cached per token (with RoPE carried on a small *decoupled* key portion), shrinking the KV cache below GQA while keeping full per-head expressivity — the attention of DeepSeek-V2/V3.

## The definition

Per token h_t, project down to a tiny shared latent and cache **only that**:
- c_t^{KV} = W^{DKV} h_t  ∈ ℝ^{d_c},  with d_c ≪ H·d_head  (e.g. 512 vs 64·128)

At attention time, reconstruct per-head K,V by up-projection:
- k_t^{C} = W^{UK} c_t^{KV},  v_t = W^{UV} c_t^{KV}  (H heads each)
- query is likewise low-rank: c_t^{Q}=W^{DQ}h_t, q_t^{C}=W^{UQ}c_t^{Q}

**The RoPE problem:** RoPE is position-dependent, so it can't be absorbed into the static W^{UK} (the rotation differs per token-pair). MLA *decouples* a small rotary part:
- k_t^{R} = RoPE(W^{KR} h_t)  — one shared rotary key (dim d_head^R, e.g. 64), cached alongside c_t^{KV}
- q_t^{R} = RoPE(W^{QR} c_t^{Q})  — per head
- final key/query = concat(content part, rotary part); score = [q^C; q^R]·[k^C; k^R]ᵀ / √(d_head + d_head^R)

**Cache per token** = d_c + d_head^R (e.g. 512 + 64 = 576) vs GQA's 2·H_kv·d_head. **Matrix-absorption trick:** W^{UK} folds into W^{UQ} and W^{UV} into W^O at inference, so you never materialize full K/V — you attend directly against the cached latent.

## Where it appears

- **DeepSeek-V2 (2024)** — introduced MLA; ~93% KV-cache reduction vs its MHA baseline, enabling cheap long-context serving.
- **DeepSeek-V3 / R1 (2025)** — MLA + DeepSeekMoE as the core block; the latent KV is what makes 128K context economical at scale.
- **Kimi, MiniMax and other 2025 frontier models** — adopt MLA-style latent KV compression as the GQA successor.

## Common mistake

Thinking MLA is just "low-rank K/V" and that you can RoPE the whole thing. The *decoupled rotary key* is the crux: without splitting position info into a separate small key, RoPE blocks the matrix-absorption that lets you cache the latent instead of full K/V — you'd be forced to reconstruct K every step and lose the memory win.

## See also
- [[gqa]] — the predecessor it beats: GQA drops KV heads, MLA compresses them into a shared latent
- [[kv-cache]] — the exact memory bottleneck MLA attacks, harder than GQA
- [[rope]] — the positional scheme whose absorption conflict forces MLA's decoupled rotary key
