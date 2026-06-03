# Sliding-Window / Local Attention

**One-liner:** Restrict each query to attend only to the previous w keys (a fixed window), cutting attention cost to O(n·w) and capping KV memory at w per layer, while stacked layers grow the effective receptive field linearly with depth.

## The definition

Standard causal mask: token i attends to all j ≤ i. Sliding window of size w: token i attends only to j with i−w < j ≤ i.

attn_mask[i,j] = 1  iff  (i − w) < j ≤ i

- Compute / memory per layer: O(n·w) instead of O(n²).
- KV cache is bounded: only the last w keys/values need to be kept (rolling buffer), so memory does **not** grow with sequence length once you pass w.

**Receptive field via depth:** one layer reaches w tokens back; a token's information can propagate across windows through stacking. After L layers the effective receptive field is ≈ L·w tokens (e.g. Mistral 7B: w=4096, 32 layers → ~131k token theoretical reach).

## Where it appears

- **Mistral 7B (2023)** — pure sliding-window attention, w=4096, relies on depth × window for long-range; rolling KV buffer of size w.
- **Longformer / BigBird** — sliding window + a few global tokens; the original "local + global" recipe for long documents.
- **Gemma 2 / Gemma 3** — interleave local (sliding-window) and global (full) attention layers, e.g. 5:1 ratio, to bound KV cache while keeping some full-context layers.
- **GPT-OSS, Character.AI, Cohere Command** — alternate local/global layers as a standard 2024–2026 KV-saving pattern for long context serving.

## Common mistake

Assuming a token w-away is invisible. Information *does* reach beyond w through stacked layers (receptive field ≈ L·w), but the path is indirect and lossy — exact retrieval of a specific distant token (e.g. needle-in-haystack) degrades far more than the receptive-field math suggests, which is exactly why frontier models interleave a few **global** layers rather than going pure-local.

## See also
- [[kv-cache]] — the rolling-buffer bound on KV is the main payoff of sliding windows
- [[attention-sinks]] — pairing a window with always-kept initial tokens fixes streaming perplexity collapse
- [[rope-context-extension]] — windowing and positional extension are the two levers for long context
