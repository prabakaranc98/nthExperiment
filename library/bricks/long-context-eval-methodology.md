# Long-Context Eval Methodology (Needle-in-Haystack, RULER)

**One-liner:** Synthetic retrieval/aggregation probes — planting "needle" facts in long filler ("haystack") and varying depth × length — to measure the *effective* context length, which is almost always far shorter than the advertised window.

## The methodology

**NIAH (Needle-in-a-Haystack, Kamradt 2023):** insert a sentence like "The magic number is 7" at relative depth d ∈ [0,1] inside a context of length L, then ask the model to retrieve it. Sweep a 2D grid (L on x-axis, d on y-axis); report per-cell exact-match accuracy as a heatmap. Reveals the "lost-in-the-middle" failure: accuracy dips for needles in the middle depths.

**RULER (Hsieh et al., 2024)** — the standard synthetic battery, 13 tasks in 4 families:
- **Retrieval (multi-needle, multi-key/value):** find k needles among distractors.
- **Multi-hop tracing (variable tracking):** chase `X1=X2; X2=X3; ...` chains.
- **Aggregation:** common-words / frequent-words extraction over the whole context.
- **QA:** real questions with gold paragraph hidden in distractors.

**Effective context length** = largest L at which a model still beats a fixed quality threshold (RULER uses the Llama2-7B-4k base score, ~85%). Define:

L_eff = max{ L : acc(L) ≥ τ }

Models advertising 128k–1M routinely have L_eff of 16k–64k.

## Where it appears

- **RULER (NVIDIA, 2024)** — exposed that "GPT-4-128k", Command-R, Yi, Mixtral etc. degrade well before their claimed window; the standard report card for new long-context releases.
- **Lost in the Middle (Liu et al., 2023)** — U-shaped positional accuracy curve; primacy/recency bias.
- **∞Bench, LongBench v2, NoLiMa, HELMET, BABILong** — push toward *non-lexical* matching, reasoning, and natural (not synthetic) long-doc tasks; NoLiMa removes literal keyword overlap so models can't shortcut via string match.
- **RoPE context-extension papers (YaRN, LongRoPE)** — NIAH heatmaps are the go-to validation that extension didn't break retrieval.

## Common mistake

Treating a near-perfect single-needle NIAH heatmap as proof of "true" long context. Single-needle retrieval is the *easiest* probe — it's solvable by lexical/attention-sink shortcuts. Multi-needle, multi-hop tracing, and aggregation collapse much earlier; and synthetic exact-match overstates real-world performance where the answer requires synthesis across the whole window, not copy-out of one planted string.

## See also
- [[rope-context-extension]] — NIAH/RULER are the standard validation harness for extension methods
- [[attention-sinks]] — a mechanism that inflates easy single-needle scores without real long-range reasoning
- [[benchmark-saturation-dynamic-private-benchmarks]] — why static synthetic needles get gamed and need refreshing
