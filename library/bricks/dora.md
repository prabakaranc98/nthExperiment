# DoRA (Weight-Decomposed LoRA)

**One-liner:** Split each pretrained weight into magnitude (a per-column norm) and direction (the unit vector), train the magnitude directly while adapting the direction with LoRA — recovering much of the LoRA-vs-full-FT gap at ~LoRA cost.

## The formula / definition

Decompose the base weight W₀ ∈ ℝ^{d×k} column-wise:

  W₀ = m · (V / ‖V‖_c),  m = ‖W₀‖_c ∈ ℝ^{1×k}  (per-column L2 norm)

where ‖·‖_c is the column-wise norm and V is the directional matrix. DoRA fine-tunes:

  W' = m · ( (W₀ + BA) / ‖W₀ + BA‖_c )

- **m** (magnitude): trainable vector, updated directly by the optimizer.
- **BA** (direction): the LoRA low-rank update, B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, applied to V.
- The denominator ‖W₀ + BA‖_c renormalizes so BA only changes *direction*; magnitude is owned by m. Treat ‖·‖_c as a constant in backprop (stop-gradient) for efficiency.

At inference DoRA merges back to a single W' — **zero added latency**, exactly like LoRA.

## The key insight

Full FT shows large magnitude changes with small directional changes (or vice-versa), a negative correlation; LoRA forces magnitude and direction to move together (positive correlation), limiting expressivity. Decoupling them makes LoRA's learning pattern resemble full FT.

## Where it appears

- **DoRA (Liu et al., ICML 2024)** — the original; beats LoRA on commonsense reasoning, LLaMA/VL instruction tuning, often at lower rank.
- **PEFT / HuggingFace** — shipped as `use_dora=True` on `LoraConfig`; one flag over an existing LoRA setup.
- **QDoRA / quantized adapters** — DoRA composed with NF4 base weights (à la QLoRA) for memory-bound fine-tuning of large models.

## Common mistake

Thinking DoRA adds inference cost or extra served parameters. The magnitude m and merged direction collapse into one weight matrix at deploy time — same as LoRA. The overhead is purely at *training* time (computing the column norm and its gradient).

## See also
- [[lora]] — DoRA is LoRA applied only to the directional component plus a trainable magnitude
- [[qlora]] — combine with 4-bit base weights for memory-efficient fine-tuning (QDoRA)
- [[lora-merging-multi-adapter-serving]] — DoRA merges to a single weight, so the same serving tricks apply
