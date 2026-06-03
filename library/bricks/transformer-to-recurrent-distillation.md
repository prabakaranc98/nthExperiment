# Transformer-to-Recurrent Distillation / Linearization

**One-liner:** Convert a pretrained softmax-attention transformer into a linear-attention/SSM student by reusing the teacher's weights and matching it layer-by-layer or end-to-end, recovering most quality for a tiny fraction of from-scratch compute (often <1% of pretraining tokens).

## The recipe (shared skeleton)

Most methods follow the same staged pipeline:

1. **Architecture swap** — replace softmax attention `O = softmax(QKᵀ/√d)V` with a linear/recurrent mixer. Reuse Q,K,V,O projections and all MLP/norm weights from the teacher.
2. **Block / matrix alignment** — match the student's *sequence-mixing operator* to the teacher's attention matrix or hidden states, layer-by-layer (distillation through the mixer), minimizing e.g. `‖A_student − softmax(QKᵀ/√d)‖`.
3. **Hidden-state distillation** — train student to match teacher activations at each block: `Σ_ℓ ‖h_student^ℓ − h_teacher^ℓ‖²`.
4. **End-to-end finetune** — final logit KD: `KL(p_teacher ‖ p_student)` on a small token budget.

The student's recurrent form gives O(1) per-step state and O(N) total (no KV cache growth).

## Where it appears

- **SUPRA** (Mercat et al., 2024) — "Uptraining" linear-RNN attention from a transformer; swaps softmax for GroupNorm-normalized linear attention, finetunes on ~20B tokens.
- **MOHAWK** (Bick et al., 2024) — three-stage distillation (matrix orientation → hidden-state alignment → weight transfer) to make Mamba-2 students; <1% of teacher's pretraining data.
- **Mamba-in-Llama** (Wang/Gu et al., 2024) — initializes Mamba SSM from Llama Q,K,V; keeps MLPs frozen; adds speculative decoding for the distilled hybrid.
- **LoLCATs** (Zhang et al., 2024) — attention-transfer to learnable linear/sliding-window attention, then **LoRA** to recover quality; linearizes 70B/405B Llama cheaply.

## Common mistake

Assuming the converted model inherits the teacher's long-context behavior for free. The recurrent student has a **fixed-size state**, so associative recall and long-context retrieval degrade; that is the capacity bottleneck distillation does *not* fix. Hybrids (a few retained full-attention layers) exist precisely to patch this.

## See also
- [[linear-attention]] — the target architecture's `(φ(Q)(φ(K)ᵀV))` recurrent form
- [[selective-state-space-models-mamba]] — the most common SSM student
- [[knowledge-distillation]] — the logit/hidden-state matching objective being reused
