# Large-Batch / Memory-Bank Negatives

**One-liner:** Contrastive learning needs many negatives per positive; you get them via huge in-batch negatives (CLIP), a momentum-encoded memory queue decoupled from batch size (MoCo), or a batch-decoupled loss (SigLIP) — a perpetual systems-vs-objective tradeoff.

## The key insight

InfoNCE's denominator is a sum over negatives; gradient quality and the implied bound on mutual information improve with the number of negatives K. The problem: how do you get K large without paying O(K) in batch size and activation memory?

Three answers:

1. **In-batch negatives (SimCLR, CLIP).** Every other example in the batch is a negative. K = B−1 (or 2B−1). Free negatives, but K is coupled to batch size, so scaling K means scaling B (CLIP used B=32768). The N×N similarity matrix and the all-gather across data-parallel ranks become the cost.

2. **Memory bank / momentum queue (MoCo).** Maintain a FIFO queue of K encoded keys from prior batches (K up to 65536), decoupled from B. Keys are encoded by a **momentum encoder** updated as an EMA of the query encoder: θ_k ← m·θ_k + (1−m)·θ_q (m ≈ 0.999). The EMA keeps queued keys consistent despite encoder drift, so stale negatives stay usable.

3. **Batch-decoupled loss (SigLIP).** Replace the softmax (which couples all negatives through one normalizer) with a per-pair **sigmoid** loss: −Σ_{i,j} log σ( z_{ij} · (s_i·t_j·τ + b) ), z_{ij}=+1 if matched else −1. No global normalization → no all-gather of the full similarity matrix → strong results at much smaller batch (≈16–32k still helps but the loss is robust down to a few k).

## Where it appears

- **CLIP / OpenCLIP** — pure in-batch negatives; performance scales with batch size, motivating sharded contrastive loss across GPUs.
- **MoCo v1/v2/v3** — momentum queue for vision SSL; decouples negative count from batch, runs on modest hardware.
- **SigLIP / SigLIP 2** — sigmoid loss removes the cross-device softmax normalizer; near-CLIP quality without TPU-pod-scale batches.
- **Dense retrieval (DPR, E5, GTE)** — in-batch + hard negatives; cross-batch / cross-device negative sharing to inflate K cheaply.

## Common mistake

Believing "more negatives is always strictly better." Quality matters more than count: a few mined hard negatives beat thousands of trivially-easy ones, and a stale memory bank without a momentum encoder hurts because queued keys come from an inconsistent (drifted) encoder. Also: with large B, the InfoNCE temperature τ must be re-tuned — it is not batch-size invariant.

## See also
- [[infonce-contrastive-loss-with-temperature]] — the loss whose denominator these methods are scaling
- [[siglip-sigmoid-contrastive-loss]] — the batch-decoupled alternative that sidesteps large batches
- [[hard-negative-mining]] — the quality-over-quantity complement to raw negative count
