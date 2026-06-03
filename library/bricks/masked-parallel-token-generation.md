# Masked / Parallel Token Generation (MaskGIT)

**One-liner:** Generate a grid of discrete visual tokens non-autoregressively by iteratively unmasking the highest-confidence predictions over a handful of decoding rounds, replacing O(N) raster-order steps with O(8-24) bidirectional passes.

## The algorithm (Chang et al., MaskGIT 2022)

Train a bidirectional transformer with a masked-token objective (BERT-style) over VQ tokens: mask a random subset (ratio sampled from a schedule), predict the masked tokens. Decode in T rounds (T ≪ N):

1. Start fully masked. For step t = 0..T-1:
2. Predict all masked tokens in parallel; sample a candidate + its confidence (predicted prob) for each masked position.
3. Compute mask ratio for next step via a **mask schedule** γ(t/T): keep n = ⌈γ((t+1)/T)·N⌉ positions masked.
4. **Re-mask** the N·(1−...) lowest-confidence tokens; freeze the high-confidence ones as final.

Mask schedule γ is typically **cosine**: γ(r) = cos(πr/2), unmasking few tokens early, many late. Confidence is often perturbed with Gumbel noise (temperature-annealed) to add diversity.

## Where it appears

- **MaskGIT (2022)** — original; ~10-12 steps vs 256+ for raster AR, ~30-60x faster, better FID on ImageNet.
- **Muse (2023)** — text-to-image at scale; masked decoding over VQGAN tokens conditioned on T5 embeddings, plus a super-res token model; uses CFG.
- **MAGVIT / MAGVIT-v2 (2023)** — video generation via masked token modeling over a 3D (spatiotemporal) tokenizer; v2's lookup-free quantizer lets it beat diffusion on video FVD.
- **Discrete masked diffusion** — MaskGIT is the absorbing-state limit; modern masked diffusion LMs (e.g. for text/code) generalize the schedule into a continuous-time process.

## Common mistake

Confusing the *parallel* prediction with *independent* sampling. Predicting all masked tokens at once assumes conditional independence given the current context — committing to all of them in one shot produces incoherent samples. The iterative confidence-based re-masking is what fixes this: only a few high-confidence tokens are frozen per round so later rounds condition on them. Cut the rounds to 1 and quality collapses.

## See also
- [[vq-vae-discrete-visual-tokenizers]] — supplies the discrete token grid MaskGIT operates over
- [[discrete-masked-diffusion-models]] — MaskGIT as the absorbing-state special case / continuous-time generalization
- [[autoregressive-visual-generation]] — the slow raster-order baseline this replaces
