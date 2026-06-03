# Bidirectional / Multi-Directional SSM Scanning

**One-liner:** Adapt the inherently-causal selective-SSM (Mamba) recurrence to non-sequential data by running multiple directed scans over a flattened token grid and summing their outputs — restoring the "every token sees every token" property that a single 1D left-to-right pass destroys.

## The key insight

A selective SSM computes `h_t = A_t h_{t-1} + B_t x_t`, `y_t = C_t h_t` — strictly causal, so token *t* only integrates tokens `≤ t` in the chosen scan order. For a 2D image (H×W patches) or audio spectrogram, any single flattening imposes an artificial order: row-major neighbors are far apart in the sequence, and the top-left patch can never attend to the bottom-right. Fix it by scanning the sequence in *K* directions and merging:

```
y = (1/K) Σ_k  Scan_k(x)            # or learned-weighted / concat-then-proj merge
```

- **Bidirectional (Vim):** K=2, forward + backward over the row-major sequence (two separate (A,B,C) projections), outputs summed.
- **Cross-scan (VMamba):** K=4 — row-major forward/backward + column-major forward/backward; each direction has independent parameters; results re-folded to the 2D grid and added (the "CSM"/SS2D module).
- Cost is K× the scan FLOPs but stays **O(L) / linear-time**, unlike attention's O(L²).

## Where it appears

- **Vision Mamba (Vim, 2024)** — bidirectional scan + a class token placed mid-sequence so both passes reach it; ViT-style backbone.
- **VMamba (2024)** — 2D Selective Scan (SS2D) with 4-way cross-scan; the canonical multi-directional vision SSM.
- **Audio / spectrogram & point-cloud / graph SSMs** — directional or serpentine/space-filling-curve scans to give locality across the 2D or 3D layout.
- **Mamba-2 vision variants & hybrid blocks** — same scan trick under the state-space-duality chunked form.

## Common mistake

Thinking multi-directional scanning makes the model "non-causal like attention" for free. Each token still only sees the union of what the *K* fixed scan orders reach — locality is order-dependent, far-apart 2D neighbors are still distant in every linear order, and you pay K× compute plus K× parameter sets. It approximates global mixing; it is not equivalent to full self-attention.

## See also
- [[selective-state-space-models-mamba]] — the causal recurrence being adapted
- [[hardware-aware-parallel-scan]] — how each directional pass is computed efficiently
- [[vision-transformer-patchification]] — the patch-grid-to-sequence step that creates the ordering problem
