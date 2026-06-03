# Neural Audio Codecs (RVQ)

**One-liner:** Convolutional autoencoder + residual vector quantization compresses waveform audio into N stacked discrete code streams at ~kbps bitrates (SoundStream, EnCodec, DAC), giving the discrete "tokens" that token-based audio LMs (AudioLM, MusicGen, VALL-E) model autoregressively.

## The architecture / RVQ recipe

Encoder (strided 1D convs, ~320x downsample) maps waveform → continuous latent z. RVQ quantizes z with N codebooks applied to the **residual**:

```
r0 = z
for i in 1..N:
    q_i = nearest_codebook_entry(C_i, r_{i-1})   # argmin ||r - c||
    r_i = r_{i-1} - q_i                           # pass residual onward
z_hat = sum_i q_i                                 # reconstruct latent
```

Decoder (transposed convs) maps z_hat → waveform. Trained with reconstruction + multi-scale spectrogram (mel) loss + adversarial (multi-scale STFT discriminator) loss + commitment loss. Gradients flow through quantization via straight-through estimator; codebooks updated by EMA. **Quantizer dropout** (train with random N) enables one model to serve multiple bitrates.

Each codebook of size V over N levels at frame rate f gives bitrate = N · f · log2(V).

## Where it appears

- **SoundStream / EnCodec** — the canonical RVQ neural codecs; EnCodec tokens underlie MusicGen and many audio LMs
- **DAC (Descript Audio Codec, 2023)** — adds factorized + L2-normalized codes, snake activation; near-lossless at 8 kbps, the 2024+ default tokenizer
- **AudioLM / VALL-E / MusicGen** — model the RVQ code streams; AudioLM splits into semantic (w2v-BERT) + acoustic (codec) tokens; VALL-E uses an AR model for codebook-0 and non-AR for the rest
- **Mimi (Moshi, 2024)** — distills semantic info into the first codec level for full-duplex speech dialogue

## Common mistake

Treating the N code streams as independent or flattening them naively into one sequence. The codebooks are **hierarchical**: codebook 0 carries the coarse/most semantic content and later books only refine the residual. You must respect the coarse-to-fine dependency (delay patterns, per-level heads, or two-stage AR+NAR), or quality collapses and sequence length explodes.

## See also
- [[vq-vae-discrete-visual-tokenizers]] — the discrete-latent autoencoder RVQ generalizes (stacked VQ residuals)
- [[gumbel-softmax-straight-through-estimator]] — how gradients pass through the non-differentiable codebook argmin
- [[autoregressive-visual-generation]] — the analogous "tokenize then model with an LM" paradigm in vision
