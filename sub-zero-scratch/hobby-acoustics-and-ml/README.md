# Hobby — Acoustics + ML

**Type:** Hobby (Artistic Brain Outlet)
**Serves:** Keeps the artistic brain shipping. Not on the research ledger — meaning: not measured by publications or benchmarks, but by whether it sounds good and feels alive.

---

## The Premise

DDSP (Differentiable Digital Signal Processing) is the intersection of audio synthesis and ML that is underexplored and genuinely beautiful. Carnatic music — specifically gamaka, the ornamental microtonal inflections that are the soul of the style — is a domain where standard pitch-quantized ML completely fails. The combination is: can you build a system that generates, transforms, or understands Carnatic gamaka in a way that is musically meaningful?

This is not a research project. It is a hobby with technical depth. The constraint: if it stops being fun, pause it.

---

## What "From Scratch" Means Here
- Understand audio at the signal level: waveforms, spectrograms, Fourier transforms, mel-filterbanks — not just as API calls
- Build a basic synthesizer in Python before using DDSP
- Understand what gamaka is from a musical standpoint before trying to model it

---

## Three Areas

### 1. Acoustics Foundations
Sound is physics. Before neural audio, understand what you're modeling:
- The Fourier transform as a change of basis
- The short-time Fourier transform (STFT) and spectrograms
- Mel-filterbanks and why they approximate human perception
- Fundamental frequency, harmonics, formants — the structure of musical tones
- What makes two instruments playing the same note sound different (timbre)

### 2. DDSP — Differentiable Digital Signal Processing
Magenta's framework for combining classical DSP with neural networks. The key idea: instead of generating waveforms directly (difficult), generate the *parameters* of a synthesizer (harmonic oscillator + filtered noise), then use DSP to produce audio. This gives you:
- Interpretable latent space (F0, loudness, timbre)
- Physically grounded synthesis
- Much better sample efficiency for musical signals

### 3. Carnatic Gamaka
Gamaka (గమక / गमक) refers to the ornamental glides, oscillations, and microtonal inflections in Carnatic music. A single note is not a fixed pitch — it is a trajectory. This makes standard MIDI-style modeling completely wrong for this music.

The technical problem: how do you represent and generate the continuous pitch trajectories of gamaka in a way that is musically correct, not just statistically plausible?

---

## Build Progression

- [ ] **Step 1:** Implement a Fourier transform from scratch (not numpy.fft — the actual DFT). Understand what it computes.

- [ ] **Step 2:** Build a simple additive synthesizer: given F0 and harmonic amplitudes, generate a waveform. Render a note.

- [ ] **Step 3:** Run the DDSP autoencoder on a violin recording. Visualize the F0, loudness, and timbre latents. Understand what each controls.

- [ ] **Step 4:** Record or source 10 examples of a single Carnatic raga (e.g., Bhairavi) with clear gamaka. Visualize the F0 trajectories. What patterns appear?

- [ ] **Step 5:** Train a simple model (LSTM or small transformer) to generate F0 trajectories in the style of the raga. Feed into the DDSP synthesizer. Does it sound like it's in the right style?

- [ ] **Step 6:** Break something interesting — try to transfer gamaka style from one raga to another. What holds, what falls apart?

---

## Essential Resources

- [ ] **DDSP: Differentiable Digital Signal Processing — Engel et al. (2020)**
  - §2: Differentiable synthesizers
  - §3: The DDSP autoencoder
  - §5: Applications

- [ ] **The Science of Sound — Rossing, Moore, Wheeler** *(selected)*
  - Ch. 1–3: Vibration, waves, sound
  - Ch. 7: The human voice

- [ ] **Librosa documentation** — the audio ML toolkit

- [ ] **Carnatic Music Theory — various**
  - Gamaka types: kampita, janta, sphurita, ahata, pratyahata
  - Raga grammar and phrase structure

- [ ] **Neural Audio Synthesis overview** *(blog posts by Engel / Magenta team)*

---

## Build Log

| # | What I Built | Does It Sound Good? | Notes | Date |
|---|-------------|-------------------|-------|------|
| — | — | — | — | — |
