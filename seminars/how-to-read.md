# How to Read a Paper

> A paper is not a fixed object — it changes shape depending on who's holding it. The same eight pages are a *lineage* to a historian, a *claim under trial* to a reviewer, a *system to build* to an engineer, and a *liability* to a practitioner. Don't read a paper once. Read it two or three times through different eyes, and let the angles disagree — that friction is where understanding actually forms.

---

## Step 0 — First Contact (before you pick an angle)

Fast orientation pass: title, abstract, figures, conclusion. Ten minutes. Answer one question:

> *Why does this paper matter to me right now?*

If it survives:
- **Second pass** — follow the figures, mark unknown references, find the one sentence that is the actual contribution
- **Third pass** (earned, not default) — reconstruct the method as if re-deriving it yourself

Most papers get one angle. Tentpole papers earn several.

---

## The Eight Angles

### 🏺 Archaeologist — read for lineage and buried assumptions
Understand why the paper exists before judging what it does. What did the state of the art look like the night before it was written? Trace citations backward — every paper is one move in a longer conversation. Hunt for unstated assumptions: the "standard" benchmark everyone inherited, the modeling choice treated as obvious. Often the real contribution is a quiet *reframing*, not the headline number.

### 🔍 Reviewer — read adversarially
State the central claim in one sentence, then find the weakest link. Which experiment would *you* have run that they left out? Are the baselines fair? What result would falsify the thesis — did the authors actually test it, or avoid it? A paper you cannot criticize is a paper you have only been persuaded by.

### 🧬 Core Researcher — read for the frontier it sits on
Strip to mechanism. What does it explain or unlock that nothing before it could? Not *"is this useful?"* — *"is this true, and what does it make newly possible?"* The best souvenir is not a summary. It's a sharper open question you now genuinely want to chase.

### 🌿 Applied / Interdisciplinary Researcher — read for transfer
Never read in isolation. Ask: *where else does this mechanism live?* Take "policy optimization" and try it in recommendation systems, biology, markets, scheduling. When a transfer feels alive, log it as an idea. Curiosity compounds fastest here — you're carrying a proven tool into unproven territory.

### 🛠️ Engineer — read for what you'd actually have to build
Two modes:
- **Enabling systems** — compute, memory footprint, data pipeline, failure modes at scale. What is the minimal reproduction that proves it runs at all?
- **Decision engineering** — how does this become a decision-making system? Where do modeling, inference, reasoning, and optimization plug in?

### 🏭 Industrial Practitioner — read for survival in the wild
Forget the leaderboard. Does this survive messy real-world data, latency budgets, cost ceilings, distribution shift, and a team maintaining it at 3am? Is the 2% accuracy gain worth 10× complexity? *Would I actually bet a product — or a paycheck — on this?*

### 💻 Hacker — read for the exploit
Method section, one key figure, the repo — then build. Smallest scrappiest version, tonight. Break it, bend it, feel where it's brittle. Can you make it 10× simpler? What happens when you feed it something it wasn't designed for? Understanding through the fingertips.

### 🗣️ Teacher — read in order to explain
Final exam for every other angle: can you teach this to someone who has never seen it — no jargon, no hand-waving? Wherever you stumble is the exact edge of your understanding. Write the Cornell note. Give yourself the personal seminar. Teaching is simultaneously the *assessment* and the *artifact*.

---

## How to Sequence the Angles

Don't run all eight on every paper — that's how you read four papers a year.

Use the orientation pass to choose 2–3 angles that fit *why this paper matters to you.*

**Default arc:**
`Archaeologist → Reviewer → (Core or Applied) → Engineer or Practitioner → Teacher`

| Paper type | Angles to use |
|---|---|
| Skim / context only | Archaeologist + Reviewer |
| Method paper | Core + Engineer + Teacher |
| Applied / systems paper | Practitioner + Engineer + Hacker |
| Theoretical / foundational | Archaeologist + Core + Reviewer |
| Interdisciplinary | Applied + Core + Teacher |
| Tentpole (seminal) | All of them, across sittings |

---

## Close Every Reading With Three Exits

1. **Steelman** — one sentence, your own words, the strongest version of what the paper claims
2. **Open question** — the sharpest thing you still can't answer
3. **Next action** — deepen it (Core path) or branch it (Applied path)

Then write the Cornell note. A paper read and never written about evaporates. A paper read and *taught* compounds.
