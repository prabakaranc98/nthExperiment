# Anki — Spaced Repetition

Retrieval practice + spacing is the single best-evidenced learning intervention — and the antidote to the *fluency illusion* you get from re-reading dense bricks. This is the system that turns the library from exposition you recognize into knowledge you can recall cold.

**Deck naming:** `nthexp::[track]::[topic]` — e.g. `nthexp::frontier-ai::attention`, `nthexp::bricks::mistakes`.

---

## Start now (don't wait for Cornell notes)

A **668-card reserve deck is already seeded** from the brick library — every brick's one-liner ("Define / recall: X") and common mistake ("Common mistake — X?") is a card. It's derived, so regenerate it whenever the bricks grow:

```bash
python3 scripts/anki-bricks.py      # -> anki/decks/bricks-starter.tsv (668 cards)
```

1. **Import** `anki/decks/bricks-starter.tsv` into Anki (File → Import; it's tab-separated, `#tags` in column 3, HTML on).
2. **Don't activate all 668 at once.** Suspend the deck, then un-suspend ~20–30 cards per topic *as you study that topic* — keep daily review under ~20 min.
3. Regenerate any time the bricks grow (the `.tsv` is derived, not hand-maintained).

---

## The ongoing loop: harvest flags → cards

When reading, flag cards in the note under a `## Anki` heading (the [Cornell template](../_templates/cornell-note.md) already has this section):

```markdown
## Anki
- [ ] Q: what does the √dₖ scaling in attention prevent? / A: softmax saturation → dead gradients
```

Then pull all unmade flags into one backlog:

```bash
bash scripts/harvest-anki.sh            # list every "- [ ]" flag with its source
bash scripts/harvest-anki.sh > anki/backlog.md
```

Make the cards, then check the box (`- [x]`) in the source note so it stops showing up.

---

## Decks

| Deck | Cards | Source | Notes |
|------|-------|--------|-------|
| `nthexp::bricks` + `::mistakes` | 668 | [bricks-starter.tsv](decks/bricks-starter.tsv) | Seeded reserve; activate per topic |

---

## Rhythm

- **Daily:** 15–20 min review — Anki schedules; you just show up. (Surfaced in [now.md](../now.md).)
- **While reading:** flag cards immediately under `## Anki`; harvest weekly.
- **Weekly review:** log cards added / reviewed / retention (the [week-review template](../_templates/week-review.md) has the row).
- Don't add more than you can review in 20 min/day. When a deck feels solid, stop adding and let it mature.
