#!/usr/bin/env bash
# harvest-anki.sh — pull un-made Anki cards flagged under "## Anki" in Cornell
# notes, zettels, and library notes into one backlog you can knock out.
#
# Convention: in a note, flag cards as checkboxes under a "## Anki" heading:
#     ## Anki
#     - [ ] Q: what is the online-softmax trick? / A: running max + rescale
# This script lists every still-unmade ("- [ ]") flag with its source file.
#
# Usage: bash scripts/harvest-anki.sh            # print to stdout
#        bash scripts/harvest-anki.sh > anki/backlog.md
set -euo pipefail
cd "$(dirname "$0")/.."
DIRS=(seminars zettel lecture-myself library reviews capstones)

echo "# Anki harvest — $(date +%F)"
echo "*Unmade card flags (\`- [ ]\` with content, under a \`## Anki\` heading). Make them, then check them off in the source note.*"
echo
while IFS= read -r f; do
  awk '
    /^#{2,}[[:space:]]/ { in_anki = ($0 ~ /^##[[:space:]]+Anki/); next }
    in_anki && /^- \[ \][[:space:]]*[^[:space:]]/ { print "- `" FILENAME "`: " substr($0, 7) }
  ' "$f"
done < <(grep -rl '^## Anki' "${DIRS[@]}" 2>/dev/null || true)
echo
echo "> Seeded reserve deck: **anki/decks/bricks-starter.tsv** (668 cards from the brick library) — import that first, activate per topic."
