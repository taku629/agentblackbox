# Paper-track submission checklist (Kaggle UI)

Writeups can't be created via the public API — do these steps in the browser.

## Assets (all ready)

- Writeup text: `/home/takumu/kaggle/arc-prize-2026-paper-track/writeup.md`
  (**1496 words**, limit 1500 — verified)
- Cover image: `/home/takumu/kaggle/arc-agi-3/cover_dual.png`
- Public notebook (Project Link):
  `https://www.kaggle.com/code/takumuhata/arc-prize-2026-agents-ocean-forge`
- Documented submissions:
  - Forge ref `56378196` — public score **0.08** (in writeup)
  - OCEAN ref `56408885` — public score **0.14** (in writeup)
- OCEAN v14 auto-submits at 00:10 UTC via
  `/home/takumu/kaggle/arc-agi-3/watch_and_submit.py` → `watch_submit.log`;
  kernel `takumuhata/arc-prize-2026-arc-agi-3-starter` v14

## Steps

1. Go to `https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups`
   (or the competition's "Writeups" tab → "New Writeup").
2. Title: `Online World-Model Induction and Hybrid RL–Search for ARC-AGI-3`.
3. Subtitle: `Inducing each game's rules from pixels — then planning over the induced model`.
4. Paste the `writeup.md` content (skip the `#`/`##` title+subtitle lines
   since they're entered in the fields above; start from `## Overview`).
5. Upload `cover_dual.png` as the cover/banner image (Media Gallery).
6. In Project Links, add the public notebook URL above.
7. Select track: **Main Track**.
8. Publish.

## Notes

OCEAN v13 scored **0.14** (ref 56408885) — already reflected in the writeup,
which is submittable as-is. If the v14 submission lands before you publish,
its ref/score can be added but is not required.
