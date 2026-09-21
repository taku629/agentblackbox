# Online World-Model Induction and Hybrid RL–Search for ARC-AGI-3

## Inducing each game's rules from pixels — then planning over the induced model

## Overview

ARC-AGI-3 replaces static grid prediction with interactive control: each task is a
small video game with hidden rules, observed only through 64×64 color frames
and a handful of action types (four keyboard directions, a select/cycle key,
a coordinate click, and an undo/reset). Evaluation games are unseen, the API
hides all internal state, and scoring rewards *efficiency*: a level in `a`
actions against a human baseline `h` scores `min((h/a)²·100, 115)`, so flailing
is almost as bad as failing. We built two complementary agents — a
training-free online induction system (OCEAN) and a hybrid
RL-plus-symbolic-search system (Forge) — and submitted both to ARC-AGI-3.

## Why the task is hard

Local analysis of the 25 development games reveals extreme heterogeneity:
slide mazes, Sokoban-style pushing, click-to-select puzzles, and
parameter-setting games with no avatar at all. Every game enforces a step
limit displayed only as an on-screen counter. There is no shared "move the
blue square to the flag" semantics — each game's physics, goal, and even
which objects are controllable must be induced online, from pixels, within a
few hundred actions.

## System 1 — OCEAN: training-free online world-model induction

OCEAN treats each level as a system-identification problem:

* **Action-semantics probing.** It issues each available action a few times and
  compares consecutive frames by *pixel-diff motion*: for every color, cells that
  disappeared vs. appeared. The avatar is the color group whose displacement is
  consistent and aligned with the canonical direction of the action probed —
  robust in dense lattices of identical objects where nearest-centroid tracking
  fails.
* **Mechanism classification.** Each keyboard action is classified as a fixed
  quantum move or a *slide* (repeat-until-blocked), estimating the lattice
  quantum from the mode of observed displacements.
* **Edge-blocking wall model.** Maze walls in ARC-AGI-3 sit *between* cells, not
  on them, so a bump is localized to the swept gap between the old and predicted
  footprints; only that gap's colors are learned as blockers. A learned
  edge-block predicate drives BFS over abstract positions.
* **Goal and hazard learning.** The completing action's predicted destination
  reveals the goal color; fatal destinations accumulate hazard counts, but a
  color the avatar has already stood beside safely can never be blamed —
  this gates out the timeout deaths that would otherwise poison floor colors.
  Goals persist across levels; the same machinery carries over after
  level-ups, including the *positions* where earlier levels completed and the
  *colors of objects consumed* just before completion (collect-them-all
  semantics), which bias later-level frontier choice and target selection.
* **Frontier exploration.** With no known goal, BFS over learned edges
  targets nearest unexplored positions — critical under step budgets.
* **Unlock and click fallbacks.** If every keyboard action yields no motion
  (masking always-ticking UI counters), OCEAN tries select/cycle and samples
  clicks on salient objects, and it learns fatal-click colors to avoid repeated
  deaths in sequence puzzles.
* **Effective-object click dwelling.** For click-only games, OCEAN sweeps
  objects small-first; a click producing a substantial non-ticking frame
  change marks its containing object "effective", and effective objects are
  pressed repeatedly — with a per-object press cap so unproductive targets
  stop consuming lives. This alone solved lp85, s5i5 and vc33.

## System 2 — Forge: hybrid RL + symbolic search (submitted)

Forge combines learned and symbolic components under a time budget:

* **ForgeNet**, a compact CNN with channel/spatial attention (CBAM) over the
  last ten 64×64 frames, predicts action values and is trained online with
  prioritized experience replay.
* **A multi-solver cascade** — Dijkstra transfer across levels, beam search,
  IDA*, bounded BFS, A*, and MCTS — runs in a *background thread* so action
  selection never blocks; solutions are replayed when found.
* **Test-time training and a persistent archive** carry learned models and
  memories across levels and games within the 9-hour window.
* **Engineering details that mattered**: consistent ACTION6 coordinates,
  keeping undo out of learned transition maps, and background-solver
  timeouts scaled to the remaining budget.

Forge was submitted to ARC-AGI-3 (submission ref 56378196, GPU kernel,
`submission.parquet` output), scoring **0.08** on the public leaderboard over
the 110 unseen evaluation games. OCEAN — no training, no GPU, no
privileged state — scores **0.14** on the same hidden set (ref 56408885),
nearly doubling the hybrid.

## Results

| Agent | Games with ≥1 level solved | Levels solved | Mean game score |
|---|---|---|---|
| Random | 1/25 | 1 | ~0% |
| OCEAN (training-free) | 11/25 | 14 | ~6.5% local / **0.14 public** (ref 56408885) |
| Oracle solver (privileged state, dev only) | 10/25 | 13 | — |
| Forge (competition submission) | ref 56378196 | — | 0.08 public (110 unseen games) |

Local numbers are on the 25 development games at an 8,000-action budget.
OCEAN's wins include a maximum-score solve (ar25 level 1 in 23
actions for 115.0 — faster than the human baseline; it also completes level
2, and m0r0 in 56 for 28.7), plus cd82, sp80, tn36, cn04, the click-driven
lp85 (two levels), s5i5, vc33 (two levels), and the formerly-unsolved lf52
and r11l —
all from the same induction path with no game-specific code. Two
representation fixes drove
most of the gain: marking a blocked node at the cell *past where the avatar
actually stopped* rather than the predicted slide destination (ar25:
~700 → 36 actions), and gating avatar tracking to components near the
predicted or last-known position — flickering UI elements sharing the
avatar's palette otherwise hijack centroid tracking (ar25: 36 → 23, a
perfect score). A third fix unlocked multi-level play: state for the new
level must be reinitialized *before* the completed-level counter is stored,
otherwise every later-level frame looks like a fresh transition and the map
is wiped each step — correcting the ordering took vc33 from 1 to 2 levels.

## Failure analysis — where universality breaks

The failures are more informative than the wins:

* **Avatar tracking corruption.** Multi-color sprites merged with static
  decorations hijack centroid tracking; motion-anchored union-mask tracking
  fixes most cases, but thin (1px/turn) movers still occasionally lose lock.
* **Slide-family mazes.** Games where every piece slides until blocked — and
  *all* pieces must reach exits — defeat single-avatar navigation. These need
  joint-state search, which our privileged solver does but the frame-only agent
  cannot yet bootstrap.
* **No-avatar mechanics.** Parameter-setting and pure click-sequence games have
  no moving object to anchor on; OCEAN's unlock/click fallbacks reach them but
  rarely solve them.
* **Non-stationary geometry.** Some mazes are *turnstiles* whose wall arms
  rotate every action — a static learned wall-map is wrong by construction;
  the model must induce wall *dynamics*, not just wall positions.
* **Goals hidden in the avatar's palette.** Several games render the exit in
  the avatar's own colors; OCEAN excludes avatar-colored objects from
  targeting, so the goal is literally invisible to its object model and only
  frontier exploration can stumble onto it — often too slowly under a
  per-life step budget (~130 actions).
* **Shared-color poisoning.** When two semantically different objects share a
  color — e.g. a collectible key and an animated timer gauge — one bump into
  the gauge teaches "this color is a wall" and blinds the agent to the keys.
  Consumed-object tracking and tick-detection mitigate but do not cure it.
* **Collect-all semantics.** Games requiring every instance of an object type
  to be consumed expose a gap in single-target navigation: the agent learns
  *that* touching a key works but has no model of *how many remain*.
* **Efficiency vs. discovery.** The quadratic scoring means a 10×-slower solve
  is worth ~1% of a clean one. Learned transition models are essential —
  random-walk discovery is scored near zero even when it eventually succeeds.

## Discussion

Two lessons generalize beyond ARC — and explain *why* the design works, not
just that it does. First, *probe-driven identification* — spending a few
actions to establish "which action moves which thing, how" — converts
unstructured exploration into standard planning over an induced model; the
task's physics are simple but *unknown*, so the bottleneck is
identification, not search depth. Second, *the right representation is
task-induced, not fixed*: OCEAN's symbolic edge model works where games are
spatial; Forge's learned value function covers cases where symbols are
brittle. The failures cluster exactly where the induced representation loses
information — avatar-palette goals, collect-all counting, rotating geometry —
evidence that the limits are representational, not algorithmic. A unified
system would classify each game's induced model family online and route to
the matching solver — the natural next step.

## Limitations and future work

Neither agent yet solves the majority of development games. Planned work:
joint multi-object state search for slide families; learned click-effect
models for sequence puzzles; and tighter integration of Forge's background
solver with OCEAN's induction front-end, which is far cheaper than pixel-level
search.

## Reproducibility

All code is public in the linked notebook
(`kaggle.com/code/takumuhata/arc-prize-2026-agents-ocean-forge`): both agents,
the local evaluation harness (`run_local.py`), and the full result tables.
Both systems run with no internet access and no external model calls at
evaluation time.
