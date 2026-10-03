"""Offline selection-strategy benchmark for AGI-2 decoded candidates.

Loads the pickles an evalscan kernel run saved (inference_outputs/*.bz2-free
files are bz2-compressed pickles), reconstructs per-test-input candidate
lists exactly like arc_decoder.ArcDecoder, then scores selection strategies
against the evaluation solutions.

Usage:
    python3 select_bench.py <inference_outputs_dir> [dsl_preds.pkl]

Reports, per strategy: score = sum over test inputs of (1/#gold for its task)
when the correct grid is in the first 2 selected guesses — i.e. the local
eval score that mirrors Kaggle's metric.
"""
import bz2
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

BASE = "/home/ubuntu/repos/arc-prize-2026-agent-work/arc-agi-2"


def hashable(g):
    return tuple(map(tuple, g))


def load_candidates(store):
    """base_key -> {subkey.out{i}: sample}  (same shape as ArcDecoder)."""
    decoded = {}
    for fname in sorted(os.listdir(store)):
        path = os.path.join(store, fname)
        if not os.path.isfile(path):
            continue
        try:
            with bz2.BZ2File(path, "r") as f:
                outputs = pickle.load(f)
        except Exception:
            with open(path, "rb") as f:
                outputs = pickle.load(f)
        base_key = fname.split(".")[0]
        decoded.setdefault(base_key, {})
        for i, sample in enumerate(outputs):
            decoded[base_key][f"{fname}.out{i}"] = sample
    return decoded


def aug_family(subkey):
    """Recover the geometric family from a subkey name.

    Subkeys look like '<task>_<test>.<mod chain>' where the mod chain encodes
    transpose/rot90 ops; the first op token distinguishes rotation-family
    (no 'transpose') vs transpose-family.
    """
    mods = subkey.split(".")[1:] if "." in subkey else []
    geo = ".".join(m for m in mods if "permute" not in m)
    return geo if geo else "base"


def vote_groups(guesses):
    """grid-hash -> list of samples producing that grid."""
    groups = defaultdict(list)
    for g in guesses.values():
        groups[hashable(g["solution"])].append(g)
    return groups


def kgmon_rank(groups):
    scored = []
    for h, gs in groups.items():
        inf = len(gs)
        aug = np.mean([np.mean(g["score_aug"]) for g in gs])
        scored.append((inf - aug, gs[0]["solution"]))
    return [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)]


def probmul_rank(groups, baseline=3):
    scored = []
    for h, gs in groups.items():
        inf = np.sum([baseline - g["beam_score"] for g in gs])
        aug = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in gs])
        scored.append((inf + aug, gs[0]["solution"]))
    return [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)]


def diverse_rank(groups):
    """kgmon top-1, then best-scoring grid from a different aug family."""
    fam_votes = defaultdict(lambda: defaultdict(int))
    for h, gs in groups.items():
        for g in gs:
            pass  # subkey info lives in dict keys; see diverse_rank2
    return kgmon_rank(groups)


def diverse_rank2(samples):
    """Attempt-2 diversity using subkey names.

    samples: {subkey.out{i}: sample}. Group by grid; attempt_1 = kgmon winner.
    For attempt_2 prefer the highest-kgmon group whose supporting subkeys'
    dominant aug family differs from the winner's.
    """
    groups = defaultdict(list)
    for subkey, g in samples.items():
        groups[hashable(g["solution"])].append((subkey, g))
    scored = []
    for h, pairs in groups.items():
        inf = len(pairs)
        aug = np.mean([np.mean(g["score_aug"]) for _, g in pairs])
        fams = defaultdict(int)
        for sk, _ in pairs:
            fams[aug_family(sk.split(".")[0])] += 1
        dom_fam = max(fams, key=fams.get)
        scored.append((inf - aug, pairs[0][1]["solution"], dom_fam))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []
    out = [scored[0][1]]
    win_fam = scored[0][2]
    for sc, sol, fam in scored[1:]:
        if fam != win_fam:
            out.append(sol)
            break
    if len(out) < 2:
        out.extend(s for _, s, _ in scored[1:])
    return out


def main():
    store = sys.argv[1]
    ch = json.load(open(f"{BASE}/arc-agi_evaluation_challenges.json"))
    so = json.load(open(f"{BASE}/arc-agi_evaluation_solutions.json"))
    decoded = load_candidates(store)
    print(f"loaded {len(decoded)} base keys from {store}")

    n_tests = {k: len(v) for k, v in so.items()}
    gold = {}
    for k, replies in so.items():
        for i, r in enumerate(replies):
            gold[f"{k}_{i}"] = np.asarray(r)

    # task class: compare input vs gold output area
    tclass = {}
    for k, replies in so.items():
        for i, r in enumerate(replies):
            ti = ch[k]['test'][i]['input']
            ai, ao = len(ti)*len(ti[0]), len(r)*len(r[0])
            tclass[f"{k}_{i}"] = ('same' if ao == ai
                                  else 'extract' if ao < ai else 'larger')

    # oracle headroom: correct grid exists anywhere in candidates?
    oracle = 0.0
    has_any = 0.0
    ocl = defaultdict(float)
    tcl = defaultdict(float)
    for bk, g in gold.items():
        task = bk.split("_")[0]
        w = 1 / n_tests[task]
        tcl[tclass[bk]] += w
        cands = decoded.get(bk, {})
        grids = {hashable(s["solution"]) for s in cands.values()}
        if len(grids):
            has_any += w
        if hashable(g) in grids:
            oracle += w
            ocl[tclass[bk]] += w
    print(f"oracle (correct grid present in candidates): {oracle:.2f}")
    for cl in sorted(tcl):
        print(f"  class {cl:7s}: oracle {ocl[cl]:.2f} / {tcl[cl]:.2f} possible")
    print(f"coverage (any candidate produced): {has_any:.2f} "
          f"({len(decoded)}/{len(gold)} test inputs)")

    strategies = {
        "kgmon (prod)": lambda smp: kgmon_rank(vote_groups(smp)),
        "probmul_3": lambda smp: probmul_rank(vote_groups(smp)),
        "kgmon+diverse2": diverse_rank2,
    }

    for name, fn in strategies.items():
        score = 0.0
        scl = defaultdict(float)
        for bk, g in gold.items():
            task = bk.split("_")[0]
            w = 1 / n_tests[task]
            order = fn(decoded.get(bk, {}))
            if any(np.array_equal(g, cand) for cand in order[:2]):
                score += w
                scl[tclass[bk]] += w
        print(f"{name}: {score:.2f}/120  " +
              " ".join(f"[{cl} {scl[cl]:.1f}]" for cl in sorted(tcl)))


if __name__ == "__main__":
    main()
