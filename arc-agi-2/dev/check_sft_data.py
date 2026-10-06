"""Quality gate for AGI-2 SFT data: format, contamination against the 120 eval tasks, distribution.

    python3 check_sft_data.py <source> [<source> ...] [--write-exclude exclude.json] [--max-rows N]
    python3 check_sft_data.py --selftest

A source is one of
    tasks:<challenges.json>[:<solutions.json>]   ARC json (public training set)
    jsonl:<file.jsonl[.gz]>                      rows with `full_text` (Nabidnur sft128: chat transcripts)
                                                 or rows with `train` / `test` lists (synthetic curriculum)

Checks
  format         every transcript is exactly what the production formatter emits for the grids it
                 contains (arc_loader.QwenFormatter), uses only the 16 hard-coded tokens, and fits 8192
  contamination  an (input, output) pair is compared with every pair of the 120 evaluation tasks
                 (demo pairs AND test pairs with their solutions) up to the 8 dihedral transforms and
                 any colour permutation. One matching pair marks the whole task. The exclude list is
                 what sft_ag2.py refuses to train on.
  distribution   same / extract / larger share, sequence lengths, pairs per sequence
"""
import gzip
import hashlib
import json
import os
import re
import sys
import types
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
PROD = os.path.join(os.path.dirname(BASE), "submit_ag2_perfpatch", "out")

TOKEN_IDS = {**{str(d): d for d in range(10)}, "\n": 10, "user": 11, "assistant": 12,
             "<|im_start|>": 14, "<|im_end|>": 15}                      # 13 = pad, never in text
TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|user|assistant|\n|\d")
TURN_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.S)
MAX_LEN = 8192


def encode(text):
    """Text -> token ids with the kernel's hard-coded vocabulary. Raises on anything else."""
    pos, out = 0, []
    for m in TOKEN_RE.finditer(text):
        if m.start() != pos:
            raise ValueError(f"character outside the grid vocabulary at {pos}: {text[pos:m.start()][:20]!r}")
        out.append(TOKEN_IDS[m.group(0)])
        pos = m.end()
    if pos != len(text):
        raise ValueError(f"character outside the grid vocabulary at {pos}: {text[pos:pos + 20]!r}")
    return out


def formatter():
    """Production formatter (single source of truth for the transcript format)."""
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=object)
    for d in (HERE, PROD):
        if os.path.exists(os.path.join(d, "arc_loader.py")) and d not in sys.path:
            sys.path.insert(0, d)
    from arc_loader import QwenFormatter
    return QwenFormatter(tokenizer=None)


def to_text(pairs, fmt=None):
    """[(input, output), ...] -> the production training transcript (last pair is the challenge)."""
    fmt = fmt or formatter()
    return fmt.fmt_train([{"input": i, "output": o} for i, o in pairs], last_is_challenge=True)


def parse_text(text):
    """Transcript -> [(input, output), ...]; raises if it is not alternating user / assistant grids."""
    turns = TURN_RE.findall(text)
    if "".join(f"<|im_start|>{r}\n{b}<|im_end|>" for r, b in turns) != text:
        raise ValueError("text is not a clean sequence of <|im_start|>role\\n...<|im_end|> turns")
    if len(turns) % 2 or any(r != ("user", "assistant")[i % 2] for i, (r, _) in enumerate(turns)):
        raise ValueError("turns do not alternate user / assistant")
    grids = []
    for _, body in turns:
        rows = body.split("\n")
        if not body or len({len(r) for r in rows}) != 1 or not all(r.isdigit() for r in rows):
            raise ValueError("turn is not a rectangular digit grid")
        grids.append([[int(ch) for ch in r] for r in rows])
    return [(grids[i], grids[i + 1]) for i in range(0, len(grids), 2)]


def pair_hash(inp, out):
    """Hash of a pair, invariant to the 8 dihedral transforms (applied jointly) and to colour permutation."""
    best = None
    a0, b0 = np.asarray(inp), np.asarray(out)
    for t in range(2):
        a1, b1 = (a0.T, b0.T) if t else (a0, b0)
        for k in range(4):
            a, b = np.rot90(a1, k), np.rot90(b1, k)
            relabel, seq = {}, []
            for g in (a, b):
                seq.append(g.shape)
                for v in g.ravel().tolist():
                    seq.append(relabel.setdefault(v, len(relabel)))
            key = repr(seq)
            if best is None or key < best:
                best = key
    return hashlib.sha1(best.encode()).hexdigest()


def eval_hashes(challenges=None, solutions=None):
    ch = json.load(open(challenges or os.path.join(BASE, "arc-agi_evaluation_challenges.json")))
    so = json.load(open(solutions or os.path.join(BASE, "arc-agi_evaluation_solutions.json")))
    out = {}
    for k, t in ch.items():
        for p in t["train"]:
            out[pair_hash(p["input"], p["output"])] = k
        for i, p in enumerate(t["test"]):
            out[pair_hash(p["input"], so[k][i])] = k
    return out


def iter_source(spec, max_rows=None):
    """-> yields (task_id, pairs, raw_text_or_None)."""
    kind, _, rest = spec.partition(":")
    if kind == "tasks":
        cpath, _, spath = rest.partition(":")
        ch = json.load(open(cpath))
        so = json.load(open(spath)) if spath else {}
        for n, (k, t) in enumerate(ch.items()):
            if max_rows and n >= max_rows:
                return
            pairs = [(p["input"], p["output"]) for p in t["train"]]
            for i, p in enumerate(t["test"]):
                if "output" in p:
                    pairs.append((p["input"], p["output"]))
                elif k in so:
                    pairs.append((p["input"], so[k][i]))
            yield k, pairs, None
    elif kind == "jsonl":
        op = gzip.open if rest.endswith(".gz") else open
        with op(rest, "rt") as f:
            for n, line in enumerate(f):
                if max_rows and n >= max_rows:
                    return
                if not line.strip():
                    continue
                r = json.loads(line)
                tid = str(r.get("task_id", n))
                if "full_text" in r or "text" in r:
                    text = r.get("full_text") or r.get("text")
                    try:
                        yield tid, parse_text(text), text
                    except ValueError as e:
                        yield tid, None, f"UNPARSEABLE: {e}"
                else:
                    pairs = [(p["input"], p["output"]) for p in r["train"]]
                    pairs += [(p["input"], p["output"]) for p in r.get("test", []) if "output" in p]
                    yield tid, pairs, None
    else:
        raise SystemExit(f"unknown source kind in {spec!r} (use tasks: or jsonl:)")


def klass(inp, out):
    a, b = np.asarray(inp).size, np.asarray(out).size
    return "same" if a == b else "extract" if b < a else "larger"


def check(spec, ev=None, max_rows=None, verbose=True):
    ev = ev if ev is not None else eval_hashes()
    fmt = formatter()
    n = bad_parse = bad_format = too_long = 0
    contaminated = defaultdict(set)                    # task id -> eval task ids it overlaps
    classes, lens, npairs, tasks = Counter(), [], [], set()
    first_issue = []
    for tid, pairs, text in iter_source(spec, max_rows):
        n += 1
        tasks.add(tid)
        if pairs is None:
            bad_parse += 1
            if len(first_issue) < 3:
                first_issue.append(f"{tid}: {text}")
            continue
        if text is not None:
            try:
                ok = to_text(pairs, fmt) == text
            except Exception:                           # noqa: BLE001
                ok = False
            if not ok:
                bad_format += 1
                if len(first_issue) < 3:
                    first_issue.append(f"{tid}: transcript differs from the production formatter output")
        ntok = sum(np.asarray(i).size + len(i) - 1 + np.asarray(o).size + len(o) - 1 + 8 for i, o in pairs)
        lens.append(ntok)
        too_long += ntok > MAX_LEN
        npairs.append(len(pairs))
        classes[klass(*pairs[-1])] += 1
        for i, o in pairs:
            h = pair_hash(i, o)
            if h in ev:
                contaminated[tid].add(ev[h])
    res = {"source": spec, "rows": n, "tasks": len(tasks), "unparseable": bad_parse, "format_mismatch": bad_format,
           "over_8192_tokens": int(too_long), "contaminated_tasks": {k: sorted(v) for k, v in sorted(contaminated.items())},
           "classes": dict(classes), "issues": first_issue,
           "tokens_median": float(np.median(lens)) if lens else 0.0,
           "tokens_p95": float(np.percentile(lens, 95)) if lens else 0.0,
           "pairs_median": float(np.median(npairs)) if npairs else 0.0}
    res["ok"] = bad_parse == 0 and bad_format == 0
    if verbose:
        tot = max(sum(classes.values()), 1)
        print(f"\n== {spec}")
        print(f"  rows {n} from {len(tasks)} tasks; unparseable {bad_parse}; format mismatch {bad_format}; "
              f"longer than {MAX_LEN} tokens {int(too_long)} (the trainer drops leading demos to fit)")
        print(f"  tokens per sequence: median {res['tokens_median']:.0f}, p95 {res['tokens_p95']:.0f}; "
              f"pairs per sequence: median {res['pairs_median']:.0f}")
        print("  class of the challenge pair: " + "  ".join(f"{k} {v} ({v / tot * 100:.0f}%)" for k, v in sorted(classes.items())))
        if contaminated:
            ex = "; ".join(f"{k}->{','.join(v)}" for k, v in list(res["contaminated_tasks"].items())[:5])
            print(f"  CONTAMINATED: {len(contaminated)} task(s) share a pair with the evaluation set ({ex})")
        else:
            print("  contamination: none (no pair matches an evaluation pair under rotation/reflection/recolouring)")
        for msg in first_issue:
            print("  issue:", msg)
    return res


def selftest():
    import tempfile
    assert encode("<|im_start|>user\n01\n23<|im_end|><|im_start|>assistant\n9<|im_end|>") == \
        [14, 11, 10, 0, 1, 10, 2, 3, 15, 14, 12, 10, 9, 15]
    for bad in ("<|im_start|>user\n0 1<|im_end|>", "abc"):
        try:
            encode(bad)
            raise SystemExit("bad text encoded")
        except ValueError:
            pass
    pairs = [([[0, 1], [2, 3]], [[3]]), ([[5, 5]], [[5, 5], [0, 0]])]
    text = to_text(pairs)
    assert parse_text(text) == pairs and text.count("<|im_end|>") == 4
    for bad in (text + "x", text.replace("assistant", "user", 1), text.replace("01\n23", "01\n2")):
        try:
            parse_text(bad)
            raise SystemExit("malformed transcript parsed")
        except ValueError:
            pass
    # invariances of the contamination hash
    a, b = np.array([[1, 2, 0], [0, 2, 2]]), np.array([[2, 2], [1, 0]])
    h = pair_hash(a, b)
    perm = np.array([7, 3, 9, 0, 1, 2, 4, 5, 6, 8])
    assert pair_hash(np.rot90(a), np.rot90(b)) == h and pair_hash(a.T, b.T) == h and pair_hash(perm[a], perm[b]) == h
    assert pair_hash(perm[np.rot90(a.T, 3)], perm[np.rot90(b.T, 3)]) == h
    assert pair_hash(a, b + 0 * b + np.array([[0, 0], [0, 1]])) != h and pair_hash(b, a) != h
    assert pair_hash(np.rot90(a), b) != h                                # transforms must be joint

    ev = eval_hashes()
    ch = json.load(open(os.path.join(BASE, "arc-agi_evaluation_challenges.json")))
    so = json.load(open(os.path.join(BASE, "arc-agi_evaluation_solutions.json")))
    k0 = sorted(ch)[0]
    with tempfile.TemporaryDirectory() as d:
        # a clean task, a disguised eval DEMO pair, a disguised eval TEST pair
        clean = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}
        p = ch[k0]["train"][0]
        leak1 = {"train": [{"input": perm[np.rot90(np.asarray(p["input"]))].tolist(),
                            "output": perm[np.rot90(np.asarray(p["output"]))].tolist()}], "test": []}
        leak2 = {"train": [{"input": np.asarray(ch[k0]["test"][0]["input"]).T.tolist(),
                            "output": np.asarray(so[k0][0]).T.tolist()}], "test": []}
        tj = os.path.join(d, "t.json")
        json.dump({"clean000": clean, "leak0001": leak1, "leak0002": leak2}, open(tj, "w"))
        r = check("tasks:" + tj, ev, verbose=False)
        assert r["contaminated_tasks"] == {"leak0001": [k0], "leak0002": [k0]} and r["ok"], r
        # transcripts: exact, wrong separator, garbage
        jl = os.path.join(d, "s.jsonl.gz")
        with gzip.open(jl, "wt") as f:
            f.write(json.dumps({"task_id": "good", "full_text": text}) + "\n")
            f.write(json.dumps({"task_id": "spaces", "full_text": text.replace("01\n23", "0 1\n2 3")}) + "\n")
            f.write(json.dumps({"task_id": "leak", "full_text": to_text([(leak1["train"][0]["input"],
                                                                         leak1["train"][0]["output"])])}) + "\n")
            f.write(json.dumps({"task_id": "syn", "train": clean["train"], "test": [{"input": [[1]], "output": [[1]]}]}) + "\n")
        r = check("jsonl:" + jl, ev, verbose=False)
        assert r["rows"] == 4 and r["unparseable"] == 1 and not r["ok"] and list(r["contaminated_tasks"]) == ["leak"], r
        assert check("jsonl:" + jl, ev, max_rows=1, verbose=False)["ok"]
    print("selftest ok: encoder matches the hard-coded ids, transcripts round-trip through the production formatter, "
          "the contamination hash is invariant to rotation/reflection/recolouring and catches disguised eval demo "
          "and test pairs")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--selftest":
        selftest()
        sys.exit(0)
    max_rows = int(a[a.index("--max-rows") + 1]) if "--max-rows" in a else None
    out = a[a.index("--write-exclude") + 1] if "--write-exclude" in a else None
    specs = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--max-rows", "--write-exclude"))]
    ev = eval_hashes()
    print(f"evaluation reference: {len(ev)} distinct pairs from 120 tasks")
    exclude, ok = {}, True
    for spec in specs:
        r = check(spec, ev, max_rows)
        ok &= r["ok"]
        exclude[spec] = sorted(r["contaminated_tasks"])
    n_ex = sum(len(v) for v in exclude.values())
    if out:
        json.dump(exclude, open(out, "w"), indent=1)
        print(f"\nexclude list ({n_ex} task ids) written to {out}")
    print("\nRESULT:", "format OK" if ok else "FORMAT PROBLEMS -- do not train on the flagged source",
          "|", f"{n_ex} contaminated task(s) to exclude")
    sys.exit(0 if ok else 1)
