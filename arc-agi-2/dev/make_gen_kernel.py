"""Build the generation-lever kernel dir from the production perfpatch notebook.

    python3 make_gen_kernel.py --verdict /tmp/evalscan/select_verdict.json [--slack 0.4] [--with-select]
    python3 make_gen_kernel.py --levers '{"g1": "empty_input"}'
    python3 make_gen_kernel.py --selftest

Levers come from gen_boost.decide(verdict["diagnostics"], slack) unless forced.
If decide() turns nothing on, nothing is built. --with-select also applies the
selection patch (verdict winner). Writes ../../submit_ag2_genboost/ -- it only
BUILDS the directory; pushing is a separate, manual step.

Edits to the base notebook (anchor-asserted, everything else byte-identical):
  1. new cell     %%writefile gen_boost.py
  2. arc_solver   GEN_LEVERS constant; the decode block of worker() is replaced
                  by one gen_boost.decode_task(...) call (identical output when
                  all levers are off -- see gen_boost.py --selftest)
"""
import argparse
import ast
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_boost                      # noqa: E402
import make_select_kernel as msk      # noqa: E402

OUT_DIR = os.path.join(msk.REPO, "submit_ag2_genboost")
TITLE, SLUG = "ARC AGI2 GenBoost", "arc-agi2-genboost"

BLOCK_START = "        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)\n"
BLOCK_END = ("        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2\n"
             "        print(f\"[Rank {rank}] allocated {memory_allocated}MB for inference\")")
CONST_ANCHOR = "EOS_ID = 15\n"
CALL = '''        import gen_boost
        gen_boost.decode_task(
            dict(model=model, tokenizer=tokenizer, formatter=formatter, ArcDataset=ArcDataset,
                 inference_turbo_dfs=inference_turbo_dfs, calc_scores=calc_scores, arc_token_ids=_arc_token_ids,
                 EOS_ID=EOS_ID, PAD_ID=PAD_ID, max_new_tokens=max_new_tokens, max_score=max_score,
                 max_seq_length=max_seq_length, dir_outputs=dir_outputs, rank=rank, key=key),
            puzzle_ds_multi, start_time, end_time, levers=GEN_LEVERS, tasks_left=max(queue.qsize() - 4, 0))

'''
CTX_NAMES = ["model", "tokenizer", "formatter", "max_new_tokens", "max_score", "max_seq_length", "dir_outputs",
             "rank", "key", "puzzle_ds_multi", "start_time", "end_time", "queue"]


def patch_solver(text, levers):
    for anchor in (BLOCK_START, BLOCK_END, CONST_ANCHOR):
        assert text.count(anchor) == 1, f"arc_solver anchor matched {text.count(anchor)}x: {anchor.strip()[:60]!r}"
    a, b = text.index(BLOCK_START), text.index(BLOCK_END)
    assert a < b
    text = text[:a] + CALL + text[b:]
    return text.replace(CONST_ANCHOR, CONST_ANCHOR + f"\nGEN_LEVERS = {levers!r}\n")


def build(levers, nb=None, meta=None):
    unknown = set(levers) - set(gen_boost.LEVERS_OFF)
    assert not unknown, f"unknown levers {unknown}"
    assert levers.get("g1", "off") in ("off", "empty_input", "empty_view")
    full = dict(gen_boost.LEVERS_OFF, **levers)
    if nb is None:
        nb, meta = json.load(open(msk.BASE_NB)), json.load(open(msk.BASE_META))
    cells = nb["cells"]
    i = msk.find(cells, "%%writefile arc_solver.py")
    msk.set_src(cells[i], patch_solver(msk.src(cells[i]), full))
    gb = copy.deepcopy(cells[i])
    msk.set_src(gb, "%%writefile gen_boost.py\n" + open(os.path.join(HERE, "gen_boost.py")).read())
    cells.insert(i, gb)
    owner = meta["id"].split("/")[0]
    meta.update(id=f"{owner}/{SLUG}", title=TITLE, code_file=f"{SLUG}.ipynb")
    return nb, meta


def selftest():
    base = json.load(open(msk.BASE_NB))["cells"]
    levers = {"g1": "empty_input", "faircap": True}
    nb, meta = build(levers)
    assert meta["id"].endswith("/" + SLUG)
    solver = msk.src(nb["cells"][msk.find(nb["cells"], "%%writefile arc_solver.py")]).split("\n", 1)[1]
    tree = ast.parse(solver)
    ns = {}
    exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "GEN_LEVERS"], []),
                 "x", "exec"), ns)
    assert ns["GEN_LEVERS"] == dict(gen_boost.LEVERS_OFF, **levers)
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    names = [ast.unparse(c.func) for c in calls]
    assert names.count("gen_boost.decode_task") == 1
    assert "inference_turbo_dfs" not in names and "calc_scores" not in names, "old decode block must be gone"
    # every name handed to decode_task is bound in worker() (argument or assignment) before the call line
    call = next(c for c in calls if ast.unparse(c.func) == "gen_boost.decode_task")
    bound = {a.arg for a in worker.args.args}
    for n in ast.walk(worker):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.lineno < call.lineno:
            bound.add(n.id)
    missing = [n for n in CTX_NAMES if n not in bound]
    assert not missing, f"not bound before the call: {missing}"
    module_defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} | \
                  {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert {"inference_turbo_dfs", "calc_scores", "_arc_token_ids", "EOS_ID", "PAD_ID", "GEN_LEVERS"} <= module_defs
    # TTT / budget code before the block and everything after it are untouched
    orig = msk.src(base[msk.find(base, "%%writefile arc_solver.py")]).split("\n", 1)[1]
    a, b = orig.index(BLOCK_START), orig.index(BLOCK_END)
    assert solver.replace(f"\nGEN_LEVERS = {ns['GEN_LEVERS']!r}\n", "") == orig[:a] + CALL + orig[b:]
    # the gen_boost cell is this file's sibling verbatim and compiles; other cells byte-identical
    gb = msk.src(nb["cells"][msk.find(nb["cells"], "%%writefile gen_boost.py")]).split("\n", 1)[1]
    assert gb == open(os.path.join(HERE, "gen_boost.py")).read()
    compile(gb, "gen_boost.py", "exec")
    kept = [msk.src(c) for c in nb["cells"] if not msk.src(c).startswith(("%%writefile gen_boost.py", "%%writefile arc_solver.py"))]
    assert kept == [msk.src(c) for c in base if not msk.src(c).startswith("%%writefile arc_solver.py")]
    # chained with the selection patch
    nb2, meta2 = msk.build("kgmon+priors")
    nb2, meta2 = build(levers, nb2, meta2)
    starts = [msk.src(c).split("\n", 1)[0] for c in nb2["cells"] if msk.src(c).startswith("%%writefile")]
    assert starts == ["%%writefile arc_loader.py", "%%writefile arc_decoder.py", "%%writefile select_v2.py",
                      "%%writefile gen_boost.py", "%%writefile arc_solver.py", "%%writefile starter.py"], starts
    assert "SELECT_STRATEGY = 'kgmon+priors'" in msk.src(nb2["cells"][-1])
    # refusal paths
    for bad in ({"g9": True}, {"g1": "always"}):
        try:
            build(bad)
            raise SystemExit("bad levers accepted")
        except AssertionError:
            pass
    print("selftest ok: worker() decode block replaced by exactly one decode_task call with all inputs bound, "
          "rest of arc_solver and all other cells byte-identical to perfpatch; chains with the selection patch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict")
    ap.add_argument("--slack", type=float)
    ap.add_argument("--levers")
    ap.add_argument("--with-select", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    verdict = json.load(open(a.verdict)) if a.verdict else {}
    if a.levers:
        levers = json.loads(a.levers)
    elif verdict:
        d = gen_boost.decide(verdict["diagnostics"], a.slack)
        for line in d["why"]:
            print(" -", line)
        if not d["changed"]:
            sys.exit("decide(): no generation lever is supported by the diagnostics; nothing built.")
        levers = {k: v for k, v in d["levers"].items() if v != gen_boost.LEVERS_OFF[k]}
    else:
        ap.error("give --verdict or --levers")
    nb = meta = None
    if a.with_select:
        if not verdict.get("winner"):
            sys.exit("--with-select needs a verdict with a winning strategy")
        nb, meta = msk.build(verdict["winner"], verdict.get("ranker"))
    nb, meta = build(levers, nb, meta)
    print("built", msk.write(nb, meta, OUT_DIR), "levers:", levers, "(not pushed)")
