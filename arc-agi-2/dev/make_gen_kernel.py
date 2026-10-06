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


BASE_MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


def model_path(source):
    """'owner/slug/Framework/variation/version' -> its mount path in a kernel."""
    owner, slug, framework, variation, version = source.split("/")
    return f"/kaggle/input/models/{owner}/{slug}/{framework.lower()}/{variation}/{version}"


def swap_model(nb, meta, source):
    """Point the kernel at another DROP-IN checkpoint (same architecture and the same
    16-token grid vocabulary; verify the directory with check_swap_model.py first)."""
    old, new = model_path(BASE_MODEL_SOURCE), model_path(source)
    hits = 0
    for c in nb["cells"]:
        s = msk.src(c)
        if old in s:
            hits += s.count(old)
            msk.set_src(c, s.replace(old, new))
    assert hits == 1, f"expected the base model path exactly once, found {hits}"
    assert meta["model_sources"] == [BASE_MODEL_SOURCE], meta["model_sources"]
    meta["model_sources"] = [source]
    return nb, meta


WHITELIST_OLD = 'if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:'
OUT_OLD, OUT_NEW = '"/kaggle/inference_outputs"', '"/kaggle/working/inference_outputs"'


def set_whitelist(nb, ids):
    """Which evaluation tasks a NORMAL (non-rerun) run decodes. Production decodes 4; `ids` is a
    list of task ids or "all". The competition rerun always decodes every hidden task regardless."""
    hits = 0
    for c in nb["cells"]:
        s = msk.src(c)
        if WHITELIST_OLD in s:
            hits += s.count(WHITELIST_OLD)
            if ids == "all":
                new = "if False:"
            else:
                known = json.load(open(os.path.join(os.path.dirname(HERE), "arc-agi_evaluation_challenges.json")))
                bad = [i for i in ids if i not in known]
                assert not bad, f"not evaluation task ids: {bad}"
                assert len(set(ids)) == len(ids) and ids, "empty or duplicated whitelist"
                new = f"if key not in {sorted(ids)!r}:"
            msk.set_src(c, s.replace(WHITELIST_OLD, new))
    assert hits == 1, f"expected the whitelist line exactly once, found {hits}"
    return nb


def persist_outputs(nb):
    """Write the candidate pickles under /kaggle/working so they are in the kernel output
    (production keeps them in /kaggle/inference_outputs, which is discarded). Measurement runs need this."""
    hits = 0
    for c in nb["cells"]:
        s = msk.src(c)
        if OUT_OLD in s:
            hits += s.count(OUT_OLD)
            msk.set_src(c, s.replace(OUT_OLD, OUT_NEW))
    assert hits == 2, f"expected the output dir in arc_solver and the final cell (2), found {hits}"
    return nb


def rename(meta, title):
    """Give a measurement kernel its own slug and directory so it never overwrites (or pushes
    a new version of) the genboost submission kernel. Kaggle requires id == slugified title."""
    slug = "-".join(title.lower().split())
    assert slug and all(ch.isalnum() or ch == "-" for ch in slug), f"title must be words of letters/digits: {title!r}"
    assert slug != SLUG, "pick a name different from the genboost kernel"
    owner = meta["id"].split("/")[0]
    meta.update(id=f"{owner}/{slug}", title=title, code_file=f"{slug}.ipynb")
    return meta, os.path.join(msk.REPO, "submit_ag2_" + slug.replace("arc-agi2-", "").replace("-", "_"))


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
    # model swap: one path string + model_sources, nothing else
    nb3, meta3 = build({"probe_gold": True})
    nb3, meta3 = swap_model(nb3, meta3, "pranshubahadur/xcalibur-aa2-sft-500/Transformers/bf16/1")
    s3 = msk.src(nb3["cells"][msk.find(nb3["cells"], "%%writefile arc_solver.py")])
    assert 'model_name="/kaggle/input/models/pranshubahadur/xcalibur-aa2-sft-500/transformers/bf16/1"' in s3
    assert "sorokin" not in s3 and meta3["model_sources"] == ["pranshubahadur/xcalibur-aa2-sft-500/Transformers/bf16/1"]
    staged = os.path.join(msk.REPO, "submit_ag2_xcalibur", "arc-agi2-xcalibur.ipynb")
    if os.path.exists(staged):                    # same edit as the hand-staged xcalibur kernel
        ref = json.load(open(staged))["cells"]
        assert model_path("pranshubahadur/xcalibur-aa2-sft-500/Transformers/bf16/1") in \
            msk.src(ref[msk.find(ref, "%%writefile arc_solver.py")])
    # whitelist + persisted outputs for measurement runs
    nb4, _ = build({"probe_gold": True})
    nb4 = persist_outputs(set_whitelist(nb4, ["aa4ec2a5", "0934a4d8", "135a2760"]))
    st = msk.src(nb4["cells"][msk.find(nb4["cells"], "%%writefile starter.py")])
    assert "if key not in ['0934a4d8', '135a2760', 'aa4ec2a5']:" in st and WHITELIST_OLD not in st
    compile(st.split("\n", 1)[1], "starter.py", "exec")
    sv = msk.src(nb4["cells"][msk.find(nb4["cells"], "%%writefile arc_solver.py")])
    assert 'dir_outputs = "/kaggle/working/inference_outputs"' in sv
    assert 'decoder.load_decoded_results("/kaggle/working/inference_outputs")' in msk.src(nb4["cells"][-1])
    nb5, _ = build({})
    st5 = msk.src(set_whitelist(nb5, "all")["cells"][msk.find(nb5["cells"], "%%writefile starter.py")])
    assert "if False:" in st5 and "rerun_mode" in st5
    for bad_ids in (["zzzzzzzz"], [], ["aa4ec2a5", "aa4ec2a5"]):
        try:
            set_whitelist(build({})[0], bad_ids)
            raise SystemExit("bad whitelist accepted")
        except AssertionError:
            pass
    m6, d6 = rename(dict(meta3), "ARC AGI2 Xcalibur Panel")
    assert m6["id"].endswith("/arc-agi2-xcalibur-panel") and m6["code_file"] == "arc-agi2-xcalibur-panel.ipynb"
    assert d6.endswith("submit_ag2_xcalibur_panel") and d6 != OUT_DIR
    for bad_title in ("ARC AGI2 GenBoost", "bad/title", ""):
        try:
            rename(dict(meta3), bad_title)
            raise SystemExit("bad kernel name accepted")
        except AssertionError:
            pass
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
    ap.add_argument("--model-source", help="drop-in checkpoint, e.g. owner/slug/Transformers/bf16/1")
    ap.add_argument("--probe-gold", action="store_true", help="also write gold NLL per input (eval runs only)")
    ap.add_argument("--whitelist", help='eval tasks a normal run decodes: comma-separated ids or "all" '
                                        "(default: the 4 production tasks); implies --persist-outputs")
    ap.add_argument("--persist-outputs", action="store_true", help="keep candidate pickles in the kernel output")
    ap.add_argument("--name", help='kernel title for a measurement run, e.g. "ARC AGI2 Xcalibur Panel" '
                                   "(own slug + own submit_ag2_* directory; default: the genboost kernel)")
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
        if not d["changed"] and not (a.model_source or a.probe_gold or a.whitelist):
            sys.exit("decide(): no generation lever is supported by the diagnostics; nothing built.")
        levers = {k: v for k, v in d["levers"].items() if v != gen_boost.LEVERS_OFF[k]}
    elif a.model_source or a.probe_gold or a.whitelist:
        levers = {}
    else:
        ap.error("give --verdict, --levers, --model-source, --probe-gold or --whitelist")
    if a.probe_gold:
        levers["probe_gold"] = True
    nb = meta = None
    if a.with_select:
        if not verdict.get("winner"):
            sys.exit("--with-select needs a verdict with a winning strategy")
        nb, meta = msk.build(verdict["winner"], verdict.get("ranker"))
    nb, meta = build(levers, nb, meta)
    if a.model_source:
        nb, meta = swap_model(nb, meta, a.model_source)
    if a.whitelist:
        nb = set_whitelist(nb, "all" if a.whitelist == "all" else [x for x in a.whitelist.split(",") if x])
    if a.whitelist or a.persist_outputs:
        nb = persist_outputs(nb)
    if levers.get("probe_gold") and not a.whitelist:
        print("NOTE: a normal run decodes only the 4 production whitelist tasks -- probe_gold will measure "
              "those 4 only. Use --whitelist all (12 h) or a panel from swap_eval.py for a real measurement.")
    out_dir = OUT_DIR
    if a.name:
        meta, out_dir = rename(meta, a.name)
    elif a.model_source or a.whitelist:
        print("WARNING: no --name given: this build overwrites submit_ag2_genboost and shares its kernel slug.")
    print("built", msk.write(nb, meta, out_dir), "id:", meta["id"], "levers:", levers,
          "model:", meta["model_sources"], "(not pushed)")
