"""Build the two-model UNION kernel: model A on every task, then model B on the tasks A left unsettled.

    python3 make_union_kernel.py --second pranshubahadur/xcalibur-aa2-sft-500/Transformers/bf16/1 \
        [--levers '{"g1": "empty_input"}'] [--reserve 0.0] [--whitelist ids|all] [--name "ARC AGI2 Union Panel"]
    python3 make_union_kernel.py --selftest

Writes ../../submit_ag2_union/ (or the --name directory). It only BUILDS; pushing is manual.
Build it only after `swap_eval.py union` on the panel shows the two models are complementary.

Design (sequential, one model in GPU memory at a time):
  pass 1  every task: TTT + decode with model A (production behaviour, fair-share cap capped at the
          production 1200 s). A task where some test input has no grid backed by 3 views is put on a
          second queue.
  swap    worker frees model A (function scope ends, gc, empty_cache) and loads model B.
  pass 2  tasks from the second queue: TTT + decode with model B. Files get the tag ".runm2", so they
          load under the same test input and simply add candidates and votes. Selection is unchanged
          (kgmon over the union).
  The per-task LoRA is re-initialised for every task in both passes, exactly as in production.

Memory: identical to production at any moment (one 3.6 B bf16 model + its r=256 TTT adapter per L4).
Disk: both checkpoints are mounted through model_sources (~7.3 GB each).

Time -- read this before building. The rerun budget is 4 workers x (12 h - 10 min) = 47.3 worker-hours.
The placeholder arc-agi_test_challenges.json in this repo has 240 tasks, which suggests the hidden
rerun decodes 240 tasks, not 120: ~710 s per task on average. Whitelist tasks measured 582-1239 s
with ONE model. So in the rerun there may be little or no time for a second model, and pass 2 is built
to degrade gracefully: it only uses what pass 1 leaves (reserve 0.0 by default), pass 1 always covers
every task first, and with no time left the kernel behaves like the single-model genboost kernel
(minus nothing: pass-1 output is already on disk). --reserve R holds back a fraction R of the budget
for pass 2 at the cost of tighter per-task caps in pass 1; only use it with evidence from
rerun_sim.py that pass 1 does not need the time.
  cost of pass 2 = (share of tasks unsettled after pass 1) x (TTT + decode of one task) + one model
  load per worker (~1-2 min, estimate). TTT is 35-67% of a task and is paid again for model B.

Unverified on GPU: loading a second FastLanguageModel in the same process after the first was
released (unsloth patches model classes at import time). The first real run should be a small
--whitelist panel, and its log must show '[Rank r] pass 2: loaded' and tagged files.
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
import make_gen_kernel as mgk         # noqa: E402
import make_select_kernel as msk      # noqa: E402

TITLE, SLUG = "ARC AGI2 Union", "arc-agi2-union"
TAG2 = ".runm2"

SOLVER_EDITS = [
    ("def worker(rank, queue, end_time):\n",
     "def _worker_pass(rank, queue, end_time, model_path, file_tag, unsettled_out):\n"),
    ('        model_name="/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",\n',
     "        model_name=model_path,\n"),
    ("        gen_boost.decode_task(\n", "        _res = gen_boost.decode_task(\n"),
    ("                 max_seq_length=max_seq_length, dir_outputs=dir_outputs, rank=rank, key=key),\n",
     "                 max_seq_length=max_seq_length, dir_outputs=dir_outputs, rank=rank, key=key,\n"
     '                 file_tag=file_tag, cap_ceil=UNION["pass1_cap"] if unsettled_out is not None else 2400.0),\n'),
    ("            puzzle_ds_multi, start_time, end_time, levers=GEN_LEVERS, tasks_left=max(queue.qsize() - 4, 0))\n",
     "            puzzle_ds_multi, start_time, end_time, levers=GEN_LEVERS, tasks_left=max(queue.qsize() - 4, 0))\n"
     "        if unsettled_out is not None and gen_boost.unsettled(_res):\n"
     "            unsettled_out.put(key)\n"),
]

WORKER = '''

def worker(rank, queue, end_time, queue2=None, done=None):
    """Pass 1: model A on every task. Pass 2: model B on the tasks pass 1 left unsettled."""
    two = queue2 is not None and len(UNION["models"]) > 1
    t0 = time.time()
    end1 = t0 + (1.0 - UNION["reserve"]) * (end_time - t0) if two else end_time
    try:
        _worker_pass(rank, queue, end1, UNION["models"][0], "", queue2 if two else None)
    finally:
        if done is not None:
            done.put(rank)                       # tells waiting pass-2 workers this producer is finished
    if not two or time.time() > end_time:
        return
    gc.collect()
    torch.cuda.empty_cache()
    import gen_boost
    print(f"[Rank {rank}] pass 2: loading {UNION['models'][1]} with {end_time - time.time():.0f}s left")
    _worker_pass(rank, gen_boost.WaitQueue(queue2, done), end_time, UNION["models"][1], UNION["tag"], None)
    print(f"[Rank {rank}] pass 2: done")
'''

STARTER_EDITS = [
    ("def local_worker(rank, queue, end_time):\n", "def local_worker(rank, queue, end_time, queue2, done):\n"),
    ("    worker(rank, queue, end_time)\n", "    worker(rank, queue, end_time, queue2, done)\n"),
    ("    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)",
     "    queue2, done = mp.Manager().Queue(), mp.Manager().Queue()\n"
     "    mp.spawn(local_worker, args=(queue, args.end_time, queue2, done), nprocs=4)"),
]


def _edit(text, edits, what):
    for a, b in edits:
        assert text.count(a) == 1, f"{what}: anchor matched {text.count(a)}x: {a.strip()[:60]!r}"
        text = text.replace(a, b)
    return text


def build(second, levers=None, reserve=0.0, first=mgk.BASE_MODEL_SOURCE):
    assert 0.0 <= reserve <= 0.5, "reserve must be within 0..0.5"
    assert second and second != first, "the second model must differ from the first"
    lv = dict(levers or {})
    lv["faircap"] = True                         # pass 1 must be coverage-first, or pass 2 never gets time
    nb, meta = mgk.build(lv)
    cells = nb["cells"]
    i = msk.find(cells, "%%writefile arc_solver.py")
    union = {"models": [mgk.model_path(first), mgk.model_path(second)], "tag": TAG2, "reserve": reserve,
             "pass1_cap": gen_boost.BASE_CAP}
    s = _edit(msk.src(cells[i]), SOLVER_EDITS, "arc_solver")
    lev_line = [ln for ln in s.splitlines() if ln.startswith("GEN_LEVERS = ")]
    assert len(lev_line) == 1
    s = s.replace(lev_line[0] + "\n", lev_line[0] + f"\nUNION = {union!r}\n") + WORKER
    msk.set_src(cells[i], s)
    j = msk.find(cells, "%%writefile starter.py")
    msk.set_src(cells[j], _edit(msk.src(cells[j]), STARTER_EDITS, "starter"))
    assert meta["model_sources"] == [mgk.BASE_MODEL_SOURCE]
    meta["model_sources"] = [first, second]
    owner = meta["id"].split("/")[0]
    meta.update(id=f"{owner}/{SLUG}", title=TITLE, code_file=f"{SLUG}.ipynb")
    return nb, meta


def selftest():
    import bz2
    import gc
    import pickle
    import queue as _q
    import tempfile
    import numpy as np
    second = "pranshubahadur/xcalibur-aa2-sft-500/Transformers/bf16/1"
    nb, meta = build(second, {"g1": "empty_input"}, reserve=0.25)
    assert meta["model_sources"] == [mgk.BASE_MODEL_SOURCE, second] and meta["id"].endswith("/" + SLUG)
    solver = msk.src(nb["cells"][msk.find(nb["cells"], "%%writefile arc_solver.py")]).split("\n", 1)[1]
    starter = msk.src(nb["cells"][msk.find(nb["cells"], "%%writefile starter.py")]).split("\n", 1)[1]
    compile(starter, "starter.py", "exec")
    tree = ast.parse(solver)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_worker_pass" in fns and "worker" in fns
    assert [a.arg for a in fns["_worker_pass"].args.args] == ["rank", "queue", "end_time", "model_path", "file_tag", "unsettled_out"]
    assert "sorokin" not in ast.unparse(fns["_worker_pass"]) and "model_name=model_path" in ast.unparse(fns["_worker_pass"])
    assert ast.unparse(fns["_worker_pass"]).count("gen_boost.decode_task(") == 1
    assert "worker(rank, queue, end_time, queue2, done)" in starter and "args=(queue, args.end_time, queue2, done)" in starter
    # only the union edits differ from the genboost build
    g_nb, _ = mgk.build({"g1": "empty_input", "faircap": True})
    g_solver = msk.src(g_nb["cells"][msk.find(g_nb["cells"], "%%writefile arc_solver.py")]).split("\n", 1)[1]
    back = solver.replace(WORKER, "")
    for a, b in SOLVER_EDITS:
        assert back.count(b) == 1
        back = back.replace(b, a)
    back = "\n".join(ln for ln in back.split("\n") if not ln.startswith("UNION = "))
    assert back == g_solver, "union build must equal the genboost build plus the listed edits"

    # run the generated worker() against stubs: order of passes, end times, done token, queues
    ns = {"time": __import__("time"), "gc": gc}
    exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.Assign)
                             and getattr(n.targets[0], "id", "") in ("UNION", "GEN_LEVERS")], []), "consts", "exec"), ns)
    assert ns["UNION"]["models"][1].endswith("xcalibur-aa2-sft-500/transformers/bf16/1") and ns["GEN_LEVERS"]["faircap"]
    calls, emptied = [], []
    ns["torch"] = type("T", (), {"cuda": type("C", (), {"empty_cache": staticmethod(lambda: emptied.append(1))})})

    def fake_pass(rank, queue, end_time, model_path, file_tag, unsettled_out):
        calls.append((model_path, file_tag, type(queue).__name__, unsettled_out, end_time))
        if unsettled_out is not None:
            unsettled_out.put("task_b")
        else:
            got = []
            while not queue.empty():
                got.append(queue.get())
            calls.append(got)
    ns["_worker_pass"] = fake_pass
    exec(compile(ast.Module([fns["worker"]], []), "worker", "exec"), ns)
    q1, q2, done = _q.Queue(), _q.Queue(), _q.Queue()
    for _ in range(3):
        done.put(0)                                              # the other three workers already finished pass 1
    import time as _t
    t = _t.time()
    ns["worker"](0, q1, t + 1000.0, q2, done)
    assert calls[0][:4] == (ns["UNION"]["models"][0], "", "Queue", q2) and 740 < calls[0][4] - t < 760      # 25% reserved
    assert calls[1][:4] == (ns["UNION"]["models"][1], TAG2, "WaitQueue", None) and abs(calls[1][4] - (t + 1000)) < 1
    assert calls[2] == ["task_b"] and done.qsize() == 4 and emptied == [1]
    calls.clear()
    ns["worker"](0, q1, t - 1.0, q2, done)                       # budget gone after pass 1 -> no pass 2
    assert len(calls) == 1
    calls.clear()
    ns["worker"](0, q1, t + 1000.0)                              # production-style call still works: single pass
    assert len(calls) == 2 and calls[0][3] is None and calls[1] == [] and abs(calls[0][4] - (t + 1000)) < 1

    # file-level union under the production decoder and kgmon
    sys.path.insert(0, os.path.join(msk.REPO, "submit_ag2_perfpatch", "out"))
    import arc_decoder
    gold = np.array([[1, 2], [3, 4]])
    wrong = np.array([[1, 2], [3, 0]])
    other = np.array([[9, 9], [9, 9]])
    mk = lambda g, aug: {"beam_score": 0.3, "score_aug": [aug] * 8, "solution": g}       # noqa: E731

    def top2(files):
        with tempfile.TemporaryDirectory() as d:
            for name, samples in files.items():
                with bz2.BZ2File(os.path.join(d, name), "w") as f:
                    pickle.dump(samples, f)
            dec = arc_decoder.ArcDecoder(type("DS", (), {"queries": {}, "replies": {}})(), n_guesses=2)
            dec.load_decoded_results(d)
            assert list(dec.decoded_results) == ["aaaa0001_0"]
            return [g.tolist() for g in dec.run_selection_algo()["aaaa0001_0"][:2]]

    views = [f"aaaa0001_0{g}.permute0123456789.ex01" for g in ("", ".rot90", ".transpose")]
    a_files = {views[0]: [mk(wrong, 5.0)], views[1]: [mk(wrong, 5.0)], views[2]: [mk(other, 9.0)]}
    same_b = {v + TAG2: s for v, s in a_files.items()}                         # B agrees with A everywhere
    assert top2({**a_files, **same_b}) == top2(a_files) == [wrong.tolist(), other.tolist()]
    b_gold = {v + TAG2: [mk(gold, 2.0)] for v in views}                        # B finds what A never generated
    assert top2({**a_files, **b_gold})[0] == gold.tolist()
    assert top2(b_gold) == [gold.tolist()]                                     # A produced nothing at all
    # refusals
    for bad in (dict(second=mgk.BASE_MODEL_SOURCE), dict(second=second, reserve=0.9)):
        try:
            build(**bad)
            raise SystemExit("bad union config accepted")
        except AssertionError:
            pass
    nb2, meta2 = build(second)
    nb2 = mgk.persist_outputs(mgk.set_whitelist(nb2, ["aa4ec2a5", "0934a4d8"]))
    meta2, out_dir = mgk.rename(meta2, "ARC AGI2 Union Panel")
    assert out_dir.endswith("submit_ag2_union_panel") and "'reserve': 0.0" in msk.src(nb2["cells"][msk.find(nb2["cells"], "%%writefile arc_solver.py")])
    print("selftest ok: worker() runs model A on every task then model B on the unsettled queue (tagged files, "
          "reserve honoured, no pass 2 without time, single-pass call still valid); the build equals genboost plus "
          "the listed edits; under the production decoder a second model that agrees leaves kgmon's top-2 unchanged "
          "and one that finds a new grid adds it")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--second")
    ap.add_argument("--first", default=mgk.BASE_MODEL_SOURCE)
    ap.add_argument("--levers", default="{}")
    ap.add_argument("--reserve", type=float, default=0.0)
    ap.add_argument("--whitelist")
    ap.add_argument("--probe-gold", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not a.second:
        ap.error("give --second owner/slug/Framework/variation/version")
    levers = json.loads(a.levers)
    if a.probe_gold:
        levers["probe_gold"] = True
    nb, meta = build(a.second, levers, a.reserve, a.first)
    if a.first != mgk.BASE_MODEL_SOURCE:
        raise SystemExit("--first other than sft139 is not supported yet (the base path is patched in place)")
    out_dir = os.path.join(msk.REPO, "submit_ag2_union")
    if a.whitelist:
        nb = mgk.persist_outputs(mgk.set_whitelist(nb, "all" if a.whitelist == "all" else [x for x in a.whitelist.split(",") if x]))
    if a.name:
        meta, out_dir = mgk.rename(meta, a.name)
    print("built", msk.write(nb, meta, out_dir), "id:", meta["id"], "models:", meta["model_sources"],
          "reserve:", a.reserve, "(not pushed)")
