"""Dry-run the in-notebook source patch of a *-patched AGI-3 kernel against a local copy of its bundle.

    python3 check_patch_cell.py <notebook.ipynb> <bundle_dir> [--serving-setup FILE] [--fix OUT.ipynb]
    python3 check_patch_cell.py --selftest

Why: the patch cell of arc3-duck-flashnext-nvfp4-patched died on Kaggle with
    AssertionError: patch anchor missing/dup: def _chat_completion(\\n
after the GPU session had started. Everything that cell does is plain string surgery on three files of
the mounted bundle, so it can be executed locally, in seconds, on a copy of the same bundle.

<bundle_dir> is the UNPATCHED bundle the kernel mounts as dataset index 0 (it must contain
src/ARC3-Inference/inference/agent/{tool_agent,prompts}.py; serving_setup.py at its root, or pass
--serving-setup). The notebook is never modified unless --fix is given.

What is checked
  1. every edit of the cell, one by one, BEFORE running it: how often its anchor occurs in the file it is
     applied to (must be exactly 1), and -- when it does not -- whether it would match once the literal
     two-character sequence backslash+n is read as a newline (the bug class above) or the other way round
  2. the cell's patch block is then executed as written on a temp copy: its own asserts and ast.parse run
  3. the patched tool_agent.py / prompts.py / serving_setup.py compile, contain no stray backslash-n outside
     string literals, and every name the inserted code relies on exists where it is used
     (helper is a method of the class that owns _chat_completion, the argument it is called with is a
     parameter of _chat_completion, the failure counter is initialised in __init__, ...)
--fix writes a corrected notebook: only edits whose anchor fails as written and matches after the
newline reading are changed, and both strings of such an edit are converted together.
"""
import ast
import json
import os
import re
import shutil
import sys
import tempfile

START_MARKS = ("# --- PATCH", "import shutil as _shutil")
END_MARK = "BUNDLE_DIR = PATCHED_BUNDLE"
TARGETS = {"_ta": "src/ARC3-Inference/inference/agent/tool_agent.py",
           "_pr": "src/ARC3-Inference/inference/agent/prompts.py", "_ss": "serving_setup.py"}
BSN = "\\" + "n"                                   # the two characters: backslash, n


def patch_cell(nb):
    """-> (cell index, cell source, block start offset, block end offset)."""
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if c.get("cell_type") == "code" and "_edits = [" in s and END_MARK in s:
            a = min(s.index(m) for m in START_MARKS if m in s)
            b = s.index("\n", s.index(END_MARK)) + 1
            return i, s, a, b
    raise SystemExit("no patch cell found (looked for a code cell with `_edits = [` and `BUNDLE_DIR = PATCHED_BUNDLE`)")


def edits_of(block):
    """Evaluate the `_edits = [...]` / `_edits += [...]` statements only. -> list of (old, new, lineno)."""
    out = []
    for node in ast.parse(block).body:
        tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AugAssign) else None
        if getattr(tgt, "id", None) == "_edits" and isinstance(node.value, ast.List):
            for el in node.value.elts:
                old, new = ast.literal_eval(el)
                out.append((old, new, el.lineno))
    return out


def loops_of(block):
    """Which slice of _edits goes to which file: [(variable holding the text, start, stop)] from the for loops."""
    out = []
    for node in ast.parse(block).body:
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Subscript) and getattr(node.iter.value, "id", "") == "_edits":
            sl = node.iter.slice
            lo = ast.literal_eval(sl.lower) if sl.lower is not None else None
            hi = ast.literal_eval(sl.upper) if sl.upper is not None else None
            var = next((n.targets[0].id for n in ast.walk(node) if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in TARGETS), None)
            out.append((var, lo, hi))
    return out


def precheck(block, files):
    """-> rows: {i, target, count, verdict, hint} for every edit."""
    edits, rows = edits_of(block), []
    owner = {}
    for var, lo, hi in loops_of(block):
        for i in range(len(edits))[slice(lo, hi)]:
            owner[i] = var
    text = dict(files)
    for i, (old, new, line) in enumerate(edits):
        var = owner.get(i)
        row = {"i": i, "line": line, "target": TARGETS.get(var, "?"), "anchor": old[:60]}
        if var is None:
            row.update(count=None, verdict="UNUSED", hint="no loop applies this edit")
            rows.append(row)
            continue
        n = text[var].count(old)
        row["count"] = n
        if n == 1:
            row["verdict"] = "ok"
            text[var] = text[var].replace(old, new)
            try:
                ast.parse(text[var])
            except SyntaxError as e:
                row["verdict"] = "BAD_NEW"
                alt = text[var].replace(new, new.replace(BSN, "\n"))
                try:
                    ast.parse(alt)
                    row["hint"] = "replacement breaks the file (line %s); it parses when backslash-n is a newline" % e.lineno
                    row["fix"] = "to_newline"
                except SyntaxError:
                    row["hint"] = "replacement breaks the file: %s (line %s)" % (e.msg, e.lineno)
        else:
            row["verdict"] = "MISSING" if n == 0 else "DUP"
            as_nl, as_lit = old.replace(BSN, "\n"), old.replace("\n", BSN)
            if n == 0 and as_nl != old and text[var].count(as_nl) == 1:
                row["hint"], row["fix"] = "matches once when backslash-n is read as a newline", "to_newline"
                text[var] = text[var].replace(as_nl, new.replace(BSN, "\n"))
            elif n == 0 and as_lit != old and text[var].count(as_lit) == 1:
                row["hint"], row["fix"] = "matches once when the newline is written as literal backslash-n", "to_literal"
                text[var] = text[var].replace(as_lit, new.replace("\n", BSN))
        rows.append(row)
    return rows


def run_block(block, bundle, serving_setup=None):
    """Execute the block on a temp copy. -> (error or None, {var: patched text})."""
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "bundle")
        shutil.copytree(bundle, src, ignore=shutil.ignore_patterns("__pycache__", "*.pkl", "*.safetensors"))
        if serving_setup:
            shutil.copy(serving_setup, os.path.join(src, "serving_setup.py"))
        work = Path(tmp) / "working"
        work.mkdir()
        ns = {"WORKING_DIR": work, "BUNDLE_DIR": Path(src), "Path": Path, "print": lambda *a, **k: None}
        err = None
        try:
            exec(compile(block, "<patch cell>", "exec"), ns)
        except BaseException as e:                                     # noqa: BLE001  (AssertionError, SyntaxError, ...)
            err = "%s: %s" % (type(e).__name__, e)
        out = {}
        root = work / "bundle_patched"
        for var, rel in TARGETS.items():
            p = root / rel
            out[var] = p.read_text() if p.exists() else None
        return err, out


def semantic(ta):
    """Checks on the patched tool_agent.py that ast.parse alone does not give. -> list of problems."""
    bad = []
    try:
        tree = ast.parse(ta)
    except SyntaxError as e:
        return ["tool_agent.py does not parse: %s (line %s)" % (e.msg, e.lineno)]
    cls = next((c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)
                and any(isinstance(f, ast.FunctionDef) and f.name == "_chat_completion" for f in c.body)), None)
    if cls is None:
        return ["no class with _chat_completion"]
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    module_names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    module_names |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    module_names |= {a.asname or a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    src_attrs = set(re.findall(r"self\.(_\w+)\s*(?::[^=\n]+)?=[^=]", ta))
    cc = methods["_chat_completion"]
    params = {a.arg for a in cc.args.args + cc.args.kwonlyargs}
    calls = [n for n in ast.walk(cc) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_effective_max_output_tokens"]
    if "_effective_max_output_tokens" in ta:
        if "_effective_max_output_tokens" not in methods:
            bad.append("_effective_max_output_tokens is not a method of %s (indentation of the inserted def?)" % cls.name)
        if len(calls) != 1:
            bad.append("_chat_completion calls _effective_max_output_tokens %d times (expected 1)" % len(calls))
        for c in calls:
            for a in c.args:
                if isinstance(a, ast.Name) and a.id not in params:
                    bad.append("_effective_max_output_tokens(%s): not a parameter of _chat_completion" % a.id)
        if "max_tokens=self._max_output_tokens," in ast.get_source_segment(ta, cc):
            bad.append("_chat_completion still sends max_tokens=self._max_output_tokens")
    else:
        bad.append("_effective_max_output_tokens was not inserted")
    for name in ("_RETRY_SHRINK_AFTER_FAILURES", "_PERSISTENT_HISTORY_ASSISTANT_TURNS", "TOOL_CALL_FORMAT_GUIDANCE"):
        if name in ta and name not in module_names:
            bad.append("%s is used but not defined/imported at module level" % name)
    if "_consecutive_request_failures" in ta:
        init = methods.get("__init__")
        if init is None or "_consecutive_request_failures" not in ast.get_source_segment(ta, init):
            bad.append("self._consecutive_request_failures is never initialised in __init__")
    for attr in ("_keep_recent_history_turns", "_drop_until_first_user_message"):
        if "self.%s(" % attr in ta and attr not in methods:
            bad.append("self.%s() is called but %s has no such method" % (attr, cls.name))
    for attr in ("_max_output_tokens",):
        if "self.%s" % attr in ta and attr not in src_attrs:
            bad.append("self.%s is read but never assigned" % attr)
    return bad


def fix_source(cell_src, a, b, rows):
    """Convert the string literals of the edits flagged to_newline: source `\\\\n` -> `\\n`. -> new cell source."""
    block = cell_src[a:b]
    lines = block.split("\n")
    tree = ast.parse(block)
    spans = []
    k = 0
    for node in tree.body:
        tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AugAssign) else None
        if getattr(tgt, "id", None) == "_edits" and isinstance(node.value, ast.List):
            for el in node.value.elts:
                if any(r["i"] == k and r.get("fix") == "to_newline" for r in rows):
                    spans.append((el.lineno - 1, el.end_lineno - 1))
                k += 1
    for lo, hi in spans:
        for j in range(lo, hi + 1):
            lines[j] = lines[j].replace("\\" + BSN, BSN)             # source text  \\n  ->  \n
    return cell_src[:a] + "\n".join(lines) + cell_src[b:]


def check(nb_path, bundle, serving_setup=None, fix_out=None, quiet=False):
    p = (lambda *a: None) if quiet else print
    nb = json.load(open(nb_path))
    idx, src, a, b = patch_cell(nb)
    block = src[a:b]
    files = {}
    for var, rel in TARGETS.items():
        path = serving_setup if (var == "_ss" and serving_setup) else os.path.join(bundle, rel)
        files[var] = open(path).read() if os.path.exists(path) else ""
    rows = precheck(block, files)
    p("patch cell: #%d, %d edits" % (idx, len(rows)))
    for r in rows:
        p("  [%-7s] edit %-2d -> %-58s count=%s  %r%s"
          % (r["verdict"], r["i"], r["target"], r["count"], r["anchor"], "  <- " + r["hint"] if r.get("hint") else ""))
    err, out = run_block(block, bundle, serving_setup)
    p("run as written: " + ("all asserts and ast.parse passed" if err is None else "FAILED -- " + err))
    sem = semantic(out["_ta"]) if (err is None and out.get("_ta")) else []
    for m in sem:
        p("  semantic: " + m)
    if not files["_ss"]:
        p("  NOTE: the bundle has no serving_setup.py; the serving_setup edit could not be checked (pass --serving-setup)")
    ok = err is None and not sem and all(r["verdict"] == "ok" for r in rows)
    res = {"ok": ok, "rows": rows, "error": err, "semantic": sem, "fixed": None}
    fixable = [r for r in rows if r.get("fix") == "to_newline"]
    if fix_out and fixable:
        new_src = fix_source(src, a, b, rows)
        nb2 = json.loads(json.dumps(nb))
        nb2["cells"][idx]["source"] = new_src.splitlines(keepends=True) if isinstance(nb["cells"][idx]["source"], list) else new_src
        json.dump(nb2, open(fix_out, "w"), indent=1, ensure_ascii=False)
        res["fixed"] = check(fix_out, bundle, serving_setup, quiet=True)
        p("--fix: rewrote edits %s -> %s; re-check: %s"
          % ([r["i"] for r in fixable], fix_out, "OK" if res["fixed"]["ok"] else "STILL FAILING (%s)" % res["fixed"]["error"]))
    p("VERDICT: " + ("OK -- the cell patches this bundle cleanly" if ok else "NOT OK -- do not push this notebook"))
    return res


def selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    nb_path = os.path.join(repo, "submit_ag3_flashnext_patched", "arc3-duck-flashnext-nvfp4-patched.ipynb")
    bundle = os.path.join(repo, "taaf")                                 # unpatched upstream snapshot
    ss_src = os.path.join(repo, "bundle_fast", "serving_setup.py")
    for pth in (nb_path, bundle, ss_src):
        if not os.path.exists(pth):
            raise SystemExit("selftest needs %s" % pth)
    with tempfile.TemporaryDirectory() as tmp:
        ss = os.path.join(tmp, "serving_setup.py")                      # upstream wrote "0"; bundle_fast carries the 6144 fix
        text = open(ss_src).read()
        assert text.count('"LOCAL_ANALYZER_MAX_OUTPUT": "6144"') == 1
        open(ss, "w").write(text.replace('"LOCAL_ANALYZER_MAX_OUTPUT": "6144"', '"LOCAL_ANALYZER_MAX_OUTPUT": "0"'))
        # a notebook with the original bug: the two tool_agent edits written with literal backslash-n
        nb = json.load(open(nb_path))
        idx, src, a, b = patch_cell(nb)
        good_old = "    ('    def _chat_completion(" + BSN + "',"
        bad_old = "    ('    def _chat_completion(\\" + BSN + "',"
        if bad_old in src:
            broken_src = src
        else:
            assert good_old in src, "notebook layout changed"
            lines = src.split("\n")
            for j, ln in enumerate(lines):
                if "def _chat_completion(" in ln or "_effective_max_output_tokens" in ln or "max_tokens=self._max_output_tokens" in ln:
                    lines[j] = ln.replace(BSN, "\\" + BSN)
            broken_src = "\n".join(lines)
        nb["cells"][idx]["source"] = broken_src
        broken = os.path.join(tmp, "broken.ipynb")
        json.dump(nb, open(broken, "w"))
        fixed = os.path.join(tmp, "fixed.ipynb")
        r = check(broken, bundle, ss, fix_out=fixed, quiet=True)
        assert not r["ok"] and "patch anchor missing/dup:     def _chat_completion(" in r["error"], r["error"]
        bad = [x for x in r["rows"] if x["verdict"] != "ok"]
        assert [x["i"] for x in bad] == [6, 7] and all(x.get("fix") == "to_newline" for x in bad), bad
        assert all(x["verdict"] == "ok" for x in r["rows"] if x["i"] >= 8)          # prompts edits NEED the literal form
        assert r["fixed"]["ok"] and not r["fixed"]["semantic"], r["fixed"]
        f_src = "".join(json.load(open(fixed))["cells"][idx]["source"])
        # 15 escapes converted (edit 6: anchor + 12 in the inserted helper, edit 7: anchor + replacement);
        # the 5 that remain are the prompts edits, where the literal form is the correct one
        assert (broken_src.count("\\" + BSN), f_src.count("\\" + BSN)) == (20, 5)
        # half a fix (anchors converted, a replacement left literal) is caught as BAD_NEW / by the run
        half = f_src.replace("self._effective_max_output_tokens(request_timeout_seconds)," + BSN,
                             "self._effective_max_output_tokens(request_timeout_seconds),\\" + BSN)
        assert half != f_src
        nb["cells"][idx]["source"] = half
        json.dump(nb, open(broken, "w"))
        r = check(broken, bundle, ss, quiet=True)
        assert not r["ok"] and [x["verdict"] for x in r["rows"] if x["i"] == 7] == ["BAD_NEW"] and "SyntaxError" in r["error"], r
        # converting a prompts anchor as well (over-fixing) is caught too
        over = f_src.replace('still being reliable.\\' + BSN + '"\',', 'still being reliable.' + BSN + '"\',', 1)
        assert over != f_src
        nb["cells"][idx]["source"] = over
        json.dump(nb, open(broken, "w"))
        r = check(broken, bundle, ss, quiet=True)
        row8 = [x for x in r["rows"] if x["i"] == 8][0]
        assert not r["ok"] and row8["verdict"] == "MISSING" and row8.get("fix") == "to_literal" and "prompts patch anchor" in r["error"], r
        # semantic checks see what ast.parse cannot: helper pasted at the wrong indentation
        _, out = run_block("".join(json.load(open(fixed))["cells"][idx]["source"])[a:], bundle, ss)
        assert semantic(out["_ta"]) == []
        moved = out["_ta"].replace("    def _effective_max_output_tokens(self, request_timeout_seconds: float | None) -> int:",
                                   "    def _effective_max_output_tokens_x(self, request_timeout_seconds: float | None) -> int:")
        assert any("not a method" in m for m in semantic(moved))
    print("selftest ok: on the unpatched bundle the literal-backslash-n notebook fails exactly like the Kaggle run "
          "(edits 6 and 7), --fix converts both strings of those two edits (15 escapes) and the result patches, "
          "parses and passes the semantic checks; a half fix and an over-fix of the prompts anchors are both caught")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    opt = {k: sys.argv[sys.argv.index(k) + 1] for k in ("--serving-setup", "--fix") if k in sys.argv}
    r = check(sys.argv[1], sys.argv[2], opt.get("--serving-setup"), opt.get("--fix"))
    sys.exit(0 if r["ok"] else 1)
