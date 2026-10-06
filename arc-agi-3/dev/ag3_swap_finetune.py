"""Serve huikang/qwen-38-flash-next-finetune (v15) from a flash-next AGI-3 bundle, and score it honestly.

    python3 ag3_swap_finetune.py apply  <bundle_dir> [--dry-run]         # patch serving_setup.py + SOURCE_IDENTITY.json
    python3 ag3_swap_finetune.py kernel <kernel_dir>                     # add the model to kernel-metadata.json
    python3 ag3_swap_finetune.py check-mount <finetune_dir> [<reference_dir>]   # after a download, or inside a kernel
    python3 ag3_swap_finetune.py holdout <baseline_output_dir> <finetune_output_dir>
    python3 ag3_swap_finetune.py pins
    python3 ag3_swap_finetune.py --selftest

What the fine-tune is (read from its public files, 2026-10-05):
  LoRA r=16 trained on 300 ARC-AGI-3 play transcripts (14.7 M tokens), merged into
  RadixArk/Qwen3.8-Flash-Next-NVFP4 @ 7b719225 -- the exact checkpoint our bundles serve
  (keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1). Apache 2.0.
  Its tensor inventory is identical to the base (296,475 names; dtype and shape equal on all 4,497 tensors
  of the four dense shards and two expert shards that were compared header by header), its config.json
  and hf_quant_config.json are byte-identical to the base, and it ships no PLE table: assemble.py (shipped
  with it) builds a servable directory out of symlinks to the fine-tune plus the PLE of the mounted base
  (0.07 GB copied, the rest linked).

So NO re-upload is needed: mount the public model next to the base and assemble at start-up.
  kernel-metadata.json  model_sources += huikang/qwen-38-flash-next-finetune/Transformers/default/15
  serving_setup.py      the base is resolved and verified exactly as before (it is the PLE reference), then
                        assemble_finetune() verifies the fine-tune's export manifest and the two scripts it
                        is about to import against pinned hashes, runs assemble.py into /tmp, checks the
                        result, and the vLLM server (and the watchdog, which re-reads the path from the saved
                        server identity) is started on the assembled directory.
  SOURCE_IDENTITY.json  serving_setup_sha256 is updated (serving_setup.py checks its own hash).
The PLE patch gate of the runtime compares an environment variable with a constant, not the file on
disk, so the re-serialised config.json of the assembled directory does not disturb it.
A private mirror is only insurance against the author deleting v15; if you make one, upload the files
unchanged and pass --mount with its /kaggle/input/models/... path.

Evaluation -- 13 of the public-25 games are in its training data (levels 1-2 mostly):
    bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 sk48 su15 tn36 tr87 wa30
  Route A  full public-25 run, then `holdout`: only the other 12 games count
           (ar25 ft09 lp85 m0r0 r11l re86 s5i5 sb26 sc25 sp80 tu93 vc33). Do NOT run a 12-game kernel:
           fewer concurrent games means faster turns, and the result would not be comparable with any
           baseline run of 25.
  Route B  LB submission: the hidden games are unseen by construction; one submission is one sample
           (run sigma ~0.85 on the public set).
Verified without a GPU: the real assemble.py / reference_ple.py were run through the inserted
assemble_finetune() with the real pins, on the real safetensors headers of all 219 fine-tune files and the
12 base PLE shards (file bodies sparse): 296,475 tensors indexed, 138 PLE tensors, 66 MB copied, second call
idempotent, assembled config hash as pinned.
NOT verified: loading in vLLM. The assembled directory differs from the base in three ways the loader
could care about: PLE shards are named reference-model-plefp8-*.safetensors (+ reference-ple-aux.safetensors),
config.json is re-serialised with sorted keys, and there is no MODEL_MANIFEST.json. The export was made for
an SGLang loader. The first real run must show FINETUNE_MODEL_PATH and then reach vLLM readiness; if it
does not, the server log tail decides (serving_setup prints it), and nothing else in the bundle changed.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SOURCE = "huikang/qwen-38-flash-next-finetune/Transformers/default/15"
MOUNT = "/kaggle/input/models/huikang/qwen-38-flash-next-finetune/transformers/default/15"
PINS = {
    "fingerprint": "15f294183d2a123c1ef1b34c0a4e9d5d0f44d6646c428aed1bad4cfaa8091669",
    "export_manifest_sha256": "874c274490259095ec2ecb9454c75d41042afe8dedb1337f96862dae4acbf560",
    "assemble_py_sha256": "a772e533e5c4e2254a7acbe74a644226a1deef5890cebe2efb419e3893ab1488",
    "reference_ple_py_sha256": "ba39afbf6f83e484eaa4783873e15336bb61ebfb782c4ef7e29dc118e83fd0bd",
    "assembled_config_sha256": "a39bdf4478c9805b3c294a28df6bd7ae43b8a63f81b6a345e6e6c934b097fbaf",
    "file_count": 219,
    "total_bytes": 83_987_882_709,
    "base_repo": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "base_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
}
CONTAMINATED = ("bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "sk48", "su15", "tn36", "tr87", "wa30")
MARK = "def assemble_finetune("

CONSTANTS = '''FINETUNE_KAGGLE_PATH = Path(%(mount)r)
FINETUNE_FINGERPRINT = %(fingerprint)r
FINETUNE_EXPORT_MANIFEST_SHA256 = %(export_manifest_sha256)r
FINETUNE_SCRIPT_SHA256 = {"assemble.py": %(assemble_py_sha256)r, "reference_ple.py": %(reference_ple_py_sha256)r}
FINETUNE_ASSEMBLED_CONFIG_SHA256 = %(assembled_config_sha256)r
FINETUNE_FILE_COUNT = %(file_count)d
FINETUNE_TOTAL_BYTES = %(total_bytes)d
FINETUNE_OUTPUT = Path("/tmp/qwen38-flash-next-finetune-model")

'''

FUNCTION = '''def assemble_finetune(reference_dir: Path, *, verify_sha256: bool = False) -> tuple[Path, dict[str, Any]]:
    """Build the servable fine-tune directory: fine-tune weights + the verified base model's PLE."""
    mount = FINETUNE_KAGGLE_PATH
    manifest_path = mount / "export-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected the mounted fine-tune at {mount}.")
    if sha256_file(manifest_path) != FINETUNE_EXPORT_MANIFEST_SHA256:
        raise RuntimeError("Fine-tune export manifest hash does not match the pinned version.")
    manifest = read_json(manifest_path)
    files = manifest.get("files") or {}
    if (
        manifest.get("fingerprint") != FINETUNE_FINGERPRINT
        or (manifest.get("base") or {}).get("repo") != MODEL_HF_REPO
        or (manifest.get("base") or {}).get("revision") != MODEL_HF_REVISION
        or len(files) != FINETUNE_FILE_COUNT
        or sum(int(row["size"]) for row in files.values()) != FINETUNE_TOTAL_BYTES
    ):
        raise RuntimeError("Mounted fine-tune identity does not match the pinned export.")
    for name, row in files.items():
        path = mount / name
        if not path.is_file() or path.stat().st_size != int(row["size"]):
            raise RuntimeError(f"Missing or wrong-sized fine-tune file: {path}")
    for name, expected in FINETUNE_SCRIPT_SHA256.items():
        if sha256_file(mount / name) != expected or files[name]["sha256"] != expected:
            raise RuntimeError(f"Fine-tune script hash mismatch (refusing to import it): {name}")
    import importlib.util

    sys.path.insert(0, str(mount))
    try:
        spec = importlib.util.spec_from_file_location("finetune_assemble", mount / "assemble.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        started = time.monotonic()
        output = Path(module.assemble(mount, reference_dir, FINETUNE_OUTPUT, verify_sha256=verify_sha256))
    finally:
        sys.path.remove(str(mount))
    if sha256_file(output / "config.json") != FINETUNE_ASSEMBLED_CONFIG_SHA256:
        raise RuntimeError("Assembled fine-tune config hash does not match.")
    if sha256_file(output / "hf_quant_config.json") != MODEL_QUANT_CONFIG_SHA256:
        raise RuntimeError("Assembled fine-tune quantization config hash does not match.")
    record = read_json(output / "assembly.json")
    if record.get("status") != "complete" or record.get("fine_tune_fingerprint") != FINETUNE_FINGERPRINT:
        raise RuntimeError(f"Fine-tune assembly did not complete: {record}")
    for name in ("model.safetensors.index.json", "chat_template.jinja"):
        if not (output / name).is_file():
            raise RuntimeError(f"Incomplete assembled checkpoint; missing {name}")
    print(
        f"FINETUNE_MODEL_PATH {output} fingerprint={FINETUNE_FINGERPRINT[:12]} "
        f"tensors={record.get('tensor_count')} ple_tensors={record.get('ple_tensors')} "
        f"sha256_verified={verify_sha256} elapsed_s={time.monotonic() - started:.1f}",
        flush=True,
    )
    return output, {
        "fingerprint": FINETUNE_FINGERPRINT,
        "mount": str(mount),
        "model_path": str(output),
        "tensor_count": record.get("tensor_count"),
        "payload_sha256_verified": verify_sha256,
    }


'''

CONST_ANCHOR = 'VLLM_IMAGE = "vllm/vllm-openai:qwen38-flash-next"\n'
FUNC_ANCHOR = 'def verify_model(model_dir: Path, *, full_file_hashes: bool = True) -> dict[str, Any]:\n'
MAIN_OLD = ('    model_check = verify_model(model_dir, full_file_hashes=not fast_start)\n'
            '    server_identity = start_server(model_dir, env, tuning=tuning)\n')
MAIN_NEW = ('    model_check = verify_model(model_dir, full_file_hashes=not fast_start)\n'
            '    model_dir, finetune_check = assemble_finetune(model_dir, verify_sha256=not fast_start)\n'
            '    model_check["finetune"] = finetune_check\n'
            '    server_identity = start_server(model_dir, env, tuning=tuning)\n')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def patch_serving_setup(text, mount=MOUNT, pins=None):
    """-> patched text. Anchors must match exactly once; idempotent."""
    if MARK in text:
        return text
    p = dict(PINS, **(pins or {}), mount=mount)
    edits = [(CONST_ANCHOR, CONSTANTS % p + CONST_ANCHOR), (FUNC_ANCHOR, FUNCTION + FUNC_ANCHOR), (MAIN_OLD, MAIN_NEW)]
    for a, b in edits:
        n = text.count(a)
        if n != 1:
            raise SystemExit(f"serving_setup.py: anchor matched {n} times (need exactly 1): {a.strip()[:70]!r}")
        text = text.replace(a, b)
    for name in ("MODEL_HF_REPO", "MODEL_HF_REVISION", "MODEL_QUANT_CONFIG_SHA256", "def sha256_file(", "def read_json(",
                 "import time", "import sys"):
        if name not in text:
            raise SystemExit(f"serving_setup.py does not define {name}, which the inserted code needs")
    compile(text, "serving_setup.py", "exec")
    return text


def apply(bundle, dry_run=False, mount=MOUNT, pins=None):
    ss_path, si_path = os.path.join(bundle, "serving_setup.py"), os.path.join(bundle, "SOURCE_IDENTITY.json")
    old = open(ss_path).read()
    new = patch_serving_setup(old, mount, pins)
    if new == old:
        return "already patched"
    ident = json.load(open(si_path))
    old_sha = hashlib.sha256(old.encode()).hexdigest()
    if ident.get("serving_setup_sha256") != old_sha:
        raise SystemExit("SOURCE_IDENTITY.json does not match the current serving_setup.py "
                         f"({ident.get('serving_setup_sha256')} != {old_sha}); fix that first")
    others = []
    for root, _, names in os.walk(bundle):
        for name in names:
            path = os.path.join(root, name)
            if path in (ss_path, si_path) or os.path.getsize(path) > 5_000_000:
                continue
            try:
                if old_sha in open(path, encoding="utf-8", errors="ignore").read():
                    others.append(os.path.relpath(path, bundle))
            except OSError:
                pass
    if others:
        raise SystemExit("the serving_setup.py hash is also recorded in: %s -- update this tool before applying" % others)
    if dry_run:
        return "dry run ok: 3 anchors unique, patched serving_setup.py compiles, SOURCE_IDENTITY.json is the only hash record"
    open(ss_path, "w").write(new)
    ident["serving_setup_sha256"] = hashlib.sha256(new.encode()).hexdigest()
    ident["finetune"] = {"model_source": MODEL_SOURCE, "mount": mount, "fingerprint": (pins or PINS).get("fingerprint", PINS["fingerprint"]),
                         "served_from": "assembled at start-up with the pinned base model's PLE"}
    with open(si_path, "w") as f:
        json.dump(ident, f, indent=2, sort_keys=True)
        f.write("\n")
    return "patched (serving_setup_sha256 -> %s)" % ident["serving_setup_sha256"]


def patch_kernel(kernel_dir, source=MODEL_SOURCE):
    path = os.path.join(kernel_dir, "kernel-metadata.json")
    meta = json.load(open(path))
    srcs = list(meta.get("model_sources") or [])
    if not any("qwen3-8-flash-next-nvfp4" in s for s in srcs):
        raise SystemExit("kernel does not mount the flash-next base model (needed as the PLE reference): %s" % srcs)
    if meta.get("machine_shape") != "NvidiaRtxPro6000":
        raise SystemExit("machine_shape is %r, expected NvidiaRtxPro6000" % meta.get("machine_shape"))
    if source in srcs:
        return "already listed"
    meta["model_sources"] = srcs + [source]
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    return "model_sources: %s" % meta["model_sources"]


def check_mount(finetune, reference=None):
    """What assemble_finetune() checks before importing, runnable on any copy of the model."""
    out = []
    mp = os.path.join(finetune, "export-manifest.json")
    m = json.load(open(mp))
    files = m.get("files") or {}
    out.append(("export-manifest.json sha256", sha256_file(mp) == PINS["export_manifest_sha256"]))
    out.append(("fingerprint", m.get("fingerprint") == PINS["fingerprint"]))
    out.append(("base repo/revision", (m.get("base") or {}).get("repo") == PINS["base_repo"]
                and (m.get("base") or {}).get("revision") == PINS["base_revision"]))
    out.append(("file count / bytes", len(files) == PINS["file_count"]
                and sum(int(r["size"]) for r in files.values()) == PINS["total_bytes"]))
    missing = [n for n, r in files.items()
               if not os.path.isfile(os.path.join(finetune, n)) or os.path.getsize(os.path.join(finetune, n)) != int(r["size"])]
    out.append(("all %d files present with the listed sizes" % len(files), not missing))
    for name, key in (("assemble.py", "assemble_py_sha256"), ("reference_ple.py", "reference_ple_py_sha256")):
        p = os.path.join(finetune, name)
        out.append((name + " sha256", os.path.isfile(p) and sha256_file(p) == PINS[key]))
    if reference:
        for name, sha in (("config.json", "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"),
                          ("model.safetensors.index.json", "da5ca9c3b65e48e151329e64e141c2fa700bf2f99aec53cc014e4b52a6ff7a84")):
            p = os.path.join(reference, name)
            out.append(("reference " + name + " sha256", os.path.isfile(p) and sha256_file(p) == sha))
    return out, missing


def holdout(base_dir, ft_dir, quiet=False):
    """Score both runs on the 12 games the fine-tune never saw, and on the 13 it was trained on."""
    sys.path.insert(0, HERE)
    import ag3_autopsy
    runs = {}
    for label, d in (("baseline", base_dir), ("finetune", ft_dir)):
        res = ag3_autopsy.analyse(ag3_autopsy.load_run(d))
        runs[label] = {g["game"]: g for g in res["games"]}
    common = sorted(set(runs["baseline"]) & set(runs["finetune"]))
    if not common:
        raise SystemExit("the two runs share no game")
    groups = {"holdout": [g for g in common if g not in CONTAMINATED], "trained_on": [g for g in common if g in CONTAMINATED]}
    out = {}
    p = (lambda *a: None) if quiet else print
    for name, games in groups.items():
        if not games:
            continue
        b = sum(runs["baseline"][g]["score"] for g in games) / len(games)
        f = sum(runs["finetune"][g]["score"] for g in games) / len(games)
        lb = sum(runs["baseline"][g]["completed"] for g in games)
        lf = sum(runs["finetune"][g]["completed"] for g in games)
        wins = sum(runs["finetune"][g]["score"] > runs["baseline"][g]["score"] + 1e-9 for g in games)
        loss = sum(runs["finetune"][g]["score"] < runs["baseline"][g]["score"] - 1e-9 for g in games)
        out[name] = {"games": len(games), "baseline": b, "finetune": f, "delta": f - b, "levels": [lb, lf], "wins": wins, "losses": loss}
        p("%-10s %2d games: baseline %.2f -> finetune %.2f (%+.2f), levels %d -> %d, games better/worse %d/%d"
          % (name, len(games), b, f, f - b, lb, lf, wins, loss))
        for g in games:
            x, y = runs["baseline"][g], runs["finetune"][g]
            p("    %-5s %d/%d %6.2f -> %d/%d %6.2f" % (g, x["completed"], x["n_levels"], x["score"], y["completed"], y["n_levels"], y["score"]))
    h = out.get("holdout")
    if not h:
        verdict = "no holdout game in common -- nothing can be concluded"
    elif h["games"] < 12:
        verdict = "only %d of the 12 holdout games are in both runs -- rerun before deciding" % h["games"]
    elif h["delta"] >= 1.7:
        verdict = "submit the fine-tune to the LB: holdout %+.2f (>= 1.7, one run-to-run sigma of a 12-game difference)" % h["delta"]
    elif h["delta"] <= -1.7:
        verdict = "drop the fine-tune: holdout %+.2f" % h["delta"]
    else:
        verdict = ("inconclusive: holdout %+.2f is inside the noise of one run (+-1.7); a second pair of runs or one LB "
                   "submission decides" % h["delta"])
    t = out.get("trained_on")
    if h and t and t["delta"] - h["delta"] >= 3.0:
        verdict += " | memorisation signal: trained-on games %+.2f vs holdout %+.2f" % (t["delta"], h["delta"])
    p("VERDICT: " + verdict)
    out["verdict"] = verdict
    return out


def selftest():
    import shutil
    import tempfile
    repo = os.path.dirname(os.path.dirname(HERE))
    with tempfile.TemporaryDirectory() as tmp:
        # 1. the patch applies to every flash-next bundle in the repo, keeps the self-hash contract, is idempotent
        seen = 0
        for name in ("bundle_anim_v2", "bundle_fast", "bundle_hybrid"):
            src = os.path.join(repo, name)
            if not os.path.exists(os.path.join(src, "serving_setup.py")):
                continue
            dst = os.path.join(tmp, name)
            os.makedirs(dst)
            for f in ("serving_setup.py", "SOURCE_IDENTITY.json"):
                shutil.copy(os.path.join(src, f), dst)
            before = open(os.path.join(dst, "serving_setup.py")).read()
            assert apply(dst, dry_run=True).startswith("dry run ok")
            assert open(os.path.join(dst, "serving_setup.py")).read() == before
            assert apply(dst).startswith("patched") and apply(dst) == "already patched"
            text = open(os.path.join(dst, "serving_setup.py")).read()
            ident = json.load(open(os.path.join(dst, "SOURCE_IDENTITY.json")))
            assert ident["serving_setup_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            assert ident["finetune"]["fingerprint"] == PINS["fingerprint"]
            assert text.count("assemble_finetune(") == 2 and text.index("model_check = verify_model(") < text.index(
                "model_dir, finetune_check = assemble_finetune(") < text.index("server_identity = start_server(model_dir, env")
            back = text.replace(CONSTANTS % dict(PINS, mount=MOUNT), "").replace(FUNCTION, "").replace(MAIN_NEW, MAIN_OLD)
            assert back == before, "the patch must be exactly the three listed insertions"
            seen += 1
        assert seen, "no flash-next bundle found in the repo"

        # 2. the inserted function, executed for real against a miniature fine-tune + reference on disk.
        #    The assembler is a stand-in with the same interface (the real one pins the 35 MB base index).
        ft, ref = os.path.join(tmp, "ft"), os.path.join(tmp, "ref")
        os.makedirs(ft)
        os.makedirs(ref)
        cfg = {"text_config": {"ple_embedding_dtype": "float8_e4m3fn", "b": 1}, "a": 2}
        open(os.path.join(ft, "config.json"), "w").write(json.dumps(cfg))
        open(os.path.join(ft, "hf_quant_config.json"), "w").write('{"quantization": {}}')
        open(os.path.join(ft, "chat_template.jinja"), "w").write("{{ x }}")
        open(os.path.join(ft, "model.safetensors.index.json"), "w").write('{"weight_map": {}}')
        open(os.path.join(ft, "reference_ple.py"), "w").write("NAME = 'reference_ple'\n")
        open(os.path.join(ft, "assemble.py"), "w").write(
            "import json\nfrom pathlib import Path\nimport reference_ple\n"
            "def assemble(model, ple, output, *, verify_sha256=False, reference_ple=False):\n"
            "    model, ple, output = Path(model), Path(ple), Path(output)\n"
            "    m = json.loads((model / 'export-manifest.json').read_text())\n"
            "    output.mkdir(parents=True, exist_ok=True)\n"
            "    for n in ('hf_quant_config.json', 'chat_template.jinja'):\n"
            "        if not (output / n).exists(): (output / n).symlink_to(model / n)\n"
            "    c = json.loads((model / 'config.json').read_text())\n"
            "    (output / 'config.json').write_text(json.dumps(c, indent=2, sort_keys=True) + '\\n')\n"
            "    (output / 'model.safetensors.index.json').write_text('{}')\n"
            "    (output / 'assembly.json').write_text(json.dumps({'status': 'complete', 'tensor_count': 7, 'ple_tensors': 2,\n"
            "        'fine_tune_fingerprint': m['fingerprint'], 'reference': str(ple), 'verify': verify_sha256}))\n"
            "    return output\n")
        files = {n: {"size": os.path.getsize(os.path.join(ft, n)), "sha256": sha256_file(os.path.join(ft, n))} for n in os.listdir(ft)}
        body = {"format_version": 1, "base": {"repo": PINS["base_repo"], "revision": PINS["base_revision"]}, "files": files}
        body["fingerprint"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        open(os.path.join(ft, "export-manifest.json"), "w").write(json.dumps(body))
        pins = {"fingerprint": body["fingerprint"], "export_manifest_sha256": sha256_file(os.path.join(ft, "export-manifest.json")),
                "assemble_py_sha256": files["assemble.py"]["sha256"], "reference_ple_py_sha256": files["reference_ple.py"]["sha256"],
                "assembled_config_sha256": hashlib.sha256((json.dumps(cfg, indent=2, sort_keys=True) + "\n").encode()).hexdigest(),
                "file_count": len(files), "total_bytes": sum(v["size"] for v in files.values())}
        import time as _time
        from pathlib import Path

        def load(out_dir, pins_):
            ns = {"Path": Path, "Any": __import__("typing").Any, "sys": sys, "time": _time, "sha256_file": lambda p: sha256_file(str(p)),
                  "read_json": lambda p: json.load(open(p)), "MODEL_HF_REPO": PINS["base_repo"],
                  "MODEL_HF_REVISION": PINS["base_revision"],
                  "MODEL_QUANT_CONFIG_SHA256": files["hf_quant_config.json"]["sha256"], "print": lambda *a, **k: None}
            exec(CONSTANTS % dict(PINS, **pins_, mount=ft), ns)
            ns["FINETUNE_OUTPUT"] = Path(out_dir)
            exec(FUNCTION, ns)
            return ns["assemble_finetune"]

        out, rec = load(os.path.join(tmp, "o1"), pins)(Path(ref), verify_sha256=True)
        assert out == Path(tmp) / "o1" and rec["fingerprint"] == body["fingerprint"] and rec["tensor_count"] == 7
        assert json.load(open(out / "assembly.json"))["reference"] == ref and str(ft) not in sys.path
        for bad, msg in ((dict(pins, export_manifest_sha256="0" * 64), "manifest hash"),
                         (dict(pins, assemble_py_sha256="0" * 64), "refusing to import"),
                         (dict(pins, assembled_config_sha256="0" * 64), "config hash"),
                         (dict(pins, total_bytes=1), "identity")):
            try:
                load(os.path.join(tmp, "o_" + msg[:4]), bad)(Path(ref))
                raise SystemExit("accepted a wrong pin: " + msg)
            except RuntimeError as e:
                assert msg in str(e), (msg, str(e))
        os.remove(os.path.join(ft, "chat_template.jinja"))
        try:
            load(os.path.join(tmp, "o5"), pins)(Path(ref))
            raise SystemExit("accepted an incomplete mount")
        except RuntimeError as e:
            assert "Missing or wrong-sized" in str(e)

        # 3. kernel metadata
        kd = os.path.join(tmp, "kernel")
        os.makedirs(kd)
        shutil.copy(os.path.join(repo, "submit_ag3_anim_v2", "kernel-metadata.json"), kd)
        assert MODEL_SOURCE in patch_kernel(kd) and patch_kernel(kd) == "already listed"
        meta = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        assert meta["model_sources"] == ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1", MODEL_SOURCE]
        meta["machine_shape"] = "NvidiaTeslaT4"
        json.dump(meta, open(os.path.join(kd, "kernel-metadata.json"), "w"))
        try:
            patch_kernel(kd)
            raise SystemExit("patched a T4 kernel")
        except SystemExit as e:
            assert "NvidiaRtxPro6000" in str(e)

        # 4. holdout scoring from kernel-log lines: +1 level on every holdout game, nothing on trained-on games
        import ag3_autopsy

        def log(path, bonus):
            os.makedirs(path)
            with open(os.path.join(path, "kernel.log"), "w") as f:
                for g, base in sorted(ag3_autopsy.BASELINES.items()):
                    k = 1 + (bonus if g not in CONTAMINATED else 0)
                    acts = [base[i] if i < k else (5 if i == k else 0) for i in range(len(base))]
                    sc = ag3_autopsy.game_score(base, acts, k)
                    f.write("[finished] %s-x state=gave_up level=%d/%d score=%.2f actions=%d tokens=0 per-level=%s\n"
                            % (g, k, len(base), sc, sum(acts), ",".join("%d/%d" % ab for ab in zip(acts, base))))
        log(os.path.join(tmp, "base"), 0)
        log(os.path.join(tmp, "ft_run"), 1)
        r = holdout(os.path.join(tmp, "base"), os.path.join(tmp, "ft_run"), quiet=True)
        assert r["holdout"]["games"] == 12 and r["trained_on"]["games"] == 13
        assert r["holdout"]["delta"] > 1.7 and abs(r["trained_on"]["delta"]) < 1e-9 and r["holdout"]["wins"] == 12
        assert r["verdict"].startswith("submit the fine-tune") and "memorisation" not in r["verdict"]
        r = holdout(os.path.join(tmp, "base"), os.path.join(tmp, "base"), quiet=True)
        assert r["verdict"].startswith("inconclusive") and r["holdout"]["delta"] == 0
        assert len(CONTAMINATED) == 13 and set(CONTAMINATED) <= set(ag3_autopsy.BASELINES)
    print("selftest ok: serving_setup.py of %d repo bundle(s) patched by exactly three insertions with the self-hash "
          "updated; assemble_finetune() ran against an on-disk miniature mount, switched the model directory and "
          "refused four wrong pins and an incomplete mount; kernel metadata gains the model (RTX Pro 6000 only); "
          "holdout scoring separates the 12 unseen games from the 13 trained-on ones" % seen)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("apply")
    a1.add_argument("bundle")
    a1.add_argument("--dry-run", action="store_true")
    a1.add_argument("--mount", default=MOUNT)
    a2 = sub.add_parser("kernel")
    a2.add_argument("kernel_dir")
    a2.add_argument("--source", default=MODEL_SOURCE)
    a3 = sub.add_parser("check-mount")
    a3.add_argument("finetune")
    a3.add_argument("reference", nargs="?")
    a4 = sub.add_parser("holdout")
    a4.add_argument("baseline")
    a4.add_argument("finetune")
    sub.add_parser("pins")
    a = ap.parse_args()
    if a.cmd == "apply":
        print(apply(a.bundle, a.dry_run, a.mount))
    elif a.cmd == "kernel":
        print(patch_kernel(a.kernel_dir, a.source))
    elif a.cmd == "check-mount":
        rows, missing = check_mount(a.finetune, a.reference)
        for name, ok in rows:
            print("[%s] %s" % ("ok  " if ok else "FAIL", name))
        if missing:
            print("missing or wrong-sized:", " ".join(missing[:10]), "..." if len(missing) > 10 else "")
        sys.exit(0 if all(ok for _, ok in rows) else 1)
    elif a.cmd == "holdout":
        holdout(a.baseline, a.finetune)
    else:
        print(json.dumps(dict(PINS, model_source=MODEL_SOURCE, mount=MOUNT, contaminated=CONTAMINATED), indent=1))
