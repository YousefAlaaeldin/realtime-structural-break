"""Stage the submission in its own folder, so `crunch push` uploads only what the cloud needs.

Why: `crunch push` uploads EVERY file under the folder that holds .crunchdao/ (it ignores only data/, prediction/,
.crunchdao/, __pycache__/ and *.pyc). Pushed from the research folder that would be cv_out/ (about 5 GB),
diagnostics/, verify_models/, the .bak files and probably Data/ (the ignore rule is lower-case `/data/`).
This script builds a clean folder instead (HANDOFF section 47):

    submission/
        .crunchdao/          copy of project.json + token (a credential: never share or upload this folder)
        main.py              the generated single-file submission
        requirements.txt     inference packages only (numpy, scipy, joblib, numba, xgboost, threadpoolctl),
                             pinned to the versions verified here. No torch: it is
                             imported only inside train(), after the check that skips training when the shipped
                             model set matches CONFIG, so the cloud never needs it (bare `torch` would also freeze
                             to 2.14.1+cu130, which is not on PyPI, and the cloud constraints cap torch < 2.11 on
                             Python <= 3.11)
        resources/           exactly main.FILES from the trained model set (crunch test's output)

Checks (the folder is only put in place if all pass):
  1. main.py imports standalone from the staged folder (no sbrt/ on the path)
  2. main.train() on the staged resources/ returns without importing torch, i.e. the cloud will not retrain;
     the model files are unchanged afterwards (sha256)
  3. smoke: main.infer() from the STAGED folder on the first N local test series; if the models come from resources/
     and prediction/prediction.parquet is newer than its meta.json (crunch test's output from the same model set),
     the outputs must match it to float32 precision; otherwise (or with --no-compare) only the range is checked

Run:   python prepare_submission.py                 (about 1 min, mostly numba compilation for the smoke)
       python prepare_submission.py --loose         version ranges instead of exact pins (only if the cloud log
                                                    shows that a pinned version cannot be installed)
Then:  cd submission;  crunch push --dry -m "...";  crunch push -m "..."
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata

here = os.path.dirname(os.path.abspath(__file__))

# every package main.py needs outside train(); threadpoolctl limits BLAS to one thread per worker (HANDOFF 48)
INFER_PKGS = ["numpy", "scipy", "joblib", "numba", "xgboost", "threadpoolctl"]
LOOSE = {"numpy": ">=1.26,<2.4", "scipy": ">=1.11", "joblib": ">=1.3", "numba": ">=0.60", "xgboost": ">=3.0",
         "threadpoolctl": ">=3.1"}
MARK = "staged_by_prepare_submission"                                 # lives in .crunchdao/, which push ignores

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="main.py", help="submission file to stage (copied as main.py)")
ap.add_argument("--models", default="resources", help="trained model directory (crunch test writes resources/)")
ap.add_argument("--out", default="submission", help="staging folder (replaced if this script made it)")
ap.add_argument("--smoke", type=int, default=3, help="local test series run through infer() (0 = skip)")
ap.add_argument("--loose", action="store_true", help="version ranges instead of exact pins")
ap.add_argument("--no-compare", action="store_true", help="smoke: range check only, no comparison with prediction/")
args = ap.parse_args()

# run by the checks below inside the staged folder, so `import main` finds only the staged main.py
CHILD_TRAIN = r"""
import builtins, json, sys
real = builtins.__import__
def guard(name, *a, **k):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("TRAIN_WOULD_RUN")
    return real(name, *a, **k)
import main
builtins.__import__ = guard
try:
    main.train([], sys.argv[1])
    skipped = True
except RuntimeError as e:
    if str(e) != "TRAIN_WOULD_RUN":
        raise
    skipped = False
builtins.__import__ = real
print("RESULT " + json.dumps(dict(skipped=skipped, version=main.CONFIG["version"], files=list(main.FILES),
                                  parallelism=main.INFER_PARALLELISM)))
"""

CHILD_SMOKE = r"""
import json, os, sys, time
import numpy as np
import pandas as pd
import main
x_path, p_path, n, mdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
X = pd.read_parquet(x_path)
ids = list(X.index.get_level_values("id").unique()[:n])
data = []
for i in ids:
    s = X.loc[i]
    data.append((s.loc[s["period"] == 1, "value"].tolist(), s.loc[s["period"] == 2, "value"].tolist()))
t0 = time.time()
gen = main.infer(iter(data), mdir)
next(gen)
outs = [np.array([next(gen) for _ in range(len(o))], dtype=np.float64) for _, o in data]
res = dict(ids=[int(i) for i in ids], n_points=[len(o) for o in outs], seconds=time.time() - t0,
           lo=float(min(o.min() for o in outs)), hi=float(max(o.max() for o in outs)),
           finite=bool(all(np.isfinite(o).all() for o in outs)))
if p_path:
    P = pd.read_parquet(p_path)["prediction"]
    diffs, lens_ok = [], True
    for i, o in zip(ids, outs):
        ref = P.loc[i].sort_index().to_numpy(dtype=np.float64)
        lens_ok &= len(ref) == len(o)
        if len(ref) == len(o):
            diffs.append(float(np.abs(ref - o.astype(np.float32)).max()))
    res.update(lens_ok=bool(lens_ok), max_diff=max(diffs) if diffs else None)
print("RESULT " + json.dumps(res))
"""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fd:
        for b in iter(lambda: fd.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def child(code, cwd, *argv):
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1", KMP_DUPLICATE_LIB_OK="TRUE")
    r = subprocess.run([sys.executable, "-c", code, *argv], cwd=cwd, capture_output=True, text=True, env=env)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        sys.exit("check failed:\n" + (r.stdout[-2000:] + "\n" + r.stderr[-3000:]).strip())
    return json.loads(lines[-1][7:])


def requirements():
    out = []
    for p in INFER_PKGS:
        v = metadata.version(p).split("+")[0]
        out.append(f"{p}{LOOSE[p]}" if args.loose else f"{p}=={v}")
    return "\n".join(out) + "\n"


def main_():
    src_main = os.path.join(here, args.main)
    src_models = os.path.join(here, args.models)
    out = os.path.join(here, args.out)
    tmp = out + ".partial"
    for p in (src_main, src_models, os.path.join(here, ".crunchdao", "project.json"), os.path.join(here, ".crunchdao", "token")):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    if os.path.exists(out) and not os.path.exists(os.path.join(out, ".crunchdao", MARK)):
        sys.exit(f"{out} exists and was not made by this script: pick another --out")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    # 1. stage main.py + .crunchdao, check the standalone import and read CONFIG / FILES from the staged copy
    os.makedirs(os.path.join(tmp, ".crunchdao"))
    for f in ("project.json", "token"):
        shutil.copy2(os.path.join(here, ".crunchdao", f), os.path.join(tmp, ".crunchdao", f))
    open(os.path.join(tmp, ".crunchdao", MARK), "w").write(f"made by prepare_submission.py {time.ctime()}\n")
    shutil.copy2(src_main, os.path.join(tmp, "main.py"))
    open(os.path.join(tmp, "requirements.txt"), "w", newline="\n").write(requirements())

    meta_path = os.path.join(src_models, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"no {meta_path}: train the model set first (crunch test)")
    have = json.load(open(meta_path))["config"]["version"]

    # 2. copy exactly main.FILES and check that train() would skip on them
    os.makedirs(os.path.join(tmp, "resources"))
    probe = child("import json, main; print('RESULT ' + json.dumps(dict(files=list(main.FILES), "
                  "version=main.CONFIG['version'])))", tmp)
    print(f"standalone import OK; main.py CONFIG {probe['version']}, model set {have}")
    if probe["version"] != have:
        shutil.rmtree(tmp)
        sys.exit(f"model set in {args.models}/ is {have!r} but main.py is {probe['version']!r}: run crunch test first")
    missing = [f for f in probe["files"] if not os.path.exists(os.path.join(src_models, f))]
    if missing:
        shutil.rmtree(tmp)
        sys.exit(f"model files missing in {args.models}/: {missing}")
    for f in probe["files"]:
        shutil.copy2(os.path.join(src_models, f), os.path.join(tmp, "resources", f))
    before = {f: sha(os.path.join(tmp, "resources", f)) for f in probe["files"]}
    tr = child(CHILD_TRAIN, tmp, os.path.join(tmp, "resources"))
    after = {f: sha(os.path.join(tmp, "resources", f)) for f in probe["files"]}
    extra = sorted(set(os.listdir(os.path.join(tmp, "resources"))) - set(probe["files"]))
    if not tr["skipped"] or before != after or extra:
        shutil.rmtree(tmp)
        sys.exit(f"train() would not skip on the staged model set (skipped={tr['skipped']}, "
                 f"changed={before != after}, extra files={extra})")
    print(f"train() skips on the staged model set (no torch import, files unchanged); "
          f"INFER_PARALLELISM {tr['parallelism']}")

    # 3. smoke inference from the staged folder vs crunch test's predictions
    if args.smoke > 0:
        x_path = os.path.join(here, "data", "X_test.reduced.parquet")
        p_path = os.path.join(here, "prediction", "prediction.parquet")
        # crunch test writes resources/ and prediction/ together; any other model folder (e.g. a backup) is range-checked
        same = (not args.no_compare and os.path.normcase(os.path.abspath(src_models)) == os.path.normcase(os.path.join(here, "resources"))
                and os.path.exists(p_path) and os.path.getmtime(p_path) > os.path.getmtime(meta_path))
        sm = child(CHILD_SMOKE, tmp, x_path, p_path if same else "", str(args.smoke), os.path.join(tmp, "resources"))
        print(f"smoke: series {sm['ids']}, {sum(sm['n_points'])} points in {sm['seconds']:.0f}s, "
              f"outputs in [{sm['lo']:.4f}, {sm['hi']:.4f}], finite {sm['finite']}")
        ok = sm["finite"] and 0.0 <= sm["lo"] and sm["hi"] <= 1.0
        if same:
            print(f"  vs prediction/prediction.parquet (crunch test, same model set): lengths match {sm['lens_ok']}, "
                  f"max |diff| {sm['max_diff']:.2e}")
            ok = ok and sm["lens_ok"] and sm["max_diff"] is not None and sm["max_diff"] < 1e-6
        else:
            print("  no crunch test predictions from this model set (other --models folder, older or missing "
                  "prediction.parquet, or --no-compare): range check only")
        if not ok:
            shutil.rmtree(tmp)
            sys.exit("smoke inference FAILED" + (" (if resources/ was restored from a backup, prediction.parquet is "
                                                 "from another model set: rerun with --no-compare)" if same else ""))
        for d in ("__pycache__", os.path.join("resources", "__pycache__")):
            shutil.rmtree(os.path.join(tmp, d), ignore_errors=True)

    # put the folder in place and list exactly what crunch push will upload
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(tmp, out)
    print(f"\nstaged {out}")
    total = 0
    for rel in ["main.py", "requirements.txt"] + [f"resources/{f}" for f in probe["files"]]:
        p = os.path.join(out, rel)
        total += os.path.getsize(p)
        print(f"  {rel:<24} {os.path.getsize(p) / 1e6:8.2f} MB  sha256 {sha(p)[:12]}")
    print(f"  total {total / 1e6:.2f} MB (+ .crunchdao/, which push does not upload)")
    print("requirements.txt:\n  " + requirements().strip().replace("\n", "\n  "))
    print(f'\nnext (PowerShell):  cd "{out}";  crunch push --dry -m "{probe["version"]}"')


if __name__ == "__main__":
    main_()
