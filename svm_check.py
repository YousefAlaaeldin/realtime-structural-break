"""How do support vector machines fit this problem? One-fold experiment on the 270 cached features.

Run:    python svm_check.py                 # fold 1 (a typical fold), about 10-15 min on 6 CPU threads
        python svm_check.py --fold 0 --threads 4
Output: cv_out/svm_summary.txt

What it measures (same rows, labels, metric-matched weights and fold split as cv.py):
  1. linear SVM (hinge loss) on ALL training rows                      -> the linear ceiling; compare logistic 0.594
  2. RBF-kernel SVM approximated by a Nystroem feature map + linear SVM  -> nonlinear, scalable approximation
  3. exact RBF-kernel SVM (libsvm) on a small row sample                -> what an exact kernel SVM can afford here,
                                                                           with training time and support-vector count
  4. each SVM blended with the augmented XGBoost OOF scores (cv_out/oof_aug4_gpu.npz) -> does it add anything?
Inputs are signed-log transformed and standardised on the training fold (SVMs are scale-sensitive; trees are not).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--nys-rows", type=int, default=300000, help="training rows for the Nystroem approximation")
ap.add_argument("--nys-comp", type=int, default=1500, help="Nystroem components (kernel basis size)")
ap.add_argument("--svc-n", type=int, default=10000, help="training rows for the exact kernel SVM")
ap.add_argument("--svc-val-series", type=int, default=500, help="validation series scored by the exact SVM (it is slow)")
ap.add_argument("--blend", type=str, default=os.path.join("cv_out", "oof_aug4_gpu.npz"))
args = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

from threadpoolctl import threadpool_limits  # noqa: E402
from sklearn.kernel_approximation import Nystroem  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.svm import SVC, LinearSVC  # noqa: E402

from sbrt.augment import step_weights_mixed  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


os.makedirs("cv_out", exist_ok=True)
sys.stdout = Tee(os.path.join("cv_out", "svm_summary.txt"))


def prep(X, mu=None, sd=None):
    Z = np.sign(X) * np.log1p(np.abs(X))
    if mu is None:
        mu, sd = Z.mean(axis=0), Z.std(axis=0) + 1e-6
    return np.clip((Z - mu) / sd, -6, 6).astype(np.float32), mu, sd


def blend_table(name, s, va_mask):
    if not os.path.exists(args.blend):
        return
    b = np.load(args.blend)["oof"][va_mask]
    yv, sv = y[va_mask], step[va_mask]
    xs = b / b.std()
    ss = s / s.std()
    base = ts_auc(b, yv, sv)
    row = "  ".join(f"w={w:.1f}:{ts_auc((1 - w) * xs + w * ss, yv, sv):.4f}" for w in (0.1, 0.2, 0.3))
    print(f"  blend {name} into augmented XGBoost (alone {base:.4f}):  {row}")


z = np.load(os.path.join("data", "feat_270.npz"))
X, y, sid, step = z["X"], z["y"], z["sid"], z["step"]
tau = np.load(os.path.join("data", "train_cache.npz"), allow_pickle=True)["tau"]
n_series = len(tau)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
tr_s, va_s = list(skf.split(np.arange(n_series), (tau >= 0).astype(int)))[args.fold]
tr = np.isin(sid, tr_s)
va = np.isin(sid, va_s)
print(f"fold {args.fold}: {tr.sum():,} training rows, {va.sum():,} validation rows, 270 features, threads {args.threads}")
Xtr, mu, sd = prep(X[tr])
Xva, _, _ = prep(X[va], mu, sd)
ytr, str_ = y[tr], step[tr]
w = step_weights_mixed(ytr, str_, np.ones(len(ytr), bool))
yv, sv = y[va], step[va]
rng = np.random.default_rng(0)

with threadpool_limits(args.threads):
    # 1. linear SVM on all rows
    t0 = time.time()
    lin = LinearSVC(C=0.01, loss="squared_hinge", dual=False, max_iter=200, tol=1e-3)
    lin.fit(Xtr, ytr, sample_weight=w)
    s_lin = lin.decision_function(Xva)
    print(f"\n1. linear SVM, all {len(ytr):,} rows: TS-AUC {ts_auc(s_lin, yv, sv):.4f}  ({time.time() - t0:.0f}s)")
    blend_table("linear SVM", s_lin, va)

    # 2. Nystroem RBF approximation + linear SVM
    t0 = time.time()
    pick = rng.choice(len(ytr), min(args.nys_rows, len(ytr)), replace=False)
    nys = Nystroem(kernel="rbf", gamma=1.0 / 270, n_components=args.nys_comp, random_state=0)
    Ftr = nys.fit_transform(Xtr[pick]).astype(np.float32)
    rbf = LinearSVC(C=0.01, loss="squared_hinge", dual=False, max_iter=200, tol=1e-3)
    rbf.fit(Ftr, ytr[pick], sample_weight=step_weights_mixed(ytr[pick], str_[pick], np.ones(len(pick), bool)))
    del Ftr
    s_nys = np.concatenate([rbf.decision_function(nys.transform(Xva[i:i + 50000]))
                            for i in range(0, len(Xva), 50000)])
    print(f"\n2. RBF-kernel SVM (Nystroem, {args.nys_comp} components, {len(pick):,} rows): "
          f"TS-AUC {ts_auc(s_nys, yv, sv):.4f}  ({time.time() - t0:.0f}s)")
    blend_table("RBF (Nystroem) SVM", s_nys, va)

    # 3. exact RBF SVM on a small sample
    t0 = time.time()
    pick = rng.choice(len(ytr), args.svc_n, replace=False)
    svc = SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=2000)
    svc.fit(Xtr[pick], ytr[pick], sample_weight=step_weights_mixed(ytr[pick], str_[pick], np.ones(len(pick), bool)))
    t_fit = time.time() - t0
    nsv = int(svc.n_support_.sum())
    vs = rng.choice(va_s, args.svc_val_series, replace=False)
    vm = np.isin(sid[va], vs)
    t1 = time.time()
    s_svc = svc.decision_function(Xva[vm])
    t_pred = (time.time() - t1) / vm.sum()
    print(f"\n3. exact RBF SVM on {args.svc_n:,} rows: fit {t_fit:.0f}s, {nsv:,} support vectors, "
          f"prediction {1e6 * t_pred:.0f} microseconds per point")
    print(f"   TS-AUC on {args.svc_val_series} validation series: {ts_auc(s_svc, yv[vm], sv[vm]):.4f}  "
          f"(linear SVM on the same series: {ts_auc(s_lin[vm], yv[vm], sv[vm]):.4f}, "
          f"Nystroem: {ts_auc(s_nys[vm], yv[vm], sv[vm]):.4f})")
    if os.path.exists(args.blend):
        b = np.load(args.blend)["oof"][va][vm]
        print(f"   augmented XGBoost on the same series: {ts_auc(b, yv[vm], sv[vm]):.4f}")
    print(f"   cost of an exact SVM on the full training fold: about {(len(ytr) / args.svc_n) ** 2:,.0f}x this fit time "
          f"(kernel SVM training grows with the square of the rows or worse); prediction grows with the "
          f"support-vector count, which grows with the rows.")
print("\nsaved cv_out/svm_summary.txt")
