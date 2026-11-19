"""Does A (backcast pseudo-past) help specifically AT THE TRANSITION? (owner's claim, 2026-10-15; HANDOFF 32.2)

TS-AUC with the positives restricted to rows k points after the break (k in 0-10, 10-30, 30-100, 100+; negatives =
all no-break rows at the same step), for signed A_mean minus plain, minus the 30-zeros control and minus the weighted
control W_5_1 (the controls of backcast_a_control.py), on all breaks and on breaks in the first 30 online points
(where the pseudo-past sits next to the break). If the backcast content helps at the transition, A_mean must beat
the ZEROS and W controls in the k 0-10 / 10-30 columns, not only plain.

Run (CPU, about 3-5 min):  python diagnostics/backcast_a_transition.py
Out: cv_out/backcast_a_transition.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

from sbrt.data import load_train      # noqa: E402
import _backcast_common as C          # noqa: E402
import backcast_a_control as AC       # noqa: E402


def main():
    say = C.Log(os.path.join(HERE, "cv_out", "backcast_a_transition.txt"))
    S, tau_all = load_train()
    X = AC.build(S, tau_all, 2000, 10000, 8)          # cols: plain e2/rough, mean e2/rough, zero e2/rough, W grid
    step, sid, y, _ = C.layout(S, tau_all, (2000, 10000))
    tau = tau_all[2000:10000]; since = np.where(tau[sid] >= 0, step - tau[sid], -1)
    wcol = {"e2": 6 + AC.W_NAMES.index("W_5_1"), "rough": 6 + len(AC.W_NAMES) + AC.W_NAMES.index("W_5_1")}
    for label, extra in (("all breaks", np.ones(len(y), bool)), ("breaks with tau < 30", (tau[sid] < 30) | (y == 0))):
        say(f"\n{label}: signed A_mean minus control, positives restricted to k points after the break")
        say("                         k 0-10           k 10-30          k 30-100         k 100-end")
        for qi, q in enumerate(("e2", "rough")):
            for nm, c in (("plain", qi), ("zeros", 4 + qi), ("W_5_1", wcol[q])):
                row = []
                for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 10**6)):
                    m = extra & ((y == 0) | ((since >= lo) & (since < hi)))
                    d = C.boot_diff(X[:, 2 + qi], X[:, c], y, step, sid, n_boot=10, mask=m)
                    row.append(f"{d[0]:+.4f}+-{d[1]:.4f}")
                say(f"  {q:5s} mean - {nm:6s}  " + "  ".join(row))


if __name__ == "__main__":
    main()
