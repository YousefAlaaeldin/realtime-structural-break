"""Checks for sbrt/nbm.train_model(multi=...) (HANDOFF 41). CPU only, so training is bit-reproducible (about 15 s).

1. multi=4 trains exactly what diagnostics/nbm_sweep3.py trained and measured (same prepared targets, every weight
   equal, same best validation NLL), and the returned state dict keeps only the 1-step head (rows 0-1).
2. The sliced 1-step network equals the multi-head net's 1-step output exactly (torch), and the numba NbmStream
   follows torch's log scale (float64 vs float32: about 1e-7).
(The one-time check that multi=1 is bit-identical to the previous nbm.py was run on 2026-10-16 against a copy of the
old file; it is not repeatable here.)

    python diagnostics/nbm_multi_check.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # an empty string does NOT hide the GPU on this machine
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))

import numpy as np   # noqa: E402
import torch         # noqa: E402

from sbrt.data import load_train   # noqa: E402
from sbrt import nbm as NB          # noqa: E402
import nbm_sweep3 as SW             # noqa: E402


def main():
    assert not torch.cuda.is_available(), "GPU visible: these checks need CPU training"
    torch.set_num_threads(4)
    S, _ = load_train()
    hs = [S[i][1][:800] for i in range(40)]
    tr, va = hs[:32], hs[32:]

    # 1. multi=4 == nbm_sweep3
    cfg = dict(SW.REF, multi=4, epochs=2, batch=8)
    trp, vap = SW.prep(tr, cfg), SW.prep(va, cfg)
    mine = NB.prep_histories(tr, 4)
    assert all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(trp, mine)), "targets differ"
    net, best_sw = SW.train(trp, vap, cfg, 3, torch.device("cpu"), torch)
    st, best_me = NB.train_model(tr, va, epochs=2, batch=8, seed=3, verbose=False, multi=4, return_nll=True)
    sw = {k.replace("rnn.", "gru."): v for k, v in net.state_dict().items()}
    assert set(sw) == set(st), (sorted(sw), sorted(st))
    for k in st:
        ref = sw[k][:2] if k in ("out.2.weight", "out.2.bias") else sw[k]
        assert torch.equal(ref, st[k]), f"{k} differs from nbm_sweep3"
    assert best_sw == best_me, (best_sw, best_me)
    print(f"multi=4 identical to nbm_sweep3.train (best valid NLL {best_me:.5f}); "
          f"head sliced to {tuple(st['out.2.weight'].shape)}: OK")

    # 2. sliced head: torch exact, numba close
    _, NBM = NB._torch()
    n1 = NBM(64)
    n1.load_state_dict(st)
    n1.eval()
    net.eval()
    M = NB.model_arrays(st)
    prior = (448.0, 14.88, 15.26)
    worst_t, worst_n = 0.0, 0.0
    for i in range(5000, 5005):
        _, h, o = S[i]
        X, _, nh = NB.build_seq(h, o)
        with torch.no_grad():
            mf, lf = net(torch.as_tensor(X[None]))
            m1, l1 = n1(torch.as_tensor(X[None]))
        worst_t = max(worst_t, float((mf[..., 0] - m1).abs().max()), float((lf[..., 0] - l1).abs().max()))
        stream = NB.NbmStream(h, M, prior)
        lsn = []
        for x in o:
            lsn.append(stream.ST[NB.S_LS])
            stream.push(x, False)
        worst_n = max(worst_n, float(np.abs(np.array(lsn) - l1[0].numpy()[nh - 1:nh - 1 + len(o)]).max()))
    assert worst_t == 0.0, worst_t
    assert worst_n < 1e-5, worst_n
    print(f"sliced net == multi-head net's 1-step head (torch): max diff {worst_t:.1e}; "
          f"numba NbmStream vs torch log scale: max diff {worst_n:.1e}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
