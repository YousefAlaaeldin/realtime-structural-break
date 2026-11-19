# Handoff: ADIA Lab Structural Break Challenge, Real-Time Edition

Project folder: `C:\Users\youse\R projects\AIDA breakpoint detection benchmark`. Started 2026-10-13.
**Condensed 2026-10-17.** The full 2,468-line history (every design note, command and per-run detail) is in
`HANDOFF_full_2026-10-17.md.bak`; section numbers in brackets like [§26] point into that file.

---

## 1. Status (2026-10-19)

- **2026-10-22:** NBM history hidden states in the GRU burn-in `gru_cv.py --nbm-hist raw|white` and NBM gate activity `--nbm-gates` BUILT (section 46; ideas 1 and 3 of "like the earlier-layers idea"); smokes pass, flag off bit-identical. Not run yet.
- **2026-10-22:** XGBoost DART trees `cv.py --dart 0.1` CLOSED by the early stop (section 45): seed 0 folds 0 / 1 -0.0037 / -0.0009 vs the reference trees.
- **2026-10-22:** GRU distillation from the blend `gru_cv.py --distill 0.5` CLOSED (section 44): leaky teacher +0.0022 (2/2), but the CLEAN leave-pair-out teacher (`cv.py --lpo`) gives wavg -0.0012 / -0.0002 (0/2) vs the GRU s0 `_wavg` reference: the leaky gain was the leak. Pure label smoothing -0.0020. `cv.py --lpo K` stays as a tool for any future teacher / stacking test.
- **2026-10-22:** NBM score-test features `cv.py --nbm-score` (section 43): trees seed 0 -0.0043 / +0.0022 / +0.0005 / -0.0004 / +0.0015, **-0.0001 +- 0.0011, 3/5**; passed the early stop (fold 1 > 0); seed 1 next, then the seed pair decides.
- **2026-10-21:** joint NBM fine-tuning through the GRU `gru_cv.py --nbm-joint 0.1` CLOSED by its gate (section 42): wavg vs the GRU s0 wavg -0.0054 / -0.0016 (folds 0 / 1, 0/2); the NBM moved 13% with its NLL flat and the train loss unchanged, so supervision reshaped the state without adding anything the detector could use.
- **2026-10-21:** test-time views `cv.py --tta 51,150` CLOSED: after 3 folds x1 -0.0019, x2 -0.0005 vs the +0.0010 bar, out of reach (section 38).
- **2026-10-21:** NBM-stream distribution tests `cv.py --nbm-dist` CLOSED by the early stop: trees folds 0 / 1 -0.0023 / -0.0002 (section 41).
- **2026-10-21:** sign-reflection augmentation (teammate research G08) CLOSED by its fingerprint gate: flipped vs real 0.65-0.68 vs re-cut 0.48-0.51 (section 40; teammate research triage there too).
- **2026-10-21:** query-attention transformer `gru_cv.py --arch qattn` (section 39): wavg vs GRU s0 wavg -0.0152 / -0.0152 / -0.0105 (folds 0-2), stopped by the owner; UNDERFITS (train loss 0.644 vs GRU 0.619 at epoch 8). Retry with `--lr 5e-4 --warmup 375` (folds 0-1): wavg -0.0064 / -0.0086, train loss 0.635 / 0.631: fails both parts of its kill test, CLOSED.
- **2026-10-20 (later):** test-time views `cv.py --tta 51,150` BUILT, not run (section 38). Reversal augmentation (section 37): full fingerprint gate FAILED (reversed 0.65-0.68 vs re-cut 0.48-0.51): CLOSED.
- **2026-10-20 (later):** time-reversal augmentation `cv.py --aug-rev` + gate `diagnostics/rev_fingerprint.py` BUILT, not run (section 37).
- **2026-10-20 (late):** transductive NBM adaptation on the test stream CLOSED (section 35: ceiling +0.0016 / +0.0025
  vs the +0.003 bar, shift probe silent) and the break transplant, synthetic breaks
  learned from real ones with empirical-Bayes deconvolution (section 36: gate failed; owner's override cv.py run, trees
  folds 0 / 1 -0.0116 / -0.0042: CLOSED by the early stop).

- **2026-10-20: v4 (`blend-v4-hpca8-svm`, CV 0.6534) submitted: CLOUD 0.6474 TS-AUC** (+0.0054 vs v3's 0.6420;
  fold-0 production verify predicted +0.0043). CV-to-cloud offset -0.0060 (v3: -0.0069): CV gains carried over
  about one for one. Public board: 0.0009 below 50th place (0.6483), above szukiyu40 (0.6429).
- **LATEST (2026-10-20): GRU epoch averaging ADOPTED (owner's decision, section 29).** GRU = weights averaged over
  epochs 5-8 (`--swa 5-8`, `_wavg` files). Member: wavg - ep7 +0.0073 (s0, 5/5) / +0.0029 (s1, 3/5). Blend vs the
  production-like ep7 blend: **0.6527 vs 0.6517, +0.0006 +- 0.0004, 4/5** (1.5 SE: below the > 2 SE part of the bar;
  adopted by the owner). Weights now **0.4 / 0.4 / 0.2**. **New CV reference 0.6527** (no epoch selection; the old
  0.6534 picked each fold's best GRU epoch and is not comparable). Production build NOT done yet (section 29).
- **2026-10-20: GRU with 16 hidden-state components (section 30): CLOSED** (wavg -0.0044 +- 0.0023 vs K = 8, 1/5). K = 32
  CLOSED too (folds 0-3 -0.0057 on average, 0/4; owner stopped it in fold 4).
- **2026-10-20: small-break detector variants `--sens both` CLOSED (section 34):** trees folds 0-2 -0.0037 / +0.0015 / -0.0004, stopped by the owner.
- **2026-10-20: TCN 4th member CLOSED** (honest blend -0.0002 +- 0.0001, 0/5; section 17).
- **2026-10-20: XGBoost `--monotone` CLOSED** (folds 0 / 1 -0.0058 / -0.0047) and **`--base-margin nb_logodds` CLOSED**
  (5-fold -0.0001 +- 0.0017, 3/5; section 27). **`--lambda 30` CLOSED** (folds 0 / 1 -0.0037 / -0.0037). Still open: the production v5 verify (section 29).
  The fixed-round trees idea (lr 0.015) was DROPPED before building: the averaged round curves show production's fixed
  200 rounds already within 0.0003 of the best fixed round (section 27, "Tree round-selection check").
- **2026-10-20: literature sweep of time-series measurements (section 33): every axis is in production or closed;** 3 tail
  families never tested (exceedance clustering, quantilogram, tail index), small prior; oracle `diagnostics/tail_oracle.py`
  run: formal PASS (quantilogram, L = 600, var-matched only) but the ceiling is about +0.0001: CLOSED by the owner.
- **2026-10-20: detectability weights `cv.py --detw 0.2` CLOSED (section 31; two-seed trees -0.0007 +- 0.0011).** Option 2 (expected no-break
  drift, section 32) CLOSED by its gate: the detectors' no-break values are unpredictable from the history (R^2 ~0).

- **NEW BEST CV BLEND 2026-10-19: TS-AUC 0.6531** (+0.0043 +- 0.0009 vs 0.6489, 5/5): the GRU member now also reads
  the NBM's hidden state (8 PCA channels, `--nbm-hpca 8`, section 24); weights trees 0.4 / GRU 0.5 / SVM 0.1.
  **Production build v4 (`blend-v4-hpca8`) written 2026-10-19, not verified or submitted yet** (cloud still runs the
  0.6489 build, 0.6420): section 24, "Production v4". If the CV-to-cloud offset of the 0.6489 build holds (-0.0069), the cloud
  would read ~0.646: just below the public 50th place (0.6483).
- **NEW BEST CV BLEND 2026-10-19 (later): TS-AUC 0.6534**: the SVM also reads the NBM hidden-state PCA (`cv.py
  --nbm-hpca 8`, section 27; SVM 0.6374 -> 0.6412, +0.0037 5/5); blend +0.0003 +- 0.0001, 4/5 vs 0.6531, below the bar,
  **ADOPTED by the owner's override**. Weights unchanged 0.4 / 0.5 / 0.1. Not in production yet: to be folded into v4
  before its verify (section 27, "Production").
- **2026-10-21: CatBoost 4th member CLOSED (section 26):** low-lr alone 0.6395 (-0.0023 vs XGBoost seed 0), Spearman vs the tree pair 0.889, honest blend -0.0003 +- 0.0002, 0/5; in-sample weight 0.
- Previous best CV blend: TS-AUC 0.6489 (5-fold out-of-fold). Members, flags and reference files in section 6.
- **Production `main.py` = that blend, submitted: cloud 0.6420** (CONFIG `blend-v3-trees2`, trained model set in `resources/`), plus the fast
  inference changes (one BLAS thread per worker, `INFER_PARALLELISM = 12`, progress logging).
- **Submissions (late, practice only):**
  - #1 (run 127391, Lightweight CPU, 4 workers): still inferring after 2.5 h. Cause: 4 workers x 2 OpenBLAS libraries
    (numpy + scipy) x 16 threads each, busy-waiting between the per-point SVM products on 16 vCPUs. Abandoned.
  - #2 (run 127420, Powerful, 12 workers, fast build): every worker `openblas 16 -> 1`; about 1.5 ms/point after
    numba compile (runner overhead only ~0.16 ms/point); about 11 min of inference. **Cloud score 0.6420 TS-AUC**
    (2026-10-17).
- **CV-to-cloud offset: 0.6420 - 0.6489 = -0.0069**, much smaller than bamasa's reported -0.016 to -0.020. CV is a
  usable guide; the many small adoptions inflated it by at most a few thousandths. On the public (frozen) board 0.6420
  sits just below szukiyu40's 0.6429 and about 0.006 below 50th place (0.6483); 20th is ~0.663.
- **2026-10-17 (after the condensing):** five ideas closed, best unchanged at 0.6489: tail-matched BOCPD (already
  failed on 2,000 series in the MS self-test, §8), GRU `--aux-tsb` (§12), in-fold pruning (§13), self-replay blend
  calibration (§14), more NBM pretraining data (the NBM data learning curve is flat, §15). The rest of the open list (§8) are small bets: each expected about 0.
- **Next:** if CV gains carry over one for one (only one cloud data point so far, so unproven), reaching the public top 50 needs about +0.006-0.007
  more CV (target ~0.656). Every remaining idea and its status is in section 8.
- **2026-10-18:** TCN seed 0 run (section 17: alone = one GRU seed; in-sample blend +0.0004). Search over all 74
  OOF files from scratch, nested (section 18): -0.0015 +- 0.0005 vs the current blend, 0/5: CLOSED; the current blend stays.
  Covariate-shift weighting of the 2025 data (section 19): gate passed (ESS 1,694), trees folds 0 / 1 -0.0128 / -0.0054: CLOSED.
  Break shape / persistence / agreement / symbol transitions `--shape` (section 20): folds 0 / 1 -0.0031 / -0.0022: CLOSED.
  2026-10-19: owner's 8 papers reviewed; single-series Rio-CPD `--rio` (section 21): folds 0 / 1 -0.0003 / -0.0010: CLOSED.
  Per-step cross-sectional feature normalisation `--xsec` (section 22): folds 0 / 1 -0.0044 / -0.0027: CLOSED.
  Re-cut bagging (tree seed 1 on `--aug-seed 1`, section 23): fold 0 -0.0047, stopped by the owner: CLOSED.
  NBM hidden state (PCA 8) as GRU input `--nbm-hpca 8` (section 24): GRU s0 +0.0063, s1 +0.0075 (5/5 each); blend
  **0.6531**, +0.0043 +- 0.0009, 5/5: **ADOPTED, new best CV** (not in production yet).
  Earlier today: TCN built (section 17). New `blend.py --honest-paired`:
  judge any NEW MEMBER on cross-fitted weights (the in-sample paired line can only gain from an extra member).

---

## 2. Competition facts

| Item | Value |
|---|---|
| Interface | `train(datasets, model_directory_path)` and generator `infer(datasets, model_directory_path)`; `infer` yields once (ready), then exactly one float per online point; no look-ahead |
| Data | history 1,000-5,000 points (break-free, z-scored exactly); online 10-999 points; 50% of series have one break at an unknown online position; label = 1 once `online_index >= tau` |
| Metric | TS-AUC: AUC across series at each online step, averaged with weight n_pos x n_neg. ~84% of the weight sits at steps 100-700; steps < 100 carry ~12% |
| Timeline | ended 2026-10-01; winners at ADIA Lab symposium 26-28 Oct 2026. Late submissions: practice only, never on out-of-sample data, 15 h compute per two weeks, 5 per day |
| Rules | deterministic output (10% re-run, tolerance 1e-8); whitelisted libraries only (numpy, scipy, sklearn, lightgbm, xgboost, torch, numba, joblib, statsmodels, threadpoolctl ...); external data allowed if licensable (2025 edition data explicitly allowed, "not encouraged") |
| Leaderboard (public, frozen) | 1st 0.7168, 20th ~0.663, **50th 0.6483**. Public write-ups: szukiyu40 0.6429, bamasa 0.6299 |
| Cloud | Lightweight / Powerful = 16 vCPUs, 16 / 64 GB, Python 3.12, no GPU; X_test about 5.04M online points (same size as X_train). The runner does one unix-socket round trip per point; `INFER_PARALLELISM` (module constant) sets the worker count |

Not to be confused with the "Structural Break Open Benchmark" (offline, one score per series).

---

## 3. What the data is like

- 10,000 training series, 49.7% with a break, break position ~uniform over the online segment.
- History statistics carry no label signal (AUC ~0.5); no shared/duplicated histories; values float32-exact.
- 26% have |lag-1 autocorrelation| > 0.3, 15% are heavy-tailed (excess kurtosis > 3).
- **Breaks are subtle and mostly raise variance.** Signal is in variance, roughness, tails, dependence; mean shifts
  are almost absent. With the break position known, the best single statistic reaches ~0.60 at 300 points.
- Breaks are instantaneous switches (no transition period, no entropy bump) [§30, §35 event study].
- **The main obstacle is the no-break series:** their online variance wanders as much as a typical break moves it
  (sd of log variance ~0.41 over 100 points), and that wandering is partly predictable from the END of the history
  (corr ~0.5) -> led to the volatility-state features (+0.007) [§15.2].
- Error breakdown of the blend [§15.2]: variance rose > 35% -> 0.84; variance unchanged/lower -> 0.56-0.57 (the bulk of
  misses); ~4.4% of breaks change nothing measurable (0.49). On the "variance-matched" slice nothing tried helps; it
  looks mostly undetectable rather than a missing feature [§20].
- Learning curve [§12]: 25% / 50% / 100% of the data -> 0.595 / 0.613 / 0.627 (fold 0): **data-limited**, doubling
  data ~+0.015. Real re-cuts helped (+0.009); synthetic breaks and the 2025 data did not.

---

## 4. Environment and gotchas

- Windows 11, PowerShell; Python 3.13.9 (Anaconda): numpy 2.3.5 (MKL), pandas 2.3.3, scipy 1.16.3, sklearn 1.7.2,
  numba 0.62.1, lightgbm 4.7.0, xgboost 3.4.1 (GPU), torch 2.14.1+cu130, crunch-cli 12.1.0, threadpoolctl 3.5.0.
- GPU: RTX 5080 Laptop 16 GB. Install torch with `--index-url https://download.pytorch.org/whl/cu130` (a plain pin is
  silently "satisfied" by +cpu). 24 CPU cores.
- `$env:KMP_DUPLICATE_LIB_OK="TRUE"` before every Python command (duplicate OpenMP warning).
- **Windows GPU memory:** varying batch shapes + CUDA Sysmem Fallback make PyTorch spill VRAM into RAM and crawl. Fixes
  in code (pad to 512, `empty_cache()` per batch); recommended: NVIDIA Control Panel -> python.exe -> CUDA Sysmem
  Fallback Policy = Prefer No Sysmem Fallback. Never run two GPU jobs at once [§29.5].
- `CUDA_VISIBLE_DEVICES=""` does NOT hide the GPU here; use `-1`.
- Ctrl-C prints `forrtl: error (200)`: harmless.
- PowerShell: `Remove-Item a, b` (not `del a b`); `crunch test` runs 1 worker on Windows.
- Keep `cv.py`, `gru_cv.py`, `sbrt/*.py` in CRLF line endings when editing with scripts.
- **Secrets:** `.crunchdao/token` (also copied into `submission/.crunchdao/`) is a credential; never share.
- **Never `crunch push` from the project folder** (it uploads every file, GBs). Push only from `submission/`.

---

## 5. Working rules (owner)

- **Hand over commands:** write scripts, give PowerShell commands; the owner runs anything long (CV runs, training,
  gates, analyses over a minute). The assistant runs only quick smoke tests and file reads.
- Plain-language explanations, honest verdicts (including when something fails or an earlier claim was wrong).
- GPU for heavy training.
- **Accuracy first:** inference time is not a reason to drop or handicap an accuracy gain while a submission stays
  inside the 15 h / two weeks budget.
- **STANDING PASS RULE (2026-10-16):**
  1. Paired `cv.py` trees run vs the current best trees. **Early stop:** folds 0 and 1 both <= 0 paired -> stop, close.
  2. If the trees' 5-fold paired mean > 0: `blend.py` with the new member(s) vs the current best blend.
  3. **ADOPT if the blend gains >= +0.0005, in >= 4/5 folds, and > 2 SE.** A change that also feeds the GRU/SVM is
     adopted only on the blend with those members rerun (a trees-only blend is the go/no-go proxy).
  - Noise floors: one tree seed vs another = paired 0.0000 +- 0.0009 (per-fold sd ~0.0019) [§44.1]; GRU seeds vary
    ~+-0.01 per fold. New tree candidates should come with two seeds, compared seed-average vs seed-average
    (`diagnostics/oof_paired.py --a C_s0 C_s1 --ref B_s0 B_s1`).
  - Caveat: a low bar applied many times to the same 5 folds slowly inflates CV; only a submission checks that.
- Before 2026-10-16 the bar was paired >= +0.003, > 2 SE, >= 4/5 folds (GRU on both seeds).

---

## 6. The current best model (CV 0.6534 since 2026-10-19; the 0.6489 version below is the one in production)

**2026-10-19 change:** the GRU pair is replaced by the GRU with `--nbm-hpca 8` (seeds 0 and 1:
`cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz` + `..._hpca8_s1.npz`, pair 0.6451, now the strongest single member); blend
weights **trees 0.4 / GRU 0.5 / SVM 0.1** (flat plateau: six weight sets within 0.6526-0.6531). Trees and SVM
unchanged. GRU command: the GRU command below plus `--nbm-hpca 8`.
**2026-10-19 change 2 (owner's override, section 27):** the SVM is replaced by the SVM with `--nbm-hpca 8`
(`cv_out/oof_svm_nbm_hpca8.npz`, 0.6412; 114 inputs = the old 106 + `nbh_pc0..7`; best lambda 0.3 on the average
curve). Blend **0.6534**, weights 0.4 / 0.5 / 0.1. SVM command: the SVM command below plus `--nbm-hpca 8`.

**Blend (0.6489 version):** each member's raw margin/logit divided by its in-sample sd, weights **trees 0.5 / GRU 0.3 / SVM 0.2**,
then a sigmoid. Weights are flat over the grid; honest cross-fitted weights cost ~0.001 [§21]; per-step weights,
stacking and temporal smoothing all lose.

| Member | What | CV alone | OOF reference files |
|---|---|---|---|
| Trees | XGBoost (GPU), 2 seeds averaged; 321 columns; leaves 31, min-leaf 2000, lr 0.03, ff 0.5, subsample 0.8, lambda 10; production 200 rounds | 0.6443 (2 seeds) | `cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz` + `..._s1.npz` |
| GRU | Linear -> GELU -> GRU(64) -> MLP; 53 input channels; 2 seeds | 0.6354 (pair) | `cv_out/oof_gru_aug2_vol_bo_nbm.npz` + `..._s1.npz` |
| SVM | least-squares kernel SVM, 8192 random Fourier features, gmul 0.5, lambda 0.3, top-100 tree columns + 6 context | 0.6374 | `cv_out/oof_svm_nbm.npz` |

**2026-10-20 change 3 (owner's decision, section 29):** the GRU pair is the epoch-averaged one (weights of epochs 5-8,
`cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz` + `..._swa_s1_wavg.npz`, pair 0.6424); weights **0.4 / 0.4 / 0.2**;
blend **0.6527** = the new reference. GRU command: the GRU command below plus `--nbm-hpca 8 --swa 5-8 --fixed-ep 7`;
compare a new GRU's `wavg` line against the `_wavg` files (same seed, same 8 epochs, so the runs are paired).
**Paired blend command from 2026-10-20 (reference = the 0.6527 blend):**
`python blend.py --a <trees s0 s1> --b <GRU s0 s1> --c <SVM> --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`

**Paired blend command for the next candidate (reference = the 0.6534 blend from 2026-10-19; superseded):**
`python blend.py --a <new trees s0> <new trees s1> --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`
(add `--honest-paired` when the candidate is an EXTRA member.) Old command (0.6489 reference):
`python blend.py --a <new trees s0> <new trees s1> --b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --c cv_out/oof_svm_nbm.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --ref-c cv_out/oof_svm_nbm.npz`

**Current best tree command (seed 0):**
`python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag <tag>`
(add `--seed 1` for the second seed). GRU: `gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --threads 8 --device cuda`.
SVM: `cv.py --backend svm --threads 6 <same feature flags> --svm-top 100 --svm-gmul 0.5 --svm-imp cv_out/importance_aug2_df_bo_nbm_bs24_ep25.csv`.

**Feature families (all streaming, causal, numba; streaming == batch verified for each):**
- **Kernel, 270** (`sbrt/features.py`): per-series history params (mean/sd, AR(1)/AR(10), ECDF, deciles, long-run
  variances); 32 increment series (z, z^2, lag products, AR(10) residual and square/abs/tails, ranks, variogram lags
  1-16, tail flags 1-4 sigma, skew/kurtosis terms, sign persistence, ...); per step: standardised running mean, windows
  10/30/100, and a scan max/min over all start times; decile chi-square, EWMAs, 5 history descriptors (`ctx_`).
- **Evidence, 6** (`--ev compact`, `sbrt/evidence.py`): CUSUM / Shiryaev-Roberts and a Bayesian mix over 14 break
  alternatives on the history-ECDF-whitened AR residual.
- **Volatility state, 13** (`--vol`, `sbrt/volstate.py`): GARCH(1,1) fitted on the history, online residuals
  standardised by its FROZEN forecast path decaying to a slow end-of-history level (EWMA half-life 200); running
  means, windows, scans, realized/forecast ratios, 4 per-series constants.
- **Distribution tests, 16** (`--dist full --drop dist_z_`, `sbrt/disttest.py`): KS / Anderson-Darling / CvM /
  Wasserstein of online vs history for the AR residual e and the vol-adjusted u (all points, last 100, scans).
- **BOCPD, 10** (`--bocpd`, `sbrt/bocpd.py`): exact one-break Bayesian posterior (history = known pre-break regime,
  Normal-Inverse-Gamma post-break with an empirical-Bayes prior fitted on training breaks, generator hazard prior) on
  two whitened streams (w, wu): logodds, logbf, time since break, log variance, shift.
- **NBM, 6** (`--nbm`, `sbrt/nbm.py`): a small GRU (11 inputs, width 64, ~30k params) pretrained self-supervised on
  break-free histories to predict the next AR residual as a Student-t (batch 24, 25 epochs, lr 2e-3); its whitened
  online prediction errors go through the BOCPD posterior with its own EB prior; plus scale drift. Per CV fold it is
  pretrained on that fold's original training histories. `nb_lvr` is the top tree feature.
- **Training data:** originals + 2 boundary re-cuts per series (`--aug 2`, `sbrt/augment.py`: move the
  history/online boundary, real values, exact labels; folds split on originals, re-cuts train only with their parent).
- **Row weights:** one row per (series, every 3rd step); each step class-balanced and weighted like the metric.
- GRU inputs: z, AR residual + derived channels, 6 history descriptors, online flag, log t (14) + vol 3 + BOCPD 10 +
  NBM 6 + top-20 kernel features; last 100 history steps warm up the state; production trains 6 epochs.

---

## 7. How we got here: adopted steps (TS-AUC, 5-fold CV)

| Date | Change | Member result | Blend | Notes |
|---|---|---|---|---|
| 10-13 | LightGBM on 206 kernel features | 0.6120 | | |
| 10-13 | + localisation features (270) | 0.6155 | | +0.0017, kept |
| 10-13 | GRU with top-20 kernel features | 0.6152 | 0.6244 (50/50 logit average) | plain logit average works at inference (no per-step standardisation) |
| 10-13 | XGBoost GPU + boundary re-cuts `--aug 2` | 0.6233 (+0.0093, 5/5) | | biggest data gain; `--aug 4` only +0.0007 more |
| 10-13 | GRU `--aug 2` (2 seeds) | 0.6241 | **0.6336** | |
| 10-13 | trees + evidence `--ev compact` | 0.6267 (+0.0030) | **0.6345** | borderline, kept (cheap) |
| 10-14 | volatility state `--vol` (trees + GRU) | trees 0.6339 (+0.0070, 5/5); GRU pair 0.6278 | **0.6396** (+0.0050, 5/5) | new information beats new algorithms |
| 10-14 | distribution tests `--dist full --drop dist_z_` | trees 0.6345 (+0.0011) | **0.6413** (+0.0017, 4/5) | owner adopted below the then +0.003 bar |
| 10-14 | kernel SVM as third member | SVM 0.6328 | **0.6427** (+0.0012, 4/5) | weights 0.4/0.4/0.2 then |
| 10-14 | BOCPD `--bocpd` (trees, then GRU) | trees 0.6370 (+0.0019, 5/5, 6 SE); GRU +0.004/+0.007 | **0.6438** | weights now 0.5/0.3/0.2 |
| 10-15 | NBM `--nbm` (trees) | trees 0.6417 (+0.0049, 5/5) | **0.6468** (+0.0032, 5/5) | |
| 10-15 | SVM with NBM inputs | SVM 0.6374 (+0.0049, 5/5) | **0.6473** (+0.0005, 5/5) | free at inference |
| 10-15 | GRU with NBM channels | GRU pair 0.6354 (+0.0037) | **0.6483** (+0.0007, 3/5) | owner override |
| 10-16 | two tree seeds averaged | trees +0.0015 (5/5) | **0.6489** (+0.0004, 3/5) | owner override; in production |
| 10-19 | GRU reads the NBM's hidden state (8 PCA channels, `--nbm-hpca 8`; owner's "earlier layers" idea) | GRU s0 +0.0063, s1 +0.0075 (both 5/5, ~5 SE); pair 0.6451 | **0.6531** (+0.0043 +- 0.0009, 5/5) | passes the bar by ~8x; largest blend step since the volatility state; weights 0.4/0.3->0.5/0.1; NOT in production yet |
| 10-19 | SVM reads the hidden-state PCA too (`cv.py --nbm-hpca 8`) | SVM 0.6412 (+0.0037, 5/5) | **0.6534** (+0.0003, 4/5) | owner override |
| 10-20 | GRU weights averaged over epochs 5-8 (`--swa 5-8`) instead of a fixed/best epoch | wavg - ep7 +0.0073 (s0, 5/5), +0.0029 (s1, 3/5); pair 0.6424 | **0.6527** vs ep7 blend 0.6517 (+0.0006 +- 0.0004, 4/5) | owner's decision (1.5 SE); no epoch selection, so 0.6527 replaces the selection-optimistic 0.6534; weights 0.4/0.4/0.2 |

Pattern: each new feature family first gave large single-feature gains, then shrank ~3-5x into the trees and again
~2x into the blend. Member upgrades that only copy what the trees already have (BOCPD/NBM into GRU, seeds) barely
move the blend.

---

## 8. Tried and closed (do not repeat without a NEW mechanism)

Verdicts against the rule in force at the time. "Paired" = per-fold paired difference vs the then-best member.

**Data / augmentation**
| Idea | Result | Section |
|---|---|---|
| `--aug 4` vs `--aug 2` | +0.0007, n.s.; returns flatten | §13 |
| Synthetic breaks injected into real data (7 types, calibrated) | trees -0.0022 / +0.0001; GRU -0.0033: dropped | §14 |
| 2025-edition data (10,001 new series, different population: heavier tails, 29% breaks) | trees -0.0057; source-balanced -0.0006; covariate-shift weighted + balanced (ESS 1,694) folds 0 / 1 -0.0128 / -0.0054 (2026-10-18) | §15.4, section 19 |
| History-trimming re-cuts `--aug-trim` | -0.0005 | §26 |
| Early-break re-cuts `--aug-early` | early stop: -0.0031 / -0.0001 | §43 |
| Time-reversal augmentation `--aug-rev` (each regime reversed) | fingerprint gate: reversed vs real 0.65-0.68 (re-cut 0.48-0.51; top `vol_lvh1000`, `vol_lvh200`, `ctx_skew`): the series are not time-reversible in their volatility path; closed 2026-10-20, never run in cv.py | section 37 |
| Sign-reflection augmentation (-history, -online), teammate research G08 | fingerprint gate: flipped vs real 0.65-0.68 (re-cut 0.48-0.51); top `ctx_skew` with `vol_persist` (vol columns themselves sign-invariant): skew sign is tied to the vol dynamics; closed 2026-10-21, `--aug-flip` never built | section 40 |
| Distribution tests (KS / AD / CvM / W1, scans) on the NBM's whitened conditional-PIT stream `--nbm-dist` (teammate research G04-lite) | trees folds 0 / 1 -0.0023 / -0.0002: closed by the early stop 2026-10-21 | section 41 |
| Test-time views `--tta 51,150` (boundary moved 51 / 150 points earlier, logits averaged with the plain score) | same-model paired, folds 0-2: x1 -0.0045 / -0.0003 / -0.0009, x2 -0.0026 / +0.0006 / +0.0004; +0.0010 bar out of reach (2026-10-21) | section 38 |

**Models, training, combination**
| Idea | Result | Section |
|---|---|---|
| Tree ranking objective (rank:pairwise) | -0.017 on fold 0 | §14 |
| Tree settings sweep (leaves 15/63, min-leaf 500/8000, colsample 0.3/0.8, lr 0.015) | none >= +0.001; day-one settings are a local optimum. lr 0.015 makes the round curve flat 400-1000 (production robustness idea, not adopted) | §44.3 |
| Logistic regression member | 0.594, +0.002 blend (pre-SVM) | §7 |
| GRU knobs (ramp, mem-bias, hidden, dropout, lr, pos-boost), feat_top 40 | inside seed noise / worse | §7 |
| GRU capacity (2-3 layers, width 96-128) and input gates (prior/static/dynamic) | nothing: data-limited, not capacity-limited | §14 |
| GRU pairwise rank loss | -0.0011 | §14 |
| GRU auxiliary target `--aux-tsb 0.2` (predict log time since break on positive steps) | seed 0 -0.0003 +- 0.0009, 3/5: closed 2026-10-17 | section 12 |
| In-fold feature pruning `--prune-keep 150` (top 150 of 321 by first-200-tree gain, refit) | folds 0 / 1 -0.0042 / -0.0020: closed by the early stop 2026-10-17 | section 13 |
| Self-replay calibration of the final blend logit (history-suffix replays through the shipped blend; mean shift / z-score) | fold 0, cross-fitted -0.0004, 3/10 halves > 0: closed 2026-10-17. Replays predict the no-break score only weakly (Spearman 0.16-0.27) | section 14 |
| GRU top-20 from all tree features (`--feat-src all`) | -0.0008 | §24 |
| GRU seeds 2-3 (4-seed GRU) | blend +0.0002 | §24 |
| SVM: more random features (16k), BOCPD inputs, wider kernel (gmul 0.25/0.1) | 0.0000 / -0.0001 | §19, §26 |
| Second tree model (vol-only trees) as 4th member | weight 0 | §21 |
| CatBoost (symmetric depth 6, all 321 columns, lr 0.015) as 4th member | alone 0.6395 (-0.0023, 1/5); Spearman vs trees 0.889; honest blend -0.0003 +- 0.0002, 0/5; in-sample weight 0 (2026-10-21) | section 26 |
| Stacking (LightGBM on member scores, +/- step/context) | -0.0013 to -0.0044, 0/5 | §23 |
| Per-step-range blend weights | -0.0000 | §21 |
| Temporal smoothing of the score (EWMA, running max) | all negative | §12, §27 |
| Near-break training weights `--near-k` | -0.0019 / -0.0007 / -0.0006 | §32.x |

**Feature families**
| Idea | Result | Section |
|---|---|---|
| Finer tails + extra windows (388 features) | -0.005: redundant features hurt | §7 |
| 32 extra history descriptors (ACF, ARCH, spectra, quantiles ...) | no gain | §12 |
| "Predict the next point" better (AR orders, adaptive scale, t-surprise, random walk/EWMA, nonlinear GBM correction) | AR(10) residual is all the linear structure; nonlinear correction explains nothing | §7 |
| Evidence `--ev full` (48 cols) | not better than compact | §14 |
| Distribution tests on raw z | weak (autocorrelation inflates them) | §17 |
| Per-series null calibration by replaying the history | no-go: vol already does it; hurts dist/tail stats | §18 |
| Ordinal patterns / permutation entropy, nonlinear volatility (leverage, clustering lags 2-20, irreversibility) | no signal in the oracle | §20 |
| Streaming MMD on residual pairs/triples `--mmd` | -0.0004; duplicates `--dist` (rank corr 0.8-0.9) | §21 |
| Matrix profile / FLUSS / mFLUSS, kernel change-point (KCP) | no signal / variance in disguise | §24 |
| Richer BOCPD `--bocpd2` (AR coefficient change, tail-frequency stream, per-type priors) | -0.0002 | §28 |
| Learned per-point density ratio `--lratio` | closed by analogy with the pair classifier, never run | §38.6 |
| NBM pretraining knobs (width, batch, lr, epochs, wd), design sweep (2 layers, LSTM, TCN, k-step target, extra inputs) | plateau; only multi4 passed the gate | §29.4, §37 |
| NBM `multi4` (multi-horizon pretraining) | gate +0.0047, trees -0.0009 / -0.0009: closed; head streams and NBM seed ensemble dropped with it | §41 |
| NBM per-series fine-tuning (test-time adaptation) | held-out NLL gets WORSE; closed | §40 |
| End-of-history whitening reference for NBM | no win | §40 |
| NBM pretraining on more break-free data (2025 stretches, break-free online parts), checked by the NBM data learning curve (7,500 / 3,750 / 1,875 histories x 3 seeds, matched optimiser steps) | nb_logodds 0.6301 / 0.6333 / 0.6322; 100% - 50% = -0.0032 +- 0.0017 (SE); valid NLL flat: at its information limit, not data-limited. Closed 2026-10-17, the pretraining itself never built | section 15 |
| Chronos-2 / Chronos-Bolt (forecast-surprise stream; encoder drift) | surprise 0.589 single; ~55 h CPU per 10k series (over budget); gate instrument blind: closed | §29.4, §35.1 |
| Entropy: rolling permutation/value entropy, transition bump, binned-entropy drop, posterior shape `--pshape`, history entropy | no rise at breaks; pshape trees -0.0001/-0.0011/-0.0003 | §30, §31b, §32 |
| Backcasting (predict the past; pseudo-past prepended; linear and neural) | series are time-reversible: same information, later | §31, §32.1 |
| Embedding drift (contrastive CNN, learned metric on statistics `--femb`/`--emb`, thresholds) | trees +0.0001: closed (gate false positive) | §33 |
| Long-range structure (variance ratios to lag 64, DFA, low-frequency power, lags 11-50) | no usable information | §34 |
| Pair classifier scan `--pc` ("is this whole window post-break?", 2025-winner style) | single 0.613 but trees -0.0001, 2/5 | §38 |
| Data-generating equation: nonlinear mean (R1), richer one-step density (R2), frozen variance forecasts `--eqf` (R3) | R1/R2 nothing; R3 trees -0.0005: closed | §39, §42 |
| Markov-switching model of normal `--ms` | MS fits histories worse than GARCH-t; trees -0.0008, 2/5 | §45, §46.3 |
| BOCPD with a Student-t, tail-matched break model (the MS self-test "twins": updated GARCH / frozen GARCH / constant t null) | full self-test, 2,000 series (`cv_out/msvol_selftest.txt`): twins 0.5953 / 0.5930 / 0.5980 vs production bo_w 0.6046, bo_wu 0.6004; pre-written bar (twin a or b >= bo_wu + 0.005 = 0.6054) failed by ~0.01. Like for like (twin c vs bo_w, twin b vs bo_wu) the parametric t loses ~0.007 to ECDF whitening + NIG. The 200-series "win" was noise. Real but tree-fixable: bo_ over-flags heavy-tailed no-break series (top kurtosis quintiles +3.5 to +4.0 log-odds; twin a flat). Closed 2026-10-17 without a cv.py run (owner's call) | §45 |
| Derivative statistics (|dz|, |d2z|, tails of dz, higher orders) | robustness production already has; closed by the control | §46 |
| Re-cut bagging: tree seed 1 trained on a fresh re-cut draw `--aug-seed 1` (FAJ 2019 forecast combination over training sets) | fold 0 -0.0047 vs the old seed 1, stopped (2026-10-19) | section 23 |
| Per-step cross-sectional feature normalisation `--xsec` (robust z per step bin, FAJ 2019 stock-selection practice) | folds 0 / 1 -0.0044 / -0.0027; 5-fold -0.0011 +- 0.0021, 2/5 (2026-10-19) | section 22 |
| MiniRocket-style convolution exceedance on the whitened streams w / wu, history-calibrated (`diagnostics/conv_oracle.py`) | oracle gate ALL -0.0203 +- 0.0043, FLAT -0.0287 (2026-10-19); the columns are signed variance detectors (UP 0.78-0.81, DOWN 0.34-0.41), FLAT single best 0.577 < blend 0.603 | section 28 |
| GRU with 16 NBM hidden-state PCA channels instead of 8 (`--nbm-hpca 16 --swa 5-8`) | wavg vs the K = 8 wavg -0.0044 +- 0.0023, 1/5 (2026-10-20); lower train loss, worse validation: overfits | section 30 |
| GRU with 32 hidden-state PCA channels (~97.5% of the variance, effectively the full state; `--nbm-hpca 32 --swa 5-8`) | wavg vs K = 8 per fold -0.0054 / -0.0037 / -0.0083 / -0.0055 (folds 0-3, 0/4; stopped in fold 4, 2026-10-20); train loss lower still: overfits more | section 30 |
| TCN as a 4th blend member (`gru_cv.py --arch tcn`, seed 0; same inputs as the GRU) | alone 0.6353 (= one GRU seed); honest cross-fitted blend vs 0.6527 -0.0002 +- 0.0001, 0/5 (2026-10-20) | section 17 |
| XGBoost monotone evidence (`--monotone`, 23 one-sided columns may only raise the score) | folds 0 / 1 -0.0058 / -0.0047; 5-fold -0.0011 +- 0.0017 (2026-10-20) | section 27 |
| XGBoost residual boosting from nb_logodds (`--base-margin nb_logodds`) | 5-fold -0.0001 +- 0.0017, 3/5; averaged curve best 0.6414 (300-450 rounds) vs reference 0.6417 (150) (2026-10-20) | section 27 |
| XGBoost reg_lambda 30 (default 10) | folds 0 / 1 -0.0037 / -0.0037, stopped by the owner in fold 2 (2026-10-20) | section 27 |
| Expected no-break drift of the detectors from richer history inputs (option 2; NBM end state + end-of-history window stats) | gate: no-break detector values unpredictable even from what the trees have (R^2 ~0); richer inputs add <= 0.02-0.05, kill test FAIL (2026-10-20) | section 32 |
| Detectability weights `--detw 0.2` (down-weight positives of breaks that change little on 5 axes, known break position, training only) | seed 0 +0.0001, seed 1 -0.0009; two-seed pair -0.0007 +- 0.0011, 3/5 (2026-10-20) | section 31 |
| Tail families `diagnostics/tail_oracle.py`: exceedance clustering / extremal index, quantilogram, Hill index (scale-free, history-relative oracle) | clustering and Hill nothing at any L; quantilogram flat at L <= 300, formal PASS only at L = 600 on the matched slice (+0.0252 +- 0.0064), where 1.5% of the TS-AUC weight lies: ceiling ~+0.0001 (2026-10-20, owner closed) | section 33 |
| Single-series Rio-CPD `--rio` (geodesic distance of lag-covariance matrices to the history's, history-calibrated CUSUM) | folds 0 / 1 -0.0003 / -0.0010 (2026-10-19) | section 21 |
| Break shape (step vs smooth ramp), location persistence, detector agreement, symbol transitions `--shape` | folds 0 / 1 -0.0031 / -0.0022 (5-fold +0.0003 +- 0.0014, 3/5); 1.7% of gain in total | section 20 |
| Small-break detector variants `--sens both` (evidence on 14 small alternatives; BOCPD with a narrow variance prior) | single features +0.014 / +0.008 vs production; trees folds 0-2 -0.0037 / +0.0015 / -0.0004, stopped by the owner (2026-10-20) | section 34 |
| Number-format fingerprints, shared histories | none exist | §12, §20 |
| GRU distillation from the blend `gru_cv.py --distill 0.5` (soft targets (1 - a) y + a p, p = Platt-calibrated teacher on original training series) | leaky OOF teacher +0.0022 (2/2); constant teacher (pure smoothing) -0.0020; leaky trees+SVM teacher +0.0018; CLEAN leave-pair-out trees+SVM teacher (`cv.py --lpo`) -0.0012 / -0.0002, 0/2: the gain was the leak (2026-10-22) | section 44 |
| XGBoost DART booster `--dart 0.1` (skip_drop 0.5, 800 rounds) | seed 0 folds 0 / 1 -0.0037 / -0.0009 (best rounds 500 / 800); ~250 s per fold (2026-10-22) | section 45 |
| Joint NBM fine-tuning through the GRU detector `gru_cv.py --nbm-joint 0.1` (label loss trains the NBM behind the 8 hidden-state channels; NLL anchor 0.2; truncated backprop) | GRU s0 wavg vs the `_wavg` reference -0.0054 / -0.0016 (folds 0 / 1, 0/2); NBM moved 13%, NLL flat, train loss unchanged (2026-10-21) | section 42 |

**Never built (ideas left):** a per-source lower weight for early-break
re-cuts [§43]; production lr 0.015 for a flatter round curve [§44.3]; the 2025 data with `--ext-rel 0.5` [§15.4]. Covariate-shift weighting of the 2025 data: section 19.
Also open, found on the 2026-10-17 re-read: GRU lower
lr / fewer epochs (augmented runs overfit after epoch 4-11) [§13]; `--aug-trim-only 1` [§25]; NBM hidden state or
m / log s per step as GRU inputs, i.e. the pretrained-GRU-plus-head version [§29.6]; a frozen-state NBM stream [§29.6]
(likely a repeat of the closed `--eqf` R3 frozen forecasts); SVM K = 150 inputs [§26]; a TCN member (used by bamasa and szukiyu40; built 2026-10-18, section 17); `--lratio`
(cached, never run; closed by analogy with the pair classifier) [§28, §38.6].

---

## 9. Lessons (process)

- Never trust the 100-series local score (bootstrap sd ~0.05). Use `cv.py` on 10,000 series.
- New INFORMATION (re-cuts, volatility state, BOCPD, NBM) moved the score; new ALGORITHMS on the same information
  (stacking, ranking, bigger nets, more members) did not.
- A strong, weakly correlated single feature is not necessarily new to the trees: the trees substitute it for columns
  that carried the same information (pair classifier, embedding drift, multi4).
- Cheap gates fail both ways: a gate with a weak base gives false positives (embedding drift +0.004 -> 0.000); a gate
  stacked on the OOF score is blind (positive control lost 0.004). **Use the paired `cv.py` run as the test.**
- Single-seed leaders in sweeps read ~0.006 high; seed-repeat before choosing.
- A single-seed baseline's own luck enters every paired comparison (fold 3 positive for every tree config).
- Validation NLL of the NBM is not the selection criterion; downstream detection is.
- `cv.py` OOF uses each fold's best round (slightly optimistic); production uses fixed rounds/epochs (fold-0 verify:
  blend 0.6632 vs research 0.6644).
- Every new flag is checked "off = bit-identical to before" and streaming == batch == causal before a full run.
- My estimates were wrong several times (local noise, run times, the memorisation theory, cloud speed). State them as
  estimates and let runs decide.

---

## 10. Production pipeline and submission

**Files:** `sbrt/*.py` (research modules) -> `make_submission.py` embeds features, augment, evidence, volstate,
disttest, bocpd, nbm + baked column lists (tree 321, GRU 53 / top-20 hash `b740fa`, SVM 106 from
`importance_aug2_df_bo_nbm_bs24_ep25.csv`) + `submission_template.py` into **`main.py` (generated; never edit by
hand; re-run `python make_submission.py` after any template/module change)**.
- `train()`: skips if `resources/meta.json` matches CONFIG; otherwise trains the NBM (GPU, ~200 s), features for all
  series + 2 re-cuts, 2 XGBoost seeds x 200 rounds, SVM, 2 GRU seeds x 6 epochs; ~27 min on the local GPU.
- `infer()`: per series a `SeriesScorer` (all streams + numba tree evaluator + numba GRU + numpy SVM); limits BLAS to
  1 thread (threadpoolctl); logs `us/point wall` and `us/point own` every 20 series per worker.
- Cost: ~650-720 us/point locally (SVM ~230-260, GRU ~100, setup ~180); cloud ~1.5 ms/point.
- v4 (2026-10-19, `blend-v4-hpca8`, CV 0.6531): the GRU reads the NBM hidden-state PCA channels (section 24). v3
  backups: `submission_template_v3.py.bak`, `verify_submission_v3.py.bak`, `make_submission_v3.py.bak`,
  `main_v3b_submitted.py.bak` (= submission #2), `resources_v3/`.
- Backups: `main_v3_submitted.py.bak` (= submission #1), `main_v2.py.bak` / `submission_template_v2.py.bak` /
  `verify_submission_v2.py.bak` (0.6483 build), `resources_v2/` (its trained models).
- GPU training is bit-reproducible here (v3 retrain gave byte-identical svm/gru/nbm files).

**Checks:** `verify_submission.py --n-train 600 --n-val 200 --parity 5` (parity streaming vs batch, determinism;
quality numbers meaningless at that size); full `verify_submission.py` (~30 min: fold-0 production vs research,
the only check against research OOF); `crunch test` (train + infer 100 local series + determinism).

**Submission procedure:** `python prepare_submission.py` stages `submission/` (main.py, requirements.txt pinned:
numpy 2.3.5, scipy 1.16.3, joblib 1.5.2, numba 0.62.1, xgboost 3.4.1, threadpoolctl 3.5.0; `resources/` = the trained
6 files; checks standalone import, train() skips without torch, staged outputs == crunch test predictions) ->
`cd submission; crunch push --dry -m "..."` (code files must be only main.py + requirements.txt) -> `crunch push -m "..."`
-> on the site: Powerful instance, "Force first train" No. The cloud installs from requirements.txt (pins kept;
crunch-cli installed with --no-deps), never trains, needs no GPU/torch.

**Cloud log reading:** each worker's `BLAS/OpenMP threads ... -> ...` line; progress lines (judge speed by the
difference between the 2nd and 3rd lines; the first includes numba compile); kill if
wall us/point x 0.007 (= minutes for ~420k points per worker at 12 workers) is well over 60.

---

## 11. Repository map (main pieces)

| Path | Purpose |
|---|---|
| `sbrt/data.py`, `sbrt/metric.py` | data loading/cache, `ts_auc` |
| `sbrt/features.py` | kernel (run `python check_features.py` after any change) |
| `sbrt/augment.py`, `evidence.py`, `volstate.py`, `disttest.py`, `bocpd.py`, `nbm.py` | adopted feature/augmentation modules (each has a self-test: `python -m sbrt.<name>`) |
| `sbrt/synth.py`, `external.py`, `replay.py`, `mmdstream.py`, `bocpd2.py`, `lratio.py`, `pshape.py`, `embfeat.py`, `pairclf.py`, `eqfore.py`, `msvol.py` | closed experiments (code and caches kept, flags in cv.py) |
| `cv.py` | 5-fold trees (LightGBM / XGBoost `--gpu`) or SVM (`--backend svm`); `--compare` paired; writes `cv_out/summary_/importance_/oof_<tag>` |
| `gru_cv.py` | GRU, same folds/layout |
| `blend.py` | 2-4 member blends, `--ref-*` paired vs a reference blend, `--honest` cross-fitted weights |
| `diagnostics/oof_paired.py` | paired comparison of seed-averaged OOF files |
| `diagnostics/` | gates, oracles, sweeps (`info_oracle.py` is the oracle template), `infer_speed.py` (multi-worker inference speed, `--out`) |
| `submission_template.py`, `make_submission.py`, `verify_submission.py`, `prepare_submission.py` | production |
| `viz/viewer.html` (+ `export_viz.py`), `viz/plot_derivative.py` | signal viewer for intuition |
| `data/` | parquet files, `data/ext2025/` (2025 edition), feature caches (GBs; rebuild automatically) |
| `reports/literature_review_reading_list.md` | literature review of change-point strategy families |
| `structural-break-cheatsheet.html`, `approach-ledger.html` | primers |

---

## 12. GRU auxiliary target `--aux-tsb 0.2` (2026-10-17; RESULT: CLOSED at step 1)

**RESULT seed 0 (owner's run 2026-10-17, `cv_out/gru_summary_aug2_vol_bo_nbm_auxtsb.txt`):** 0.6355 vs 0.6359; paired
**-0.0003 +- 0.0009, 3/5** (folds +0.0018 / +0.0006 / -0.0021 / -0.0027 / +0.0007). Slightly lower in every step range
(<100 0.5740 vs 0.5771, 100-300 0.6251 vs 0.6263, 300-600 0.6488 vs 0.6492, 600+ 0.6699 vs 0.6721). Paired mean <= 0
-> **CLOSED** by the rule below; seed 1 and the blend not run. Side facts from the log: with the NBM channels cached a
GRU fold now takes 14-27 s (a whole 5-fold seed a few minutes, not ~25 min), and validation still peaks at epochs
3-7 and then falls while the train loss keeps dropping (the overfitting noted in §13 of the old file).

**What:** a second linear head on the GRU's hidden state learns log(1 + time since break) / 7 on positive steps (MSE,
weight 0.2), next to the main break/no-break head. The idea is that being forced to estimate how long ago the break
happened shapes the hidden state into a better break detector. Training only: the shipped model uses the main head
alone, so **inference cost is unchanged**. Built 2026-10-13 [§14]; never run in full.
**Check 2026-10-17:** smoke run with today's flags (`--n-series 300 --folds 1 --epochs 2 --aug 2 --vol --bocpd --nbm
... --aux-tsb 0.2`) trains end to end (14 s/epoch, 1.3 GB GPU); outputs deleted. Without the flag nothing changes
(the head is not created).
**Expectation (written before any run):** small. GRU member gains have been absorbed by the trees every time (BOCPD
channels GRU +0.004 -> blend +0.0004; NBM channels GRU +0.0037 -> blend +0.0007). Blend about +0.000 to +0.0005.
**Decision rule (written before any run):**
1. Seed 0 vs the current GRU seed 0, all 5 folds (GRU folds vary ~+-0.01, so no two-fold early stop). Paired mean <= 0
   -> CLOSED.
2. Otherwise seed 1 vs the current GRU seed 1, then the blend with the new GRU pair replacing the old pair.
3. ADOPT if the blend gains >= +0.0005, in >= 4/5 folds, and > 2 SE (section 5). If adopted, production `train()` needs
   the aux head and loss in the GRU training of `submission_template.py` (then `make_submission.py`, verify).
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one at a time; about 25 min per seed):**
1. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --aux-tsb 0.2 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm.npz --tag aug2_vol_bo_nbm_auxtsb`
2. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --aux-tsb 0.2 --seed 1 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --tag aug2_vol_bo_nbm_auxtsb_s1`
3. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_auxtsb.npz cv_out/oof_gru_aug2_vol_bo_nbm_auxtsb_s1.npz --c cv_out/oof_svm_nbm.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --ref-c cv_out/oof_svm_nbm.npz`

---

## 13. In-fold feature pruning `cv.py --prune-keep 150` (2026-10-17; RESULT: CLOSED by the early stop)

**RESULT seed 0 (owner's run 2026-10-17, stopped after fold 1 as the rule says):** fold 0 0.6579 vs 0.6621
(**-0.0042**), fold 1 0.6309 vs 0.6329 (**-0.0020**): both <= 0 -> **CLOSED**; no other K on these folds (rule). Pass 1
reproduced the baseline exactly (0.6621, 0.6329: determinism OK, so the comparison is clean). At round 200 (production
proxy): -0.0049 / -0.0006. The top 150 held 98.6-98.7% of the first-200-tree gain, yet the refit lost: the low-gain
columns still carry information the trees use (the gain share understates it), and the refit's best round moved to 150.
Lesson: at min-leaf 2000 / colsample 0.5 the trees are not hurt by the extra columns; the old 270 -> 388 harm does not
transfer.

**Why:** the one measured harm left on the list: 270 -> 388 redundant tail/window columns cost -0.005 [§7]. Today
95% of the trees' gain sits in the top 132 of 321 columns, 99% in the top 209; 148 columns carry < 0.05% each and 22
none. Pruning gives each tree's 50% column sample more of the useful columns.
**How (`--prune-keep K`, `--prune-rounds 200`):** per fold, pass 1 fits on all columns exactly as now; the columns are
ranked by the total gain of pass 1's FIRST 200 trees (training rows only; the later trees fit mostly after the
validation peak, best rounds 100-1150, so their gain leans on noise columns; 200 = production's round count; booster
slicing `m[:200]` checked equal to a separate 200-round model on xgboost 3.4.1); refit on the top K with the same
seed and settings; the OOF and the paired line come from the refit. Pass 1 is also scored on validation: (a)
determinism check, its per-fold best must equal the `--compare` file's (else use pass 1 as the reference), (b) the
paired difference at round 200 (production proxy; reported, not the gate). Writes `cv_out/prune_keep<tag>.txt`
(kept columns per fold); the importance file stays in the 321-name space (pruned columns 0). Cost about 2x the
trees: ~2 min per fold.
**Checks 2026-10-17:** flag off bit-identical (600-series GPU smoke before/after the edit: importance byte-identical,
oof/y/step/sid equal); flag on runs end to end, determinism check 0.00e+00, 150 nonzero importance columns; smoke
outputs deleted; `cv.py` still CRLF.
**Expectation (written before any run):** +0.000 to +0.001. The -0.005 came from LightGBM with min-leaf 200, no
re-cuts and a weaker base; the colsample sweep (ff 0.3 / 0.8), which changes the same thing, was flat [§44.3].
**Decision rule (written before any run; K = 150 fixed in advance):**
1. Seed 0 paired vs the current seed-0 trees. **Early stop:** folds 0 and 1 both <= 0 -> CLOSED. Do not then try
   other K on the same folds (single-seed sweep leaders read high).
2. 5-fold paired mean > 0: seed 1 (its own pass 1 and selection), the seed-average comparison, then the trees-only
   blend (SVM top-100 and GRU top-20 lists unchanged).
3. ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE (section 5).
**If adopted, production:** a 200-round pass 1 on all 10,000 series + re-cuts to pick the K columns, baked into the
tree column list in `make_submission.py`; inference cost unchanged (every stream is still needed by the SVM/GRU).
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU; about 12-15 min per seed):**
1. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --prune-keep 150 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_k150`
2. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --prune-keep 150 --seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag aug2_df_bo_nbm_bs24_ep25_k150_s1`
3. `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_k150.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_k150_s1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
4. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_k150.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_k150_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --c cv_out/oof_svm_nbm.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --ref-c cv_out/oof_svm_nbm.npz`

---

## 14. Self-replay calibration of the final blend score (owner's idea 2026-10-17; RESULT: CLOSED)

**RESULT (owner's run 2026-10-17, 515 s scoring + 10 s cross-fitting; `cv_out/replay_blend_f0.txt`, scores cached in
`cv_out/replay_blend_scores_f0.npz`):** cross-fitted held-out gains -0.0034 / +0.0000 / -0.0004 / +0.0000 / +0.0001 /
+0.0010 / +0.0003 / -0.0007 / -0.0004 / -0.0006: **mean -0.0004, 3/10 > 0 -> CLOSED** (raw was chosen on 2 of 10 tuning
halves). In-sample best (selected on the same rows): mean shift, pseudo-history >= 1000, carry, kap 10, lam 0.5:
0.6642 vs raw 0.6637, +0.0005 with a series-bootstrap sd of 0.0011, i.e. noise.
Mechanism (worth keeping):
- The replays DO carry per-series information: on no-break series the real blend logit correlates with the series'
  replay deviation at Spearman 0.16-0.27 (higher with carry-forward). But that is r^2 of only ~3-7% of the no-break
  spread, while the correction also moves break series: too little to gain, and the noisy part costs.
- Short pseudo-histories are biased, as §18 found: replays with 500-999 history points score +0.47 / +0.60 no-break sd
  above the real no-break level at steps 300-600 / 600+; with >= 1000 points the offset is small (-0.05 to +0.17).
  Every config the selection liked used the >= 1000 filter.
- Every config gained at steps 0-100 (+0.0016 to +0.0040 in-sample, about 5 replays per row there) and lost at
  100-600. Seen after the run, so not a lead by itself; steps < 100 carry ~12% of the metric weight.

**Idea:** replay the end of each series' break-free history as a fake online segment through the SHIPPED blend, and
correct the real blend logit by how this series scores with no break ("is this alarm unusually high for this series
at this online age?"). Distinct from §18 (that calibrated single features, not the blend's own false alarms). A
test-time correction: no retraining.
**My expectation (said before building): low, ~10-15% chance to pass.** The §18 failure reasons carry over: nested
replays all end at the history's end, so 1-2 replays at steps 300-700; shorter pseudo-histories bias the
history-referenced inputs (ECDF whitening, dist tests, chi2, BOCPD/NBM whitening); `--vol` already holds the
predictable no-break drift.
**Script:** `diagnostics/replay_blend.py` (design and rule in its docstring). Uses the production fold-0 model set
`verify_models/fold0` (config `blend-v3-trees2`, trained on fold 0's 8,000 training series), scores fold 0's 2,000
validation series and their replays with the production batch path (= infer(), parity-checked). Replays: L in
50/100/200/400/700/999 with pseudo-history >= 500, re-z-scored like the real histories (ddof 0), truncated to the
real online length (causal, so nothing changes). Deviation = replay logit minus the pooled mean of all replays with the
SAME L at the same row (each L has its own pseudo-history shortfall). Grid of 27: raw; mean shift
s - lam * n/(n + kap) * dev (lam 0.5/1, kap 1/3/10, aligned or carry-forward, pseudo-history >= 500 or >= 1000); the
owner's z-score (heavy sd shrinkage, a 0.5/1). Cross-fitted: 5 random stratified halves, pick on one half (raw is a
candidate), score on the other, swap: 10 held-out gains. Mechanism printed next to it: no-break Spearman correlation
of the real logit with the replay deviation per bucket, the replay-vs-real null offset, the scales.
**Checks 2026-10-17:** smoke 20 / 60 series end to end (15 s / 21 s: about 0.15 s per series after the numba compile);
smoke files deleted. Numbers at that size are meaningless.
**DECISION RULE (written before any run):** GO only if the mean of the 10 held-out gains >= +0.001 AND >= 8/10 > 0;
otherwise CLOSED. **GO is not adoption:** it means production model sets for folds 1-4 (`verify_submission.py --fold k`,
about 25-35 min each, ~2 h GPU) and this test on each fold; then production (replays scored at setup in `infer()`,
about 3-4x the online points of a series: a ~11 min cloud run would become roughly 35-45 min, inside the budget).
**Command (CPU only, my estimate from the smoke timing: ~5-6 min scoring + ~2-3 min analysis):**
`python diagnostics/replay_blend.py` -> `cv_out/replay_blend_f0.txt` (scores cached in `cv_out/replay_blend_scores_f0.npz`;
`--analyse-only` re-runs the analysis from the cache).

---

## 15. NBM data learning curve: is the normal-behaviour model data-limited? (2026-10-17; RESULT: CLOSED)

**RESULT (owner's run 2026-10-17, 9 pretrainings + gates in ~23 min after 73 s of preparation;
`diagnostics/nbm_datacurve.csv`):** nb_logodds 3-seed means 100% **0.6301** +- 0.0019 (sd), 50% **0.6333** +- 0.0023,
25% **0.6322** +- 0.0028; **100% minus 50% -0.0032 +- 0.0017 (SE)**, 100% minus 25% -0.0022 +- 0.0020. The bar (>= +0.006
and > 2 SE) is missed in the wrong direction -> **CLOSED**. Everything else is flat too: valid NLL 1.3849 / 1.3848 /
1.3852, nb_lvr (top tree feature) 0.5784 / 0.5785 / 0.5789, nu 12.7 / 12.7 / 13.2. The 100% arm is not ahead in any step
range (0-100 0.5713 vs 0.5740, 100-300 0.6445 vs 0.6458, 300-700 0.6344 vs 0.6383, 700+ 0.6017 vs 0.6168) or on the
variance-matched slice (0.5089 vs 0.5185).
Reading: a quarter of one fold's histories (~5.6M points) already trains the ~30k-parameter NBM to its limit; it is at
its information limit, not data-limited. The 100% arm's lower mean (-1.9 SE) is not read as harm from more data (under
2 SE, NLL flat, no mechanism). Consequence: NBM pretraining on the 2025 break-free stretches and the break-free online
parts is not built. With the width/design sweeps also flat [§29.4, §37], NBM pretraining (data, width, design) is exhausted; using
the NBM differently (hidden state or m / log s as GRU inputs, §8 open list) is a separate question.

**Why (the one untried source of NEW information found on 2026-10-17):** pretrain the NBM on more break-free data: the
2025 edition's break-free stretches (period 0, and whole series for its ~71% no-break series: roughly doubles the
pretraining data) and the break-free online parts of each fold's training series (no-break online segments,
pre-break stretches: about +10%). Unsupervised, so the labelled-2025 failure ("2025-looking series break less often",
§15.4 of the old file) cannot recur; per-series ECDF whitening and the history-descriptor inputs absorb part of the
population shift (2025: heavier tails, less autocorrelation). The NBM sweeps plateaued in width and design, which fits
"data-limited" or "at its information limit" equally: this curve tells the two apart before anything is built.
**Script:** `diagnostics/nbm_datacurve.py` (design and rule in its docstring; reuses `nbm_sweep.gpu_sums`). Production
NBM settings (bs24_ep25). Pool = histories of series 0-4499 + 7000-9999 (7,500, about one CV fold's pretraining set,
so the slope is measured at the operating point); validation 4500-4999; arms 100% / 50% / 25% of the pool (whole series,
a different subset per seed) x seeds 0 / 1 / 2; **matched updates** (epochs = 25 / fraction: 7,825 / 7,850 / 7,900
steps); gate = the sweep's nb_logodds single-feature TS-AUC on 5000-6999 (prior from 0-4999's breaks), plus nb_lvr,
valid NLL and the learned nu. Output `diagnostics/nbm_datacurve.csv` (resumable). Seed noise in the sweep: 0.004-0.006
between seeds of one setting.
**Checks 2026-10-17:** `--smoke` (300 histories, 2-8 epochs, 200 gate series) end to end in 36 s, nothing written.
**DECISION RULE (written before any run):** DATA-LIMITED only if the 100% arm's 3-seed mean nb_logodds beats the 50%
arm's by >= 0.006 AND > 2 SE; else CLOSED. DATA-LIMITED is not adoption: next would be NBM pretraining in `cv.py` on
fold histories + 2025 break-free stretches + the fold's break-free online parts (report nu with and without the 2025
data: 58% of 2025 series are heavy-tailed vs 15%), then the paired trees run with the early stop and the blend.
Inference cost unchanged; production pretraining about 2x longer.
**Expectation:** more data improves the NBM family (worth +0.0032 in the blend), it does not add a new one: even if
data-limited, a realistic blend gain is +0.000 to +0.001.
**Command (GPU; my estimate ~40 min for the 9 pretrainings + gates):** `python diagnostics/nbm_datacurve.py`

---

## 16. EDA of the variance-unchanged breaks (owner's question 2026-10-17; RESULT: confirms, nothing new to build)

**Script:** `diagnostics/eda_varmatched.py` -> `eda_out/varmatched/report.html` (+ `stats_L*.csv`, `series_L*.csv`,
`summary.txt`). Oracle (break position known): post = exactly L (100 / 300) points after the cut vs the 1,000 points
before; no-break series get a placebo cut; slice = |log var ratio of x| < 0.1 (885 / 1,424 at L = 100, 569 / 1,546 at
L = 300); 58 statistics, magnitude AUC, permutation chance band (0.5 +- 0.041 / 0.045).
**Findings (owner's full run 2026-10-17):**
- On the slice only dependence-type statistics clear chance, all weakly: innovation variance var_e 0.567 / 0.577
  (marginal variance fixed, so the AR fit got worse), lag-1/2 autocorrelation of x and e 0.55-0.56, roughness 0.56
  (L = 300). Shape, tails, quantiles, spectrum, nonlinear, number format, distribution distances: nothing.
- Direction: breaks pull autocorrelation toward zero. Slope of the change on the pre level -0.08 to -0.15 for breaks
  vs -0.02 to -0.09 for placebos; on slice series with |pre ac1| > 0.3 mean change of |ac1| -0.040 / -0.030 vs -0.012 /
  -0.010 and log innovation variance +0.09 / +0.11 vs +0.01. Post parameters stay close to the old ones (slope of post
  on pre ~0.9), so not a fresh draw. The models already carry this: e10 residual-square scans, e10prod/lag products,
  history rho1 context; bocpd2's AR(1)-change stream and its per-|rho1| priors gave nothing (section 8).
- Omnibus: only 8.5% / 7.9% of slice breaks exceed the placebos' 95th percentile of max robust z (5% = chance), AUC
  0.565. So ~3% of variance-unchanged breaks leave a trace; the rest (roughly a quarter of all breaks) look
  undetectable even with the break position known. Consistent with section 3 / [§20]: a ceiling, not a missing feature.

---

## 17. TCN as a fourth blend member (owner's request 2026-10-18; RESULT: CLOSED 2026-10-20 by the honest line)

**Why:** the one open "new model family" item; both public write-ups (bamasa, szukiyu40) used a TCN. Honest prior:
lesson 2 of section 9 says new algorithms on the same information have not moved the score, and this is exactly that
(same 53 inputs as the GRU). The hope is only that a convolutional core makes different errors than the recurrent one.
**What (`gru_cv.py --arch tcn`, everything else identical to the GRU: inputs, burn-in, folds, row weights, epochs 25,
patience 4, lr, heads):** Linear(53 -> 64) + GELU -> 9 residual blocks of two causal dilated Conv1d (kernel 3,
dilations 1, 2, ... 256, GELU, dropout 0.1, left padding only, no batch norm) -> concat with the causal running mean of
the block output -> Linear(128 -> 64) + GELU -> the GRU's MLP head. Receptive field 2,045 steps (> any sequence, max
1,099), so it sees the whole burn-in + online part like the GRU; the running mean is the "accumulated evidence" part.
~236k parameters (GRU ~30k). One configuration fixed in advance: NO sweep of it on these folds.
Outputs `cv_out/tcn_summary<tag>.txt`, `cv_out/oof_tcn<tag>.npz`; each fold now also prints its best epoch (needed for
a fixed production epoch count).
**Also new:**
- `blend.py --honest-paired` (with `--ref-*`): paired per-fold comparison with CROSS-FITTED weights for both blends.
  Needed because the in-sample paired line is biased for an extra member: the 4-member grid contains every 3-member
  blend (d = 0), so a member with no information still shows >= 0. Costs ~8 min for 4 vs 3 members.
- `diagnostics/member_corr.py`: metric-weighted per-step Spearman between two members. References: GRU seed 0 vs
  seed 1 **0.81** (no-break series 0.78); GRU pair vs tree pair **0.69** (0.64). A TCN near 0.8 vs the GRU pair is
  "another GRU seed" (worth ~+0.0002, section 8).
**Checks 2026-10-18:** causality (eval mode, altering x[:, t0:] leaves outputs < t0 exactly unchanged, t0 = 0 ... 1199;
a truncated series equals the prefix of the full one: 0.0); 300-series smoke with the production flags runs end to end
for `--arch tcn` (1.3 GB GPU) and for the default GRU (still writes `oof_gru...`); `--arch gru` creates the same
parameters in the same order as before (the mem-bias loop only skips for tcn), so the GRU path is unchanged by
construction; `blend.py --honest-paired` on two identical blends gives +0.0000 in every fold. Smoke outputs deleted;
`gru_cv.py` and `blend.py` still CRLF.
**Expectation (written before any run):** TCN alone about the GRU's level (0.62-0.635 per seed); blend +0.000 to
+0.001; my chance of passing the bar ~15%.
**DECISION RULE (written before any run):**
1. TCN seed 0, 5 folds (`--compare` the GRU seed 0 only for information: a different member, not the test). Then
   `member_corr.py` vs the GRU pair, and the 4-member blend with the single TCN seed using `--honest-paired`.
   **Early stop:** honest paired mean <= 0 -> CLOSED.
2. Otherwise TCN seed 1, then the 4-member blend with the TCN pair. **ADOPT if the HONEST paired gain >= +0.0005, in
   >= 4/5 folds, and > 2 SE** (the standing bar of section 5, applied to the cross-fitted numbers). The in-sample
   line is printed too but is not the test.
3. Report only, not a selection: the 3-member blend with the TCN pair REPLACING the GRU pair (choosing whichever
   variant looks better on the same 5 folds would be selection).
**If adopted, production:** a numba streaming TCN in `submission_template.py` (per level a ring buffer of (k-1)*d past
block inputs for each conv, plus the running sum), streaming == batch parity check; fixed epochs = the median best
epoch of the CV folds; trained on the GPU like the GRU (2 seeds). My estimate (unmeasured): ~8x the GRU's arithmetic per
point, cloud inference ~11 min -> roughly 25-35 min, far inside the 15 h budget (accuracy first, section 5).
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one at a time; my estimate ~5-15 min per seed
with the cached GRU inputs and NBM channels):**
1. `python gru_cv.py --arch tcn --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm.npz --tag aug2_vol_bo_nbm`
2. `python diagnostics/member_corr.py --a cv_out/oof_tcn_aug2_vol_bo_nbm.npz --b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz`
3. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --c cv_out/oof_svm_nbm.npz --d cv_out/oof_tcn_aug2_vol_bo_nbm.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --ref-c cv_out/oof_svm_nbm.npz --honest-paired`
4. (only if 3 is > 0) `python gru_cv.py --arch tcn --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --seed 1 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --tag aug2_vol_bo_nbm_s1`
5. Command 3 with `--d cv_out/oof_tcn_aug2_vol_bo_nbm.npz cv_out/oof_tcn_aug2_vol_bo_nbm_s1.npz`, and the report-only
   replacement: `python blend.py --a <trees s0 s1> --b cv_out/oof_tcn_aug2_vol_bo_nbm.npz cv_out/oof_tcn_aug2_vol_bo_nbm_s1.npz --c cv_out/oof_svm_nbm.npz --ref-a <trees s0 s1> --ref-b <GRU s0 s1> --ref-c cv_out/oof_svm_nbm.npz --honest-paired`

**RESULT seed 0 (owner's run 2026-10-18):** TCN alone 0.6353 mean of folds (pooled OOF 0.6282: its score scale differs
between folds; best epochs 4 / 7 / 4 / 11 / 3, a jumpy validation curve) vs GRU seed 0 0.6359: paired -0.0005 +- 0.0017,
2/5. Spearman vs the GRU pair 0.770 (no-break 0.731), a bit below GRU seed vs seed (0.81). 4-member blend IN-SAMPLE
0.6492 vs 0.6489, +0.0004 +- 0.0002, 5/5, TCN weight 0.1 (biased upward by construction). The honest line was not in
the paste: pending. Owner wants it in; my advice: honest line first, then seed 1, then decide (production needs a
streaming numba TCN).
**RESULT honest line (owner's run 2026-10-20, vs the 0.6527 reference: trees pair, epoch-averaged GRU pair, hpca8 SVM):**
in-sample 0.6527 vs 0.6527 (+0.0001 +- 0.0002, 2/5, TCN weight 0.1); **HONEST cross-fitted 0.6526 vs 0.6529,
-0.0002 +- 0.0001, 0/5** (per fold -0.0007 / -0.0002 / -0.0002 / 0.0000 / 0.0000; folds 3-4 give the TCN weight 0).
**CLOSED by the early stop of step 1**, even though the TCN file is favoured (best epoch picked on the validation fold).
No seed 1, no streaming TCN. Consistent with lesson 2 (section 9): a new algorithm on the same inputs.

---

## 18. Search over ALL existing OOF files for the best blend, from scratch (owner's idea 2026-10-18; RESULT: CLOSED)

**Idea:** do not assume the current blend is optimal. Start from the single models, and let blends of any files
(including members rejected on their own: tree variants, old GRUs, the TCN, SVM variants, closed feature families)
compete; whatever beats the peak becomes the new peak. **Danger:** choosing from 74 files on the same 5 folds finds
luck, so the test is NESTED.
**Script:** `diagnostics/blend_search.py` (design and rule in its docstring) -> `diagnostics/blend_search.txt`.
- Candidates: every `cv_out/oof_*.npz` with the full layout and no NaN (74 files on 2026-10-18; 48 smoke/probe/one-fold
  files are skipped). Each file is scaled by its sd; seed files are separate candidates.
- **From scratch (default), growing one file at a time:** size 1 = every single file (the best is the first peak; a
  single can stay the winner); size 2 = EVERY pair of the 74 files (2,701 pairs x weights 0.1-0.9); size 3, 4, ... =
  each of the 10 best blends so far + every file at weights 0.05-0.5 (s' = (1 - w) s + w c), keeping the 10 best
  distinct blends (`--beam` = blends carried forward, NOT a minimum size). Whatever beats the peak becomes the peak;
  stop when a step raises it by < +0.0002 or after 10 additions. A file can be picked again (raises its weight).
  `--start current` starts from the shipped blend instead.
- Nested: search on 4 folds, score on the 5th, next to the CURRENT blend and the best single file -> 5 honest gains.
  Then the same search on all rows gives the blend to ship (optimistic), its weights per file, the top 40 pairs and
  the best partner of every file.
- Metric: batched GPU TS-AUC by default (`--device cuda`/`auto`; sort each blend, stable-sort by step, rank = position in
  the step; ~0.8 ms per blend, 48 blends per call, checked at start-up against the CPU metric: 4e-9), or the numba CPU
  metric (`--device cpu`, ~17 ms per call; equal to `sbrt.metric.ts_auc` to 1e-9). The smoke run gives identical picks
  and scores on both (GPU 14 s, CPU 56 s).
**Check 2026-10-18:** `--smoke` (8 candidates, 1 outer fold) in 64 s. Smoke reading, not a result: on fold 0's other
folds best single pc-trees 0.6393 -> best pair pc-trees 0.6 + GRU 0.4 0.6436 -> 3 files 0.6447 -> 5 files (pc-trees 0.33 +
TCN 0.22 + trees 0.17 + GRU 0.15 + SVM 0.14) 0.6456; on held-out fold 0 that scored 0.6645 vs the current blend 0.6662
(-0.0017; the current blend's weights were picked on all rows, a small leak in its favour).
**DECISION RULE (written before any run):** the searched blend goes to production only if its NESTED gain vs the current
blend >= +0.0005, >= 4/5 folds, > 2 SE; and the blends found per outer fold should be similar (different picks per fold
= luck).
**Production cost if it passes:** depends on the picks. Tree variants of the current features (LR015, L63, FF08, ML8000,
seeds) = extra XGBoost models, cheap. Picks that use closed feature streams NOT in production (pc, femb, eqf, ms,
bo2, mmd, ...) need those streams ported into `submission_template.py`: real work and inference time. Old-feature-set
members (aug2_evc, gru_aug2_vol, ...) use subsets of today's streams: only extra models.
**Command (GPU; my estimate ~5-10 min, ~1.5 h with `--device cpu`):** `python diagnostics/blend_search.py`

**RESULT (owner's run 2026-10-18, GPU, 275 s; `diagnostics/blend_search.txt`):** NESTED gain vs the current blend per
held-out fold **-0.0009 / -0.0006 / -0.0033 / -0.0007 / -0.0021: mean -0.0015 +- 0.0005, 0/5 -> FAIL, CLOSED.** The
current blend's all-rows weights are not the reason: with cross-fitted weights (chosen on 4 folds; my check,
`scratchpad` script, GPU) the current structure scores 0.6493 vs 0.6494 fixed (leak +0.0001), and the searched blend
still loses -0.0014 +- 0.0005, 0/5.
- What the search found: best single (a tree variant) -> best pair (+0.004, always trees + a GRU) -> 5-8 files
  (+0.002 more in-sample). Held out it beats the best single by +0.0052, but the current blend beats the best single
  by more (+0.0067): grouping seeds into averaged members and fixing three families generalises better than picking
  individual files. The search keeps choosing single seed files and different files per fold (pc / FF08 / LR015 /
  femb / s1 trees, GRU s0 or s1, synth trees): it fits fold noise.
- `gru_cap_gate_static_s1` (a 10-13 GRU, old features, static gate) was picked in all 5 outer folds and in the
  all-rows blend (best pair with FF08 trees, 0.6471). Not independent evidence (outer folds share 3/4 of their
  rows), its seed 0 twin was never picked, and the held-out blends containing it lost: not a lead.
- In-sample (optimistic) all-rows blend: 0.6502 vs 0.6489 (pc trees 0.35, GRU s0 0.23, FF08 trees 0.18,
  gru_cap_gate_static_s1 0.14, SVM 0.10). It is what overfitting looks like: +0.0013 in-sample, -0.0015 held out.
- Consequence: the current 3-member blend stays. Rejected members do not rescue each other; blending more files
  is closed (also covers the TCN as a member: the search had it and never kept it held-out).

---

## 19. Covariate-shift weighting of the 2025-edition data (owner's go 2026-10-18; RESULT: CLOSED by the early stop)

**Why:** we are data-limited (doubling the data ~ +0.015 for the trees, section 3), and the 2025 edition is the only
extra labelled data. It failed as-is (trees -0.0057) and source-balanced (-0.0006) because it is a different
population (section 8). Importance weighting is the standard fix for a shifted source and is NEW here: weight each
2025 series by p_RealTime(x) / p_2025(x) of its normal behaviour x, so the weighted 2025 set looks like ours.
**Script:** `diagnostics/ext_weights.py` (design, rule and expectation in its docstring) ->
`data/ext2025/ext_weights_e1_s0_h1000.npz` + `diagnostics/ext_weights.txt`.
- x = 14 shape statistics of the LAST 1,000 HISTORY points (sd, skew, kurtosis, tails > 2/3/4, ac 1/2/5, ac of z^2 at
  1/5/20, sd of 10 block log variances, AR(5) residual variance ratio). History only, so the weights cannot see the
  label; a fixed window, because 2025 histories are capped at ~2,400 points (period-0 length) while ours run
  1,000-5,000: length is our cut geometry and would dominate the classifier.
- Cross-fitted (5-fold) shallow HistGradientBoosting tells the 9,978 2025 cuts (`--ext 1`, seed 0, exactly cv.py's
  list) from the 10,000 Real-Time originals; w = (n25 / n26) p / (1 - p), clipped at 10, mean 1. Prints the domain AUC,
  the effective sample size ESS = (sum w)^2 / sum w^2, the weighted 2025 break rate and a balance table (does the
  weighting close the gap in each statistic?).
- `cv.py --ext-weight <npz>`: multiplies every row of a 2025 series by its weight, on top of `--ext-rel` and
  `--ext-balance` (which still equalises the 2025 positive/negative shares per step; needed, the weighted break rate
  is even lower). Checks the file matches the cuts (vlen); `--n-series` uses the same prefix.
**Checks 2026-10-18:** `ext_weights.py --smoke` (600 + 600 series, 8 s). Smoke reading (small, not the gate): domain AUC
0.94 (very different populations), ESS 13% of the 2025 cuts (~1,300 of 9,978 if that holds at full size: near the
gate), weighted break rate 0.185 (vs 0.277 unweighted: the Real-Time-like 2025 series break even less), gap closed in
tails > 3 / > 4 (+98% / +96%), variance wandering (+96%), ac2 (+90%), but kurtosis only +36%. `cv.py`: without the flag
bit-identical to before (400-series GPU smoke with `--ext 1 --ext-balance`: OOF and importance files identical); with
a fake weight file the run uses the weights end to end. Smoke outputs deleted; `cv.py` still CRLF.
**GATE RESULT (owner's run 2026-10-18, 19 s; `diagnostics/ext_weights.txt`):** domain AUC **0.948**, ESS **1,694** of 9,978
(16.9% of our data) -> **PASS**. 41 weights clipped at 10; the top 10% of series hold 62% of the weight. Weighted break rate
0.289 (unweighted 0.291; ours 0.497: `--ext-balance` handles it). Gap closed: tails > 3 / > 4 +93% / +97%, kurtosis +83%,
variance wandering +96%, ac1 +95%, ac2_20 +92%; partly: ac2 +57%, ac2_5 +55%, skew +57%, sd +49%, ac5 +43%,
AR(5) residual variance only +21%; OVERSHOT: ac of z^2 at lag 1 (0.1158 ours, 0.1261 -> 0.0867 weighted, -180%):
the weighted set has less short-lag volatility clustering than ours.
**TREES RESULT seed 0 (owner's run 2026-10-18, stopped after fold 1 as the rule says):** fold 0 0.6493 vs 0.6621
(**-0.0128**), fold 1 0.6275 vs 0.6329 (**-0.0054**): both <= 0 -> **CLOSED**. Best rounds fell to 100 / 150 (the trees fit
the 2025 rows early and then degrade). Twice the harm of the plain 2025 run of 10-13 (-0.0057, older features).
My reading: (1) a design flaw of mine: weights were scaled to mean 1, so the 2025 source carried the training weight of
~10,000 series (1.31M rows, as many as the originals) while holding only ~1,700 series' worth of information (ESS),
concentrated on a few hundred series with weights up to 10 (top 10% = 62% of the weight): the trees overfit those.
Scaling the total to the ESS (`--ext-rel` ~0.17) would have been the principled choice. (2) Even so the ceiling was
low: the balanced unweighted run was already -0.0006, and the 2025 data matches ours only partly after weighting.
Not re-run with a smaller scale: that would be a second arm picked after seeing this result, with a ceiling near 0.
The 2025 data is closed for the trees in every form tried (plain, balanced, weighted).
**DECISION RULE (written before any run):**
0. GATE (`ext_weights.py`, no cv.py): domain AUC >= 0.60 AND ESS >= 1,000 series; otherwise CLOSED.
1. Trees seed 0 (the current best flags + `--ext 1 --ext-balance --ext-weight ...`) vs the current seed-0 trees.
   Early stop: folds 0 and 1 both <= 0 -> CLOSED.
2. 5-fold paired mean > 0: seed 1, `diagnostics/oof_paired.py` seed-average vs seed-average, then the trees-only blend.
3. ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE (section 5). Then the GRU / SVM with the same data.
**Expectation (written before any run):** ESS ~1,300-2,000 is +13-20% data; if it were as good as ours that is
trees +0.002-0.004 (learning curve), shrinking ~2x into the blend: realistic blend +0.000 to +0.002.
**Cost:** inference none (the cloud never trains). The first trees run builds the 2025 features for today's families
(vol, dist, BOCPD, NBM per fold; kernel and evidence caches exist): my estimate ~30-45 min extra, once.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first):**
1. Gate (CPU, my estimate ~1 min): `python diagnostics/ext_weights.py`
2. Only if the gate passes (GPU; ~45-60 min the first time): `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --ext 1 --ext-balance --ext-weight data/ext2025/ext_weights_e1_s0_h1000.npz --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_extw`
3. Only if 5-fold paired > 0: the same with `--seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag aug2_df_bo_nbm_bs24_ep25_extw_s1`, then
   `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_extw.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_extw_s1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
   and the blend (section 6 command with the new tree pair as `--a`).

---

## 20. Break shape, location persistence, detector agreement, symbol transitions `cv.py --shape` (owner's request 2026-10-18; RESULT: CLOSED)

**Why:** two facts from our own analysis: breaks are INSTANT switches (section 3), and the main error is break-free
series whose variance WANDERS smoothly. No detector here has a smooth-drift alternative, none looks at how its
break-location estimate behaves over time, and the location estimates of different detectors are never compared.
The owner added "symbols repeat in a stable signal" (communications analogy).
**Module:** `sbrt/shape.py` (design, rule and expectation in its docstring), 12 columns from the bocpd.py whitened
streams w / wu (causal, ~N(0, 1) before a break):
- A step vs drift: `sh_step_ll` (best variance step), `sh_ramp_ll` (best smooth log-variance ramp from the history
  level, 41 drift rates), `sh_step_vs_ramp` (jump-like vs drift-like).
- B persistence of the step location L*: `sh_lstar_age`, `sh_lock_frac` (time since L* last jumped), `sh_lsd50` (sd of
  L* over 50 steps).
- C agreement: |L*(w) - L*(wu)|, |L*(w) - L*(dependence w_i w_(i-1))|, |L*(w) - L*(tail rate |w| > 1.96)|, and the
  direction agreement of w and wu.
- D symbols: quartile symbols of w; online transitions vs the history's 4 x 4 transition table, conditional on the
  from-symbol (a dependence test, not frequencies): chi-square over all transitions and the last 100.
  Prior art: decile chi-square (in production), ordinal patterns, matrix profile / motifs, MMD pairs/triples (all
  section 8): D is the one untried variant and the most likely to be redundant.
- Not built: peakedness of the change profile (= the location-posterior entropy of the closed `--pshape`).
**Checks 2026-10-18:** self-test `python -m sbrt.shape`: causality on 20 series (prefix rows equal; an altered future
changes nothing before it); 2,000 held-out series in 38 s (~12 ms per series). Single-feature TS-AUC (series
5000-6999; NOT the test): step_ll 0.601, ramp_ll 0.594, step_vs_ramp 0.571, lstar_age 0.547, lock_frac 0.574, lsd50 0.443
(= 0.557 reversed), agree_wu 0.482, agree_dep 0.497, agree_tail 0.476, dir_agree 0.509, sym_chi 0.546, sym_chi100 0.541.
Reading: A and B carry signal alone; C is weak; D is about as strong as other dependence statistics. `cv.py`: flag
off bit-identical (400-series GPU smoke with --aug 2 --bocpd: OOF and importance files identical); with --shape 12
columns appended for originals and re-cuts, 2.3% of the tree gain in the smoke. Smoke outputs deleted; `cv.py` CRLF.
**DECISION RULE (written before any run):** the standing rule (section 5): trees seed 0 paired vs the current seed-0
trees, early stop if folds 0 and 1 are both <= 0; then seed 1, the seed average, the trees-only blend; ADOPT at
>= +0.0005, >= 4/5 folds, > 2 SE. Reported, not a gate: the gain share of each sub-family (A/B/C/D) in the importance
file. If adopted: a streaming version (the batch kernel is causal; no stream class yet; inference ~O(t) per point,
like BOCPD) and the GRU/SVM inputs.
**Expectation (written before any run):** ~20-25% chance to pass; blend +0.000 to +0.002.
**RESULT seed 0 (owner's run 2026-10-19, all 5 folds; `cv_out/summary_aug2_df_bo_nbm_bs24_ep25_sh.txt`):** folds
**-0.0031 / -0.0022** / +0.0020 / +0.0045 / +0.0002: folds 0 and 1 both <= 0 -> **CLOSED by the early stop**; the full
5-fold paired line is +0.0003 +- 0.0014, 3/5 (inside the seed-noise floor of +-0.0009 per paired mean, section 5),
headline 0.6414 at 150 rounds. Gain shares of the 12 columns: A step/ramp 0.87% (ramp_ll 0.55, step_vs_ramp 0.18,
step_ll 0.13), B persistence 0.42% (lstar_age 0.29, lock_frac 0.11, lsd50 0.02), C agreement 0.20%, D symbol
transitions 0.18%; best column rank 31 of 333. Reading: the trees barely use them. The single-feature signal of A and B
(0.57-0.60) is the variance-step evidence the kernel scans, BOCPD and NBM already carry; the step-vs-drift contrast
and the location stability add nothing the trees cannot already infer. D confirms section 8: symbol dependence is
covered. Not run: seed 1, blend.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU; my estimate ~6 min to build the shape caches
once + ~12-15 min per seed):**
1. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --shape --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_sh`
2. Only if the 5-fold paired mean > 0: the same with `--seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag aug2_df_bo_nbm_bs24_ep25_sh_s1`, then
   `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_sh.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_sh_s1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
   and the blend (section 6 command with the new tree pair as `--a`).

---

## 21. Single-series Rio-CPD `cv.py --rio` (owner's request 2026-10-19 after reading Deng et al., CIKM 2025; RESULT: CLOSED by the early stop)

**Paper review 2026-10-19 (owner's 8 PDFs):** Adams & MacKay 2007 (BOCPD) = our `--bocpd` (one-break version);
Habibi 2021 (prior log odds + log LR) = our `bo_*_logodds` / `logbf`; Wei & Xie 2026 JRSSB (online kernel CUSUM) and
Li et al. 2015 (M-statistic) = the closed `--mmd` family; Yu et al. 2023 (minimax CUSUM, mean change) = theory for mean
shifts our data lack; Zhang et al. 2019 (auxiliary series) = no partner series at inference; Sullivan 2018 is literary
history (Yeats's metre), unrelated. Only Rio-CPD's SPD geometry was not in the codebase.
**Module:** `sbrt/riocpd.py` (design, rule and expectation in its docstring). The K x K (K = 5) lag-covariance Toeplitz
matrix of the whitened residual w replaces Rio-CPD's multivariate correlation matrix; the history's matrix is the
reference; affine-invariant geodesic distance ||log(A^-1/2 B A^-1/2)||_F. 10 columns: covariance distance (all online /
last 100 / last 30), correlation-only distance (all / last 100), covariance distance of wu (last 100), the last-30
distance z-scored against the history's own sliding 30-point windows (step 5), CUSUMs of that z (covariance and
correlation, k = 0.5) and the time since the covariance CUSUM's last reset. Lag products reach into the last 4 history
points. Not new: the information (second-order structure: variogram lags, lag products, bocpd2, --mmd, --shape).
**Checks 2026-10-19:** self-test `python -m sbrt.riocpd`: geodesic identity / scaling (sqrt(K) |log c|) / symmetry;
causality on 20 series (prefix rows equal, an altered future changes nothing before it); 2,000 held-out series in 35 s.
Single-feature TS-AUC (series 5000-6999; not the test): cov_full 0.592, cov_100 0.577, cov_30 0.545, cor_full 0.548,
cor_100 0.532, cov_100_u 0.583, z30_cov 0.543, cusum_cov 0.562, cusum_cor 0.535, age_cov 0.559. `cv.py`: flag off
bit-identical (400-series GPU smoke with --aug 2 --bocpd: OOF and importance identical); with --rio 10 columns for
originals and re-cuts, 2.5% of the gain in the smoke. Smoke outputs deleted; `cv.py` CRLF.
**DECISION RULE (written before any run):** the standing rule (section 5): trees seed 0 paired vs the current seed-0
trees, early stop if folds 0 and 1 are both <= 0; then seed 1, the seed average, the trees-only blend; ADOPT at
>= +0.0005, >= 4/5 folds, > 2 SE. Reported, not a gate: gain share of the 10 columns.
**Expectation (written before any run):** ~10% chance to pass; blend +0.000 to +0.001.
**RESULT seed 0 (owner's run 2026-10-19, stopped during fold 2 as the rule says):** fold 0 0.6618 vs 0.6621
(**-0.0003**), fold 1 0.6320 vs 0.6329 (**-0.0010**): both <= 0 -> **CLOSED**. Small losses, inside the seed noise:
the trees get nothing from the SPD-geometry view of second-order structure they already have (the pattern of
section 9). The `forrtl: error (200)` lines are the harmless Ctrl-C message (section 4).
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU; my estimate ~6 min to build the caches once +
~12-15 min per seed):**
1. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --rio --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_rio`
2. Only if the 5-fold paired mean > 0: the same with `--seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag aug2_df_bo_nbm_bs24_ep25_rio_s1`, then `diagnostics/oof_paired.py` and the blend (section 6).

---

## 22. Per-step cross-sectional feature normalisation `cv.py --xsec` (owner's request 2026-10-19; RESULT: CLOSED)

**Source:** Rasekhschaffe & Jones 2019 (FAJ, ML for stock selection): features are standardised across stocks on each
date because the model only has to rank stocks within a date. TS-AUC has the same structure: it ranks series WITHIN
an online step. Paper review 2026-10-19 (owner's 3 new PDFs): Leung, Ng & Yau 2017 (score CUSUM of pairwise
likelihood) = the general form of Berkes et al. 2004, its parts covered (AR change closed, variance level in --vol,
volatility-dynamics change no signal); Knox 1964 (space-time interaction of events) = exceedance clustering, covered by
the tail windows/scans and --vol; Habibi 2021 = already reviewed (section 21).
**What:** every feature column except `ctx_*` and `t_log` becomes (x - median) / (IQR / 1.349) of the fold's TRAINING
rows (originals + re-cuts) in its online-step bin (41 log-spaced bin starts, 36 unique; rows exist at every 3rd step,
so ~29 bins hold rows; empty bins borrow the nearest earlier bin; at most 50,000 rows per bin for the statistics,
seeded; IQR 0 -> sd -> 1). Validation rows use the training tables (= what production would ship: a 36 x 315 table of
medians and scales). Within a step the transform is a positive affine map, so no ranks change; what changes is how the
trees pool rows across steps (today they must combine each raw feature with `t_log` to know what "large for this step"
is, e.g. scan maxima grow with t). Not tried before: only SCORE-level per-step scaling (`blend.py`, no gain).
**Checks 2026-10-19:** flag off bit-identical (400-series GPU smoke with --aug 2 --bocpd: OOF and importance identical);
with --xsec the run completes (274 of 280 columns transformed; 482 bin-columns used the sd fallback: indicator-like
columns). The smoke score fell (0.5934 vs 0.6052), not readable at 400 series (bin tables from ~10x fewer rows).
Smoke outputs deleted; `cv.py` CRLF.
**DECISION RULE (written before any run):** the standing rule (section 5): trees seed 0 paired vs the current seed-0
trees, early stop if folds 0 and 1 are both <= 0; then seed 1, the seed average, the trees-only blend; ADOPT at
>= +0.0005, >= 4/5 folds, > 2 SE. If adopted: the tables go into production `train()` (fitted on all training rows)
and `infer()` (one lookup per feature per point), and the SVM/GRU could use the same tables.
**Expectation (written before any run):** ~15% chance to pass; a representation change, no new information.
**RESULT seed 0 (owner's run 2026-10-19, all 5 folds; `cv_out/summary_aug2_df_bo_nbm_bs24_ep25_xsec.txt`):** folds
**-0.0044 / -0.0027** / +0.0005 / +0.0062 / -0.0054: folds 0 and 1 both <= 0 -> **CLOSED by the early stop**; 5-fold
-0.0011 +- 0.0021, 2/5; headline 0.6402 at 150 rounds (vs 0.6420). Pooled OOF by step vs the baseline: < 100 0.5822 vs
0.5807, 100-300 0.6359 vs 0.6339 (better early), 300-600 0.6577 vs 0.6595, 600+ 0.6783 vs 0.6791 (worse late). 315 of 321 columns transformed, 10-12 s per fold.
The importance moved from nb_lvr (7.5% -> 3.7%) to nb_logodds (2.4% -> 6.3%) and the ctx_ columns.
Reading: the information is unchanged (t_log is still there), so this only tests the representation, and the raw
one is better for the trees. Likely reasons: the trees had already learned the step dependence through t_log; the
bin tables add steps at the bin edges and estimation noise in the sparse late bins; and a feature's ABSOLUTE level
(e.g. the NBM log variance ratio) means the same thing at every step, so re-centring it per step removes a scale the
trees were using. Large per-fold spread (sd 0.0047 of the differences): the transform changes which splits are found,
not the information.
**Command (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU; my estimate ~13-17 min, ~1 min per fold extra):**
`python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --xsec --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_xsec`

---

## 23. Re-cut bagging: tree seed 1 trained on a fresh re-cut draw `--aug-seed 1` (owner's go 2026-10-19; RESULT: CLOSED, owner stopped after fold 0)

**Source:** the future-work / limitations review of the owner's 10 papers (2026-10-19): almost everything there is
theory, high dimensions or other applications; Rasekhschaffe & Jones 2019 (FAJ) recommend combining forecasts from
models trained on DIFFERENT TRAINING SETS (windows) against overfitting. Here both tree seeds train on the same draw of
re-cut series (`--aug-seed 0`); the second seed changed only the tree randomness (blend +0.0004, section 7). Training
seed 1 on a fresh draw (19,992 new boundary re-cuts of the same 10,000 series) adds data diversity between the two
averaged members. (Leung et al. 2017's stochastic-volatility angle: not built, the NBM already learns a volatility
filter and is at its information limit, section 15.)
**No code change:** `cv.py --aug-seed 1` draws new re-cuts (`sbrt/augment.py`, seeded); every cache name carries the
stub `aug2_s1_m20`, so nothing old is overwritten. The NBM per-fold weights are pretrained on originals only and are
reused; only the NBM features of the new variants are computed.
**DECISION RULE (written before any run):** the test is the PAIR, not the single run (a single seed on a new draw is
expected to tie the old seed 1, so the usual two-fold early stop does not apply to step 1; its paired line vs the old
seed 1 is printed for information only).
1. Trees seed 1 on `--aug-seed 1` (5 folds).
2. `diagnostics/oof_paired.py`: pair (seed 0 / draw 0, seed 1 / draw 1) vs the current pair (seed 0 / draw 0, seed 1 /
   draw 0). Paired mean <= 0 -> CLOSED (no blend run).
3. Otherwise the blend with the new tree pair as member a (GRU pair and SVM unchanged; same member count, so the
   in-sample paired line is fair). ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE (section 5).
**If adopted, production:** `submission_template.py` trains the two XGBoost seeds on their own re-cut draws (a second
`make_variants` with seed 1 for tree seed 1: ~20k more series of features in `train()`, roughly +10-15 min of local
training and more RAM); SVM and GRU unchanged; inference cost unchanged.
**RESULT (owner's run 2026-10-19, stopped after fold 0; no OOF file written):** fold 0 0.6573 vs 0.6621 for the old seed 1
(**-0.0047**; the old seed 0 also scores exactly 0.6621 on fold 0, checked, so the printed baseline is right). The new
re-cut draw costs features once (re-cuts 97 s, evidence 72 s, vol 68 s, dist 92 s, BOCPD 85 s, NBM 380 s per fold;
caches `*_aug2_s1_m20*` now exist). The rule judged the PAIR over 5 folds, so strictly it was not completed; but a
member 0.0048 below both current seeds on fold 0 (~2.5 per-fold seed sd) would have to be unusually complementary to
lift the current pair (0.6640 on fold 0) by the +0.0005 bar: owner's call to close accepted. Reading: the second tree
member gains from tree randomness, not from a different draw of re-cuts; a fresh draw is not more diverse in a useful
way (both draws are re-cuts of the same 10,000 series).
**Expectation (written before any run):** small: the second seed alone was +0.0004 in the blend; a different training
draw adds some diversity on top. Blend +0.0002 to +0.0006, near the bar; ~25% chance to pass.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU; my estimate ~45-75 min the first time, mostly
building the kernel / vol / dist / BOCPD / NBM features of the 19,992 new re-cuts once; later runs ~13 min):**
1. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --aug-seed 1 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag aug2_df_bo_nbm_bs24_ep25_s1_as1`
2. `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1_as1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
3. Only if 2 is > 0: `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1_as1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --c cv_out/oof_svm_nbm.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --ref-c cv_out/oof_svm_nbm.npz`

---

## 24. NBM hidden state as GRU input `gru_cv.py --nbm-hpca 8` (owner's idea 2026-10-19; RESULT: ADOPTED, new best CV 0.6531)

**Idea (owner):** earlier layers can hold knowledge that later layers dilute. Of our three members only the GRU has
layers; the owner chose the version that carries an EARLIER LAYER OF ANOTHER MODEL forward: the pretrained NBM (a GRU
that learned normal behaviour on break-free histories) is used today only through 6 summaries of its output (nb_lvr,
the top tree feature, nb_logodds, ...); its 64-d hidden state is thrown away. Open item of section 8 ("NBM hidden
state ... as GRU inputs", old file section 29.6). (The alternative, a skip connection from the GRU's input layer to its
head, was not built.)
**What:** `sbrt/nbm.py run_hidden()` returns the NBM GRU's hidden state after each online point (it has consumed points
<= t, so causal; it is the same state NbmStream keeps at inference). Per CV fold, with that fold's pretrained NBM: PCA on
the online parts of every 10th original training series (no labels), the first 8 components, standardised, clipped +-6,
zeros in the burn-in; channels placed right after the 6 NBM channels (53 -> 61 inputs). Cached per fold
(`data/gru_nbmh8v1_f<f>_...npz`). Needs --nbm. If adopted, production: the PCA mean / components / scales per model
(64 x 8), one projection per point next to the existing NbmStream; inference cost negligible.
**Checks 2026-10-19:** causality (10 series: prefix rows equal, an altered future changes nothing before it; the
returned state equals NbmStream.h step by step); flag off bit-identical (300-series GPU smoke: OOF identical to the old
gru_cv.py); with the flag: first 8 components explain 92-94% of the hidden-state variance (PC1 54-64%), 900 series
in 9 s. Smoke scores (NOT results): 300 series 0.5340 vs 0.5867 off; 1,000 series 3 epochs 0.5820 vs 0.5759 off (so
no bug; the 300-series drop is small-sample noise). Smoke outputs deleted; gru_cv.py and nbm.py CRLF.
2026-10-19 (owner saw 0% GPU at the start): each fold first extracts the hidden states on the CPU (GPU idle until
the epochs start). `run_hidden` now runs the per-point loop in one numba function (GIL released): bit-identical to
the Python loop (30 series), causality re-checked; ~5.6 -> ~2.9 min per fold for 30,000 series with 8 threads (the
per-series history fit now dominates). Cached per fold after the first run.
**DECISION RULE (written before any run):** as the GRU auxiliary target (section 12), because GRU folds vary ~+-0.01 by
seed:
1. GRU seed 0, all 5 folds, paired vs the current GRU seed 0. Paired mean <= 0 -> CLOSED.
2. Otherwise seed 1 vs the current GRU seed 1, then the blend with the new GRU pair replacing the old pair (same member
   count, in-sample paired line is fair).
3. ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE (section 5). Then production (see above).
**RESULT seed 0 (owner's run 2026-10-19, `cv_out/gru_summary_aug2_vol_bo_nbm_hpca8.txt`):** GRU 0.6422 vs 0.6359; paired
**+0.0063 +- 0.0013 (SE), 5/5** (folds +0.0065 / +0.0072 / +0.0012 / +0.0080 / +0.0085): step 1 PASSED, by far the
largest GRU gain since the NBM channels themselves (+0.0037). By step range 0.5748 / 0.6324 / 0.6575 / 0.6798 (old GRU
seed 0: 0.5771 / 0.6263 / 0.6492 / 0.6721, section 12): the gain sits at steps >= 100, +0.006 to +0.008; < 100 slightly
lower. On the full data the first 8 components explain 80-84% of the hidden-state variance (PC1 29-33%), less than
the smoke's 92-94%. Hidden-state extraction 243-298 s per fold (old Python loop), cached now. Best epochs 4-7, the
usual overfitting after that. Peak GPU memory 9.1 GB.
Reading: the NBM's internal state carries detection information its 2-number head output (and the 6 statistics built
on it) throws away: the owner's premise holds for this model. Whether it reaches the BLEND is the open question (GRU
gains shrank ~5-10x into the blend before, because the trees hold the same information; the trees do NOT have these
hidden-state channels, so the shrinkage may be smaller this time).
**RESULT seed 1 (owner's run 2026-10-19, `cv_out/gru_summary_aug2_vol_bo_nbm_hpca8_s1.txt`):** GRU 0.6421; paired vs the
old seed 1 **+0.0075 +- 0.0013, 5/5** (+0.0101 / +0.0089 / +0.0046 / +0.0096 / +0.0043). Steps 0.5791 / 0.6334 / 0.6576 /
0.6802. With the hidden states cached a seed takes ~2 min.
**RESULT blend (owner's run 2026-10-19):** members trees pair 0.6443, NEW GRU pair 0.6451 (old pair 0.6354), SVM 0.6379.
Blend **0.6531** at trees 0.4 / GRU 0.5 / SVM 0.1 vs the reference **0.6489** (0.5 / 0.3 / 0.2): paired **+0.0043 +- 0.0009,
5/5** (+0.0059 / +0.0046 / +0.0013 / +0.0035 / +0.0061). The rule (>= +0.0005, >= 4/5, > 2 SE) passes by ~8x and ~4.8 SE
-> **ADOPTED**. Flat weight plateau (0.6526-0.6531 over six weight sets), per-step scaling identical. The usual
shrinkage into the blend was small this time (GRU pair ~+0.0097 -> blend +0.0043, ~2x instead of 5-10x): the trees do
not hold this information. In-sample weights: their optimism was ~0.0001 for the old 3-member blend (section 18
check); not re-checked for this one (`blend.py --honest-paired` would).
**Next steps (in order):**
1. Production: (a) `submission_template.py`: in `train()`, after the NBM, the PCA of the hidden states (online parts
   of every 10th training series; mean 64, components 8 x 64, scales 8) saved with the GRU; the GRU training inputs
   get the 8 channels after the NBM channels; GRU epochs: check the production epoch count (6) against the best
   epochs here (4-8). (b) `infer()`: per point, after the NbmStream push, project `h` (64 x 8 multiply, negligible).
   (c) Blend weights 0.4 / 0.5 / 0.1. (d) `make_submission.py`, `verify_submission.py --n-train 600 --n-val 200
   --parity 5`, then the full verify (fold 0 production vs research), `crunch test`, submit.
2. Give the same 8 hidden-state columns to the TREES (cv.py) and the SVM: they are the other 50% of the blend and see
   none of it (the trees' gain should shrink more than the GRU's, but they are the biggest member).
3. Variants of the idea: K = 16 components (8 explain only 80-84% of the variance on the full data), the full 64-d
   state, or the NBM's pre-GRU input layer.
**Production v4 (`blend-v4-hpca8`, built 2026-10-19):** `submission_template.py`: NBM_HPCA = 8; in `train()` after the
NBM (step 0b): `NBM.run_hidden` on every 10th original series' online part -> PCA (SVD) -> h_mean (64), h_comp (8 x
64), h_sd (8), saved in `gru.npz`; `_nbm_hp()` gives every series' 8 channels; `_gru_inputs(h, o, raws, C, hp)` puts
them after the NBM channels (GRU_CIN 53 -> 61; G_HP = G_NB + 6, G_TOP = G_HP + 8). `infer()`: `SeriesScorer.push`
projects `NbmStream.h` (the state after this point) with the same numba `_hp_proj` as the batch path and passes it to
`_GruIn.push(..., hp)`. BLEND_W (0.5, 0.3, 0.2) -> **(0.4, 0.5, 0.1)**; GRU_EPOCHS 6 -> **7** (the existing rule:
production epochs = median best epoch; the hpca8 runs peaked at 4-8, median 7). CONFIG version `blend-v4-hpca8` (+
`nbm_hpca`), so the model set retrains. `verify_submission.py`: research GRU reference = the hpca8 pair; batch path
and timing section updated. `sbrt/nbm.py run_hidden` is embedded by `make_submission.py` (no change needed there).
Checks 2026-10-19: `python make_submission.py` -> main.py, standalone import OK (321 tree columns, 61 GRU inputs, 106
SVM); streaming vs batch GRU inputs in the new main.py with the v3 NBM and a stand-in PCA, 4 series (294-917 points):
**bit-identical in every block** (base, vol, BOCPD, NBM, hp, top-20). Not yet run: verify_submission (training),
crunch test, submission.
Backups: `submission_template_v3.py.bak`, `verify_submission_v3.py.bak`, `make_submission_v3.py.bak`,
`main_v3b_submitted.py.bak` (= submission #2, cloud 0.6420), `resources_v3/` (its trained models).
Inference cost: one 64 x 8 projection per point (~1 us); unchanged otherwise. Training ~+3-5 min (hidden states).
**Added 2026-10-19 (owner's override, section 27):** the SVM also takes the 8 hidden-state channels; CONFIG
`blend-v4-hpca8-svm`; details and checks in section 27.
**Pass bars for v4 (written before the runs):** (1) smoke `verify_submission.py --n-train 600 --n-val 200 --parity 5`:
runs end to end, logs "NBM hidden-state PCA: 8 components explain ...", PARITY OK (gru_in and gru ~0), numba vs torch
GRU ~1e-12, DETERMINISM identical; any traceback = stop. (2) full `verify_submission.py` (fold 0): production blend >=
research hpca8 blend - 0.002 and production GRU pair within ~0.003 of research (the v3 build was -0.0012 on the blend:
0.6632 vs 0.6644); a short GRU pair points first at the fixed 7 epochs. (3) Honest blend check, not blocking:
`blend.py` (section 24 blend command) with `--honest-paired`, logged next to the in-sample +0.0043 (the shipped weights
moved, GRU 0.3 -> 0.5).
Housekeeping: `diagnostics/replay_blend.py` (closed, section 14) calls `_gru_inputs` without the new `hp` argument and
no longer runs against main.py v4. The full verify retrains `verify_models/fold0` and overwrites the v3 verify models
there (only `resources/` is backed up, as `resources_v3/`).
**Expectation (written before any run):** ~15-20% chance: it is information the GRU does not see today, but the NBM
channels themselves moved the GRU +0.0037 and the blend only +0.0007 (section 7); blend +0.000 to +0.001.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one at a time; my estimate ~30-35 min for seed 0
the first time (~5 min per fold to extract the hidden states once), ~10 min for seed 1):**
1. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm.npz --tag aug2_vol_bo_nbm_hpca8`
2. Only if 1 is > 0: `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --seed 1 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --tag aug2_vol_bo_nbm_hpca8_s1`
3. Then: `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz --c cv_out/oof_svm_nbm.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm.npz cv_out/oof_gru_aug2_vol_bo_nbm_s1.npz --ref-c cv_out/oof_svm_nbm.npz`

## 25. SVM variants: per-step ranking loss `--svm-rank` and learned kernel widths `--svm-ard` (owner's request 2026-10-19; RESULT: both CLOSED)

**What (cv.py, `svm_fold` / `svm_ard`):**
- `--svm-rank a`: loss = (1 - a) x the old pointwise LS-SVM loss + a x a PAIRWISE loss inside each online step
  (every positive row vs every negative row of the same step, target margin 2, pair weights a_i b_j with step totals
  sqrt(Wpos_t Wneg_t) = the metric's n_pos x n_neg for uniform rows). Still closed form: per step the pairwise loss is
  r' (diag(d) - p q' - q p') r, so the Gram gets diag(d) plus the rank-2 terms Sp' Sn (Sp_t, Sn_t = per-step weighted
  feature sums, accumulated with index_add on the GPU). Checked against brute-force pair enumeration (max error 1e-12).
  Why: TS-AUC only compares series within a step; the pointwise loss spends capacity on step offsets and on pushing
  already-correct rows to +-1. a = 0 is bit-identical to the old code (checked).
- `--svm-ard family|column`: one RBF width multiplier per input family (t, ctx, nb, vol, bo, ev, dist, win, scan, full)
  or per column, learned per fold on an 80/20 SERIES split of the training rows (120k / 60k rows, 2,048 random
  features, closed-form ridge inside, Adam on the inner-validation loss of the same objective, 150 steps; ridge learned
  too, then the usual lambda grid). The widths are printed per fold. Validation fold untouched.
- Synthetic smoke test (two signal columns in 30): pointwise 0.7139, rank 1 0.7152, rank 0.5 0.7156, family ARD +
  rank 1 0.7184, column ARD + rank 1 0.7186 (column ARD found exactly the two signal columns).
- Production: not in main.py. Rank needs no inference change (same feature map, new coefficients); ARD only rescales
  the stored frequency matrix. Inference cost unchanged either way.

**Expectation:** SVM alone +0.002 to +0.006 for the ranking loss (~40%); the SVM has 0.1 of the blend, so the blend
moves +0.000 to +0.001. ARD: ~25%, similar size.

**Results (2026-10-19), 5 folds, paired vs `oof_svm_nbm` (0.6374):**
- `--svm-rank 1` (`oof_svm_nbm_rank1`): 0.6375, paired +0.0002 +- 0.0002, 4/5 (folds +0.0009 / +0.0005 / +0.0001 /
  0.0000 / -0.0003); best lambda moves from 1 to 1-3.
- `--svm-rank 0.5` (`oof_svm_nbm_rank0.5`): 0.6379 headline, paired +0.0001 +- 0.0001, 3/5.
- Verdict: the loss is not the SVM's bottleneck (the metric-matched pointwise weights plus t_log in the kernel already
  give nearly the per-step ranking solution). Rank 1 is carried into the ARD runs as the (marginal) winner; blend
  check per the standing rule, expected ~0.


---

- Blend with `oof_svm_nbm_rank1` as the SVM (owner's run): 0.6532 vs 0.6531, +0.0001 +- 0.0000, 5/5, same weights
  0.4 / 0.5 / 0.1: below the +0.0005 bar -> `--svm-rank` CLOSED.
- `--svm-ard family` on rank 1 (`svm_nbm_rank1_ardf`, owner stopped after fold 1): folds 0 / 1 -0.0008 / -0.0061 vs
  rank 1 -> CLOSED by the early stop. The inner split looked good (inner TS-AUC 0.639 -> 0.656 and 0.646 -> 0.662),
  but the learned widths disagree between folds (scan 0.76 vs 0.05, bo 1.03 vs 2.29) and both folds shrank t (0.08-0.09)
  and win (0.09-0.29). Reading: the widths fit the small inner model (2,048 features, 120k rows) rather than the full
  one, and shrinking t removes the time-conditioning the kernel needs (feature meaning changes with the step).
  `--svm-ard column` not run (more parameters, would overfit more).
## 26. CatBoost as a fourth blend member `cv.py --backend cat` (owner's go 2026-10-19; RESULT: CLOSED 2026-10-21, honest blend -0.0003, 0/5)

**Why:** the trees settings sweep is a local optimum and the vol-only second tree model got weight 0 (less information).
CatBoost keeps ALL 321 columns and changes only the learner: symmetric (oblivious) trees, one split per level, are
a different and more regularised function class than XGBoost's lossguide trees, so their errors may be less
correlated with the tree pair. Honest prior (section 9, lesson 2): new algorithms on the same information have not
moved the score; the TCN (another model on the same inputs) gave +0.0004 in-sample.
**What (`cv.py --backend cat`, catboost 1.2.10 installed 2026-10-19):** Logloss, the same rows, re-cuts, features and
metric-matched row weights as the trees; depth 6 symmetric, l2_leaf_reg 10, 254 bins, Bernoulli subsample 0.8,
Plain boosting; GPU with `--device cuda` (no per-split feature sampling on the GPU; `rsm = --ff` on the CPU only).
Options: `--cat-depth`, `--cat-grow SymmetricTree|Depthwise|Lossguide` (Lossguide uses `--leaves`/`--min-leaf`),
`--cat-ordered`, `--cat-border`. Raw scores per grid chunk via `ntree_start/ntree_end`; importance =
PredictionValuesChange. `--prune-keep` not supported with cat.
**Checks 2026-10-19:** 300-series smoke (1 fold, 100 rounds, GPU) runs end to end and writes oof/importance; GPU fit
works on the RTX 5080 (200k x 50 rows, 200 trees in 4.5 s); chunked raw scores summed = full prediction exactly (sd 0).
Second smoke on the exact command path (with `--nbm`, NBM torch on the GPU in the same process): runs, 66 s;
`torch.cuda.empty_cache()` added before the CatBoost GPU fit anyway. Smoke outputs deleted; `cv.py` still CRLF.
**Untested at full scale:** host RAM (CatBoost's Pool copies the ~4M x 321 float32 training matrix, ~5 GB more
than the XGBoost path: watch memory in fold 0) and runtime (my guess, unmeasured: 10-30 min per fold on the GPU). The XGBoost/LightGBM/SVM paths are unchanged (cat code only runs under
`args.backend == "cat"`).
**One configuration fixed in advance (no sweep on these folds):** depth 6, lr 0.05, 5000 rounds, grid 100. Symmetric
trees are weaker per tree than 31-leaf lossguide trees, so more rounds than XGBoost; the best round is picked per fold,
so a generous budget costs only time. Production would use a fixed round count (the CV best).
**Expectation (written before any run):** alone 0.636-0.643 (at or a bit below one XGBoost seed, 0.64); Spearman vs
the tree pair ~0.85-0.9; honest blend gain +0.000 to +0.001; my chance of passing the bar ~15%.
**DECISION RULE (written before any run; same as the TCN, section 17):**
1. CatBoost seed 0, 5 folds (`--compare` the XGBoost seed 0 for information only). Then `member_corr.py` vs the tree
   pair, and the 4-member blend vs the current 0.6531 blend with `--honest-paired`. **Early stop:** honest paired mean
   <= 0 -> CLOSED.
2. Otherwise CatBoost seed 1, then the 4-member blend with the CatBoost pair. **ADOPT if the HONEST paired gain >=
   +0.0005, in >= 4/5 folds, and > 2 SE.** The in-sample line is printed too but is not the test.
3. Report only, not a selection: CatBoost pair REPLACING the XGBoost pair in the 3-member blend.
**If adopted, production:** CatBoost in `submission_template.py` (catboost must be in the cloud image's requirements;
symmetric-tree inference is fast); 2 seeds, fixed rounds = the CV best round.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
1. `python cv.py --backend cat --device cuda --cat-depth 6 --lr 0.05 --rounds 5000 --grid 100 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag cat_d6`
2. `python diagnostics/member_corr.py --a cv_out/oof_cat_d6.npz --b cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
3. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz --c cv_out/oof_svm_nbm.npz --d cv_out/oof_cat_d6.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz --ref-c cv_out/oof_svm_nbm.npz --honest-paired`
4. (only if 3 is > 0) command 1 with `--seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag cat_d6_s1`,
   then command 3 with `--d cv_out/oof_cat_d6.npz cv_out/oof_cat_d6_s1.npz`.

**RESULT seed 0 (owner's run 2026-10-19, stopped by the owner during fold 2; no OOF file saved):** 1.34M original +
2.73M re-cut training rows; 5000 rounds in ~3 min per fold on the GPU (fast; host RAM was fine). Fold 0 best 0.6527 at
round 200 (last round 0.6251) vs XGBoost seed 0 0.6621: paired **-0.0094**; fold 1 0.6308 at round 200 (last 0.6030) vs
0.6329: **-0.0021**. Training logloss 0.69 -> 0.30 by round 5000: it overfits fast, like the XGBoost trees (production
200 rounds at lr 0.03). My config error: lr 0.05 put the peak at 4% of the budget, and grid 100 resolves it only to
+-100 rounds, so the peak score is understated a little (not by ~0.006). As a replacement for XGBoost it is clearly
worse (standing early stop: folds 0 and 1 both <= 0). As an EXTRA member it was not tested (no blend: no full OOF).
**Low-lr rerun (owner's request 2026-10-21; NOT RUN YET):** lr 0.015, 1500 rounds, grid 50 (the peak at round 200 of
lr 0.05 should move to ~600-700); same tag `cat_d6` (the first run saved no file). Rule fixed before the run: honest
paired 4-member blend vs the CURRENT 0.6527 reference (SWA GRU, SVM hpca8) <= 0 -> CLOSED for good; > 0 -> seed 1,
then the standing bar on the honest line.
**RESULT low-lr seed 0 (owner's run 2026-10-21, ~75 s per fold, `cv_out/oof_cat_d6.npz`):** headline **0.6395** at 600
rounds (flat 450-1200; per-fold best rounds 300 / 1400 / 250 / 1300 / 250); paired vs XGBoost seed 0 per fold -0.0076 /
-0.0024 / -0.0005 / +0.0027 / -0.0035 -> **-0.0023 +- 0.0017, 1/5**. The lower lr recovered ~+0.003 on folds 0-1 vs
lr 0.05. Alone it is weaker than one XGBoost seed (0.6417, section 7); importance leans more on the ctx_ history descriptors
(kurt/skew/logn/rho1 = 30% vs ~17% in XGBoost's top list). Spearman (member_corr) vs the XGBoost pair **0.889** (no-break 0.868): above GRU seed vs seed (0.81), i.e. close
to a copy of the trees.
**Blend (owner's run 2026-10-21) vs the 0.6527 reference:** in-sample best weights give CatBoost **0.0** (0.6527 =
0.6527); HONEST paired **-0.0003 +- 0.0002, 0/5** (folds 0 / 1 pick d = 0.1 and lose 0.0007 / 0.0008; folds 2-4 pick
d = 0). By the rule fixed before the run: **CLOSED for good.** Same lesson as the vol-only trees (section 21) and
the TCN: another learner on the same columns copies the trees (Spearman 0.89) and adds nothing.

## 27. NBM hidden-state PCA as SVM (and tree) inputs `cv.py --nbm-hpca 8` (owner's request 2026-10-19; RESULT: SVM ADOPTED by the owner's override, new best CV 0.6534; trees CLOSED)

**What:** section 24's idea for the other members (its next step 2). `cv.py --nbm-hpca K` appends columns
`nbh_pc0..K-1` (per fold, at the rows' steps 0, 3, 6, ...). It reads gru_cv.py's per-fold cache
`data/gru_nbmh8v1_f<f>_e25_h64_lr0.002_b24_wd0.0001_lf50_aug2_s0_m20.npz` (same NBM weights, same PCA), or builds
and saves it the same way if it is missing. The row layout is checked against cv.py's steps per source (exits on a
mismatch); fold-0 cache: 29,992 series, 1,682,129 original rows = cv.py's count. The SVM always takes the nbh_
columns (on top of t_log + ctx + top-100: 114 inputs, so gamma = gmul / 114 instead of / 106); the trees get them as
ordinary columns. Flag off = unchanged.
**Expectation:** the GRU gained +0.0063-0.0075; the trees and SVM already hold the 6 NBM summaries, so less. SVM
+0.001 to +0.004 (~45%); the SVM's blend weight 0.1 caps the blend effect unless the weights move.
**RESULT SVM (owner's run 2026-10-19, `cv_out/oof_svm_nbm_hpca8.npz`):** SVM 0.6412 vs 0.6374 (`oof_svm_nbm`); paired
**+0.0037 +- 0.0009, 5/5** (+0.0024 / +0.0058 / +0.0009 / +0.0055 / +0.0041). Gain at every step range: 0.5895 / 0.6352 /
0.6535 / 0.6841 (rank-1 run, ~= old SVM: 0.5832 / 0.6311 / 0.6493 / 0.6797). Best lambda moved from 1-3 to 0.3-1.
Hidden-state loads from the GRU caches worked on all folds. Now the strongest single-model gain the SVM has had
since the NBM summaries (+0.0049). Next: blend with this SVM replacing `oof_svm_nbm`; then the trees with the same flag.
**RESULT blend (owner's run 2026-10-19):** SVM member 0.6417 (blend.py scoring) in place of `oof_svm_nbm`: blend
**0.6534 vs 0.6531, +0.0003 +- 0.0001, 4/5** (+0.0001 / +0.0005 / -0.0000 / +0.0004 / +0.0005), weights unchanged
0.4 / 0.5 / 0.1 (0.3 / 0.5 / 0.2 ties at 0.6534). Consistent (~3 SE) but below the +0.0005 bar: NOT adopted by the
rule (owner's call; the 0.6489 two-seed trees were adopted by override at +0.0004, 3/5). As expected, the GRU already
carries the hidden-state information, so the SVM's own gain shrinks ~12x into the blend.
**RESULT trees seed 0 (owner stopped after fold 2):** -0.0031 / +0.0029 / -0.0026 (mean -0.0009, inside the seed noise
of +-0.0019 per fold). The early-stop rule did not formally trigger (fold 1 > 0), but 3 folds give no sign of a gain:
CLOSED. The trees do not use the hidden-state columns better than the 6 NBM summaries they already have.
**DECISION (owner, 2026-10-19): ADOPTED by override** at blend 0.6534 (+0.0003, 4/5; bar +0.0005). New best CV 0.6534;
reference files in section 6. Caveat for the record: second override on the same 5 folds (after the two-seed trees),
so the CV-to-cloud offset may widen by a few ten-thousandths.
**Production (folded into v4 before its verify; BUILT 2026-10-19 as CONFIG `blend-v4-hpca8-svm`, see the end of this section):**
1. `submission_template.py`: SVM inputs = the 106 `SVM_NAMES` columns of the tree row + the 8 hidden-state channels
   (the same `_nbm_hp()` / `_hp_proj` output the GRU gets; at training rows steps 0, 3, 6, ... as in cv.py). The hp
   channels are NOT tree columns, so `r[SVM_COLS]` becomes concat(r[SVM_COLS], hp) in `_svm_score` (batch) and
   `_svm_row` (streaming, `SeriesScorer.push` already has hp from the GRU path).
2. gamma = 0.5 / 114 (was / 106: `--svm-gmul 0.5` divides by the input count); lambda stays 0.3 (the CV average curve
   peaks at 0.3: 0.6412 vs 0.6403 at 1). Preprocessing unchanged (signed log of hp is near-identity for |x| <= 6; keep it
   identical to cv.py `_svm_prep`, which applies it to every column).
3. CONFIG version bump (e.g. `blend-v4-hpca8-svm`) so the model set retrains; `verify_submission.py` research SVM
   reference -> `cv_out/oof_svm_nbm_hpca8.npz`; `make_submission.py` baked SVM list 106 -> 114 names.
4. Checks as v4's pass bars (section 24) plus: streaming SVM input == batch SVM input bit-identical on a few series;
   fold-0 production SVM within ~0.002 of research (0.6553).
Inference cost: +8 columns in one 114 x 8192 product (~+8% of the SVM's ~200 us/point); hp is already computed.
**Built 2026-10-19:** `submission_template.py`: `_svm_in(rows, hp)` = concat(rows[:, SVM_COLS], hp) (float32, 114
columns); `_train_svm(X, HPr, ...)` with HPr = every series' hp channels at steps 0, 3, 6, ... (assert len == tree
rows), gamma = 0.5 / 114; in-sample `m_svm` on `_svm_in(X[is_orig], HPr[is_orig])`; `SeriesScorer.push` now projects
the NBM state BEFORE the SVM and fills `self.svx` (tree columns + hp32), the same hp32 goes to the GRU. CONFIG version
`blend-v4-hpca8-svm` (+ `svm_hp`), so any model set retrains. `verify_submission.py`: research SVM ->
`oof_svm_nbm_hpca8.npz`; batch path builds the SVM inputs with `_svm_in`; parity adds `svm_in` (streaming `sc.svx` vs
batch rows); timing counts the SVM row fill. `make_submission.py`: message only (SVM_NAMES stays the 106 tree columns).
Backups of the pre-change files: scratchpad only (the v4 files had no separate backup; v3 backups unchanged).
**Checks 2026-10-19:** main.py regenerated, standalone import OK (321 / 61 / 106 + 8). Parity with the v4 smoke model
set (`verify_models/fold0_n600`, copied) and a stand-in random 114-input SVM, 3 validation series (307-364 points):
SVM inputs streaming == batch **bit-identical**, SVM scores max diff 3e-8 (float32 product order, as before);
`_train_svm` on random data runs (Om 114 x 8192). Not yet run: verify smoke, full verify, crunch test.
**Pass bars (added to section 24's):** smoke `svm_in` 0 and `svm` < 1e-4 (PARITY OK); full verify: production SVM
within ~0.002 of research (`oof_svm_nbm_hpca8`, fold 0 research 0.6553) and production blend >= research blend - 0.002.
**Smoke verify (owner's run 2026-10-19, `--n-train 600 --n-val 200 --parity 5`): PASSED.** Trains in 123 s (SVM 18 s,
"106 tree columns + 8 hidden-state channels"); PCA 91.5% (small-sample, as in section 24's smoke). PARITY OK: tree
1.6e-6, svm 3.1e-7, **svm_in 0**, gru 0, gru_in 0, prob 5e-8; numba vs torch GRU 1.1e-15; determinism identical.
Timing 713 us/point (SVM 255 us, within the earlier 230-260 range): the 8 extra inputs cost nothing measurable.
Quality numbers meaningless at 600 series. Full verify running.


---

## 27. XGBoost variants: monotone evidence, residual boosting from nb_logodds, reg_lambda (owner's go 2026-10-19; RESULT: all three CLOSED 2026-10-20)

**What (`cv.py`, xgb only):**
- `--monotone`: 23 one-sided evidence columns may only RAISE the score (ev_mix_logodds, ev_logbf, all 16 dist_,
  bo_w/bo_wu logodds and logbf, nb_logodds); two-sided ones (lvr, mu, tsb, vol ratios, ev_post shares) stay free.
- `--base-margin nb_logodds`: per fold a weighted logistic fit (training rows and weights, 500k subsample) of y on
  [s, s*log1p(t), log1p(t)] gives every row a starting logit; the trees boost on the residual. Validation adds it once
  to the tree sum. Prints the slope in s per step and the margin's own validation TS-AUC. Production would need the
  3 coefficients + intercept per model.
- `--lambda X`: reg_lambda (default 10 = unchanged).
**Checks 2026-10-19:** 300-series smoke (2 folds) with all three flags runs end to end on the GPU; slopes positive
at every step in both folds. NOT verified: whether residual boosting behaves sensibly. On the smoke, fold 1 trees
(0.517) ended BELOW the margin alone (0.674), but with ~60 validation series per fold that is mostly noise; the
1,000-series comparison was stopped (owner's GPU busy). **Watch fold 0 of the base-margin run:** the "--base-margin
alone" line should be near nb_logodds alone (~0.63), and the trees' best should be above it; if the trees are below the
margin alone, stop and report (possible bug).
**DECISION RULE (standing, section 5):** each variant seed 0 paired vs the current trees seed 0; early stop if folds 0
and 1 are both <= 0. Otherwise seed 1, then the blend vs the 0.6531 blend with the new tree pair in `--a`; ADOPT if
>= +0.0005, >= 4/5 folds, > 2 SE.
**Expectation (before any run):** base margin +0.000 to +0.002; monotone +0.000 to +0.001; lambda ~0 to +0.001.
**2026-10-20, before the runs:** lambda value fixed at **30** (more L2 on leaf values; the data is data-limited, and
the 44.3 sweep never touched lambda). One value only, no sweep. Reference = trees seed 0 (best round in both, so the comparison
is like for like). TCN honest line (section 17) re-pointed to the 0.6527 reference; the TCN OOF still picks the best epoch
and has no hidden-state channels, so a positive line would need a `--swa 5-8 --nbm-hpca 8` TCN rerun before it counts.
**Full verify fold 0 (owner's run 2026-10-19, `blend-v4-hpca8-svm`):** train 1,476 s (features 1,066 s, SVM 196 s).
Production vs research OOF on the same 340,063 rows: trees 0.6592 vs 0.6640 (-0.0048), GRU pair 0.6545 vs 0.6622
(**-0.0077**), SVM 0.6566 vs 0.6553 (**+0.0013**), **BLEND 0.6675 vs 0.6721 (-0.0046)**. PARITY OK (svm_in 0, svm
2e-7, tree 1e-6, gru 0, prob 6e-8), determinism identical, 590 us/point (SVM 204 us) -> ~79 min per 8M points in one
process. **Blend bar (>= research - 0.002) MISSED; GRU bar (~0.003) MISSED; SVM and parity PASS.**
Diagnosis (no bug found): the research GRU OOF picks each fold's best epoch on the validation fold, and the hpca8 GRU's
fold-0 curve swings +-0.005-0.010 from epoch to epoch (seed 0: epoch 5 0.6580 best, epoch 7 0.6473; seed 1: epoch 7
0.6557 best, epoch 8 0.6477). Production uses fixed 7 epochs, so it lands at a typical epoch, not the best one; this
is selection optimism in the research number, bigger now than for the old GRU (-0.0041 then) because the curve is
noisier and the GRU's blend weight rose 0.3 -> 0.5. Trees -0.0048 ~ as before (-0.0041; research best rounds vs fixed
200). Against the old production build on the same rows: blend 0.6675 vs 0.6632 (0.6483 build, section 44 of the old
file) = **+0.0043 in production** (research said +0.0077 there): the change is real, about 55% of it survives.
Possible fix (not built): average the GRU over epochs (snapshot / weight averaging of epochs 5-8) or more GRU seeds,
to remove the epoch noise rather than pick an epoch.
**crunch test (owner's run 2026-10-19): PASSED.** `resources/` retrained as `blend-v4-hpca8-svm` (10,000 + 19,992
series, train 1,737 s: features 1,180 s, SVM 244 s, PCA 82.1%); inference 100 local series 50,983 points, 865 us/point
wall (744 own, 1 worker); determinism passed; 29:56 total, +3.0 GB memory. Next: `prepare_submission.py`, push from
`submission/`, Powerful instance, Force first train No.

**RESULTS 2026-10-20 (owner's runs, seed 0, ~60 s per fold):**
- `--monotone`: per fold -0.0058 / -0.0047 / +0.0003 / +0.0027 / +0.0020; folds 0 and 1 both < 0 -> **CLOSED by the
  early stop** (5-fold -0.0011 +- 0.0017, 3/5). The constraints cost on the folds where the evidence columns are
  noisiest; `nb_lvr` became the top feature.
- `--base-margin nb_logodds`: bug check passed (margin alone 0.595-0.624 on validation, trees above it in every fold;
  slopes in s positive and falling with the step, 0.63 at step 0 to 0.05 at 1000). Per fold -0.0055 / +0.0048 /
  +0.0004 / +0.0009 / -0.0009 -> **-0.0001 +- 0.0017, 3/5**: not > 0, so **CLOSED** by step 2 of the rule. Side effect:
  a flatter round curve (0.6411-0.6414 from 250 to 500 rounds), but its best fixed round (0.6414) does not beat the
  reference's (0.6417 at 150).
- `--lambda 30`: per fold -0.0037 / -0.0037 (best rounds 200 / 600; ~2x slower per fold); folds 0 and 1 both < 0 ->
  **CLOSED by the early stop** (owner stopped it in fold 2). More L2 on the leaves only costs here. The day-one tree settings
  stay a local optimum (old section 44.3), now including lambda.
**Tree round-selection check (2026-10-20, from the existing summaries; no new run):** the reference trees' fold-averaged
round curve is 0.6417 at 150 and **0.6414 at 200 (production)** for seed 0, 0.6412 at 200 for seed 1, while the per-fold
best-round mean is 0.6437 (seed 0). So the research tree number holds ~+0.002 of selection, but NO fixed round count
recovers it: production's 200 is already within 0.0003 of the best fixed count, and lr 0.015 had a flat curve at the
same level (0.6415-0.6418, old section 44.3). The proposed fixed-round / lr 0.015 tree test was **dropped before
building**. The fold-0 production trees gap (-0.0048) is NOT mainly round selection: on fold 0 seed 0's best round
WAS 200 (seed 1's was 500). The rest is something else in production vs research (unverified guesses: the production
NBM instance and its features, the re-cut draw). It would need a dedicated check (production trees with the research
NBM features on fold 0) before any fix.

---

## 28. Convolution exceedance on the whitened streams `diagnostics/conv_oracle.py` (owner's idea 2026-10-19; RESULT: CLOSED)

**Why:** after reading the ICT lab's MiniRocket study (033b: raw-series filters, linear head, -0.005 and an
undertrained solver). Owner's variant: calibrate the filters on each series' OWN history, run them only online, and
let the existing features carry everything else. Run on the bocpd.py whitened streams w / wu, where variance is
already removed, so any signal must be shape / dependence: aimed at the variance-unchanged breaks (blend 0.56-0.57).
**What:** 84 MiniRocket filters (length 9, +2/-1), dilations 1/2/4/8, causal; per-series history quantiles 0.1 / 0.9
as thresholds; trailing exceedance share over W = 32/128/512, z-scored against the same share over all history
windows. 144 agg_ columns (per channel / dilation / tail / W: mean z, mean z^2, max |z| over the 84 filters) + 672
per-filter columns for screening. Design, prior art, rule and expectation in the docstring.
**Checks 2026-10-19:** `--selftest`: causality on 10 series (prefix rows equal; altered future changes nothing
before it); rows align exactly with the OOF files; 40 series in 7.4 s (single process). Note: wu's history uses the
in-sample GARCH variance while online uses the frozen forecast (as in bocpd.py), so wu columns sit higher pre-break
(z^2 ~2.4 at W 512); same for both classes, adds noise only.
**DECISION RULE (written before any run):** gate = LightGBM [blend, step] vs [blend, step, 144 agg_] on series
5000-6999 (blend = the 0.6534 OOFs), 3 x 5 grouped folds, paired. PASS only if ALL >= +0.001 and > 2 SE, or the
FLAT slice (|post-break log var ratio| < 0.1) >= +0.005 and > 2 SE. PASS -> streaming `cv.py --conv` and the standing
rule (section 5). Otherwise CLOSED.
**Expectation (written before any run):** 10-15% to pass; blend +0.000 to +0.001 if built.
**Command (owner, ~10-20 min, CPU):** `python diagnostics/conv_oracle.py 2000 12` -> `cv_out/conv_oracle.txt`.
**RESULT (owner's run 2026-10-19, `cv_out/conv_oracle.txt`, 66 s features): CLOSED by the pre-written rule.**
Gate: ALL -0.0203 +- 0.0043 (1/15 folds > 0), FLAT -0.0287 +- 0.0071 (2/15). Slices: 113 UP / 244 FLAT / 309 DOWN
break series. Blend 0.6627 / 0.8291 / 0.6030 / 0.5996 (ALL / UP / FLAT / DOWN).
Reading:
- The design premise was wrong: whitening removes the HISTORY's scale, not a post-break variance change, so the
  zero-sum filter responses still scale with sd. The agg_ columns are signed variance detectors: UP 0.75-0.81, DOWN
  0.34-0.41 (reversed), ALL 0.55-0.58. A truly scale-free version (ranks / signs of the pattern) is ordinal patterns,
  already closed with no signal (section 8).
- FLAT single best 0.577 (agg_wu_d1_hi_w512_mz; per-filter best of 672 0.586, optimistic) is below the blend's 0.603
  on FLAT, though above every --dist column there (<= 0.522) and nearly uncorrelated with --dist (|rho| <= 0.11);
  rank correlation with the blend +0.25 to +0.34.
- The gate's size (-0.02) also says the stacker overfits: 144 smooth per-series columns, ~330 correlated rows per
  series and min-leaf 1000 (~3 series) let leaves memorise series; the 2-column base cannot. So the gate was biased
  against new columns (same mechanism as the stacking losses, section 23). The closure rests on the single-feature
  evidence and the variance-proxy finding, not on the -0.02 itself. If a stacker gate is reused: min-leaf in series
  (e.g. 20,000 rows), fewer rounds, and report the base stacker vs the raw blend.


---

## 29. GRU epoch averaging `gru_cv.py --swa 5-8` (owner's go 2026-10-20; RESULT: ADOPTED by the owner, blend 0.6527)

**Why:** v4 on the cloud scored 0.6474 (section 1). The fold-0 full verify (section 27) found that the
production GRU pair loses **-0.0077** against research and the blend loses -0.0046. Research picks each fold's best epoch on
the validation fold. Production trains a fixed 7 epochs, and the hpca8 GRU's validation curve swings +-0.005-0.010
from epoch to epoch, so production gets a typical epoch, not the best one. Averaging over epochs removes that noise
instead of picking an epoch. Target: 50th place on the public board is 0.0009 above v4's cloud score.
**What (`gru_cv.py --swa A-B --fixed-ep E`; GRU and TCN):** training runs exactly max(B, E) epochs (no early stop;
cosine schedule unchanged, T_max = --epochs = 25 as in production `GRU_TMAX`). Every fold is scored three ways, none
of which looks at the validation fold:
- `wavg`: the WEIGHTS averaged over the models after epochs A..B (equal weights, float64 sum; the GRU has no batch
  norm, so nothing needs recomputing). Free at inference: production would ship one averaged net per seed.
- `lavg`: the LOGITS of epochs A..B averaged (a snapshot ensemble). Costs (B-A+1)x the GRU at inference.
- `ep<E>`: the plain model after epoch E = what production ships today (E = 7).
The main OOF file is still the best epoch (the selection-optimistic research number), printed for information.
Extra files: `cv_out/oof_gru<tag>_wavg.npz`, `_lavg.npz`, `_ep7.npz`. Per fold it prints the three scores and
wavg - ep7, lavg - ep7, wavg - best; at the end, the paired means (same run, so the same seed and data order).
The window 5-8 was fixed BEFORE any run, from the best epochs of the hpca8 runs (4-8, median 7). No sweep of it on these folds.
**Checks 2026-10-20 (300 series, 2 folds):** `--swa 3-3 --fixed-ep 3`: wavg == lavg == ep3 exactly (max diff 0.0).
Flag off vs on with the same number of epochs (`--epochs 4` vs `--swa 2-4 --fixed-ep 4`): main OOF bit-identical. NOTE:
when --swa stops training earlier than the flag-off run would have, later folds see a different random stream (the
batch-order generator is shared across folds), so the main OOF of a --swa run is NOT the reference run's file. All
comparisons below are within one --swa run or blend vs blend. Full production flags, 300 series, 1 fold (`--swa 5-8 --fixed-ep 7`): runs end to end, 8 epochs, prints
the three scores; `data/` caches untouched (all caches are skipped when --n-series is set); outputs deleted. Backup of the pre-change file: scratchpad only. `gru_cv.py` still CRLF.
**DECISION RULE (written before any run):**
1. Seed 0. **Early stop:** if both `PAIRED wavg - ep7` and `PAIRED lavg - ep7` have a mean <= 0 -> CLOSED.
2. Seed 1, then the blend: GRU pair = the two `_wavg` files vs the reference GRU pair = the two `_ep7` files (trees and
   SVM unchanged; both sides are production-like, neither one selected on the validation fold). **ADOPT wavg if >=
   +0.0005, >= 4/5 folds, > 2 SE.** lavg goes through the same blend test, but only as a fallback if wavg fails (it
   costs 4x the GRU at inference; affordable under the 15 h budget, section 5).
3. Report only: the wavg blend vs the 0.6534 research blend (best-epoch GRU). Near 0 would mean production can now
   reach the research number.
**Expectation (written before any run):** wavg - ep7 per seed +0.002 to +0.006 (it removes most of the epoch noise
and adds a mild averaging gain); blend +0.001 to +0.003 vs the ep7 blend; ~55% chance of passing. Risk: weight
averaging along a fairly high learning rate (cosine at epochs 5-8 of 25 is still ~82-94% of the peak) can land
between minima. lavg is the safe version of the same idea.
**If adopted, production:** `submission_template.py` `_train_gru`: GRU_EPOCHS 7 -> 8, accumulate the state dict after
epochs 5-8 and save the average (wavg), or save the 4 snapshots and average their logits in infer (lavg). CONFIG version
bump; verify_submission research GRU reference -> the `_wavg` pair; full verify fold 0 bar: production GRU pair within
~0.002 of the research `_wavg` pair (no selection gap left to explain).
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one at a time; my estimate ~8-12 min per
seed, 8 epochs per fold with the cached inputs and hidden states):**
1. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz --tag aug2_vol_bo_nbm_hpca8_swa`
2. Only if 1 passes the early stop: the same with `--seed 1 --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz --tag aug2_vol_bo_nbm_hpca8_swa_s1`
3. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_ep7.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_ep7.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`
4. Report only: command 3 with `--ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz`.

**RESULT seed 0 (owner's run 2026-10-20, ~80 s per fold): PASSED the early stop.**
| | fold 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| ep7 (production today) | 0.6473 | 0.6350 | 0.6350 | 0.6311 | 0.6250 | 0.6347 |
| wavg 5-8 | 0.6578 | 0.6395 | 0.6374 | 0.6402 | 0.6349 | 0.6420 |
| lavg 5-8 | 0.6586 | 0.6398 | 0.6372 | 0.6412 | 0.6355 | 0.6425 |
| best epoch (selected) | 0.6580 (5) | 0.6376 (8) | 0.6444 (2) | 0.6394 (8) | 0.6393 (8) | 0.6437 |
- **wavg - ep7 +0.0073 +- 0.0016, 5/5; lavg - ep7 +0.0078 +- 0.0018, 5/5.** wavg vs the old research GRU seed 0
  (best epoch, `--compare`) -0.0002 +- 0.0026: averaging recovers essentially the WHOLE selection gap without looking at
  the validation fold. ep7 vs the research file -0.0075 matches the fold-0 production verify (-0.0077 on the pair).
- lavg is only +0.0005 over wavg: weight averaging is nearly as good and free at inference, so wavg stays the candidate.
- A PyTorch warning ("RNN module weights are not part of single contiguous chunk") comes from the deep copy of the
  model for the averaged pass; it only costs memory/speed for that one evaluation, results unaffected.
**RESULT seed 1 (owner's run 2026-10-20):** ep7 / wavg / lavg / best epoch = 0.6366 / 0.6395 / 0.6396 / 0.6409.
wavg - ep7 **+0.0029 +- 0.0020, 3/5** (folds 0 / 1 -0.0013 / -0.0015, where epoch 7 happened to be the best epoch;
folds 2-4 +0.0063 / +0.0087 / +0.0023). wavg vs the research best-epoch seed 1 -0.0025 +- 0.0008 (0/5).
Two-seed member summary: wavg - ep7 +0.0051 on average; averaging removes the bad-epoch cases and loses a little
when the fixed epoch happens to be a good one, as expected.
**Blend (command 3), wavg pair vs ep7 pair, trees and SVM unchanged:** 0.6527 {0.4, 0.4, 0.2} vs 0.6517 {0.4, 0.4,
0.2}; per fold +0.0004 / -0.0004 / +0.0006 / +0.0021 / +0.0005 -> **+0.0006 +- 0.0004, 4/5**. The bar: >= +0.0005 yes,
>= 4/5 yes, > 2 SE NO (1.5 SE). Members: GRU pair 0.6424, trees 0.6443, SVM 0.6417.
Reading: the member gain (+0.0051) shrinks ~8x in the blend, the usual pattern (section 7): the trees and SVM
already cover much of what the GRU's bad epochs lose. The fold-0 production verify blend loss (-0.0046) has other
parts too (trees' fixed 200 rounds -0.0048 on the member).
**DECISION (owner, 2026-10-20): ADOPTED** ("it helps"). Third adoption below the full bar on the same 5 folds (after
the two-seed trees and the hpca8 SVM); unlike those it removes a selection bias rather than adding one, and the
member effect is large and consistent on seed 0.
**New reference:** blend 0.6527, weights 0.4 / 0.4 / 0.2, GRU files `_swa_wavg` / `_swa_s1_wavg` (section 6).
**Production v5 (BUILT 2026-10-20, `blend-v5-swa`; not trained or verified yet):**
- `submission_template.py`: GRU_EPOCHS 7 -> 8 (GRU_TMAX stays 25); new `GRU_SWA_A, GRU_SWA_B = 5, 8`. `_train_gru`
  adds the floating-point state dict after epochs 5, 6, 7, 8 into a float64 sum (`wsum` / `wcnt`, same code as
  gru_cv.py --swa), then loads the equal-weight average back INTO the trained model (in place, so the weights stay one
  contiguous chunk; expected: no RNN warning, unchecked on CUDA) before `model.eval()`. So both the saved weights and the in-sample logits for the
  blend scale (`sd_gru`) come from the averaged net. Logs "GRU seed s: weights averaged over epochs 5-8 (4 models)".
  BLEND_W (0.4, 0.5, 0.1) -> (0.4, 0.4, 0.2). CONFIG version `blend-v5-swa`, plus `gru_swa=[5, 8]` in CONFIG, so any
  old model set retrains.
- `verify_submission.py`: research GRU = `oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz` + `..._swa_s1_wavg.npz`
  (both exist in cv_out). The research blend takes its weights from `main.BLEND_W`, so it is now 0.4 / 0.4 / 0.2 = the
  0.6527 reference. `make_submission.py` unchanged (re-run only).
- Inference unchanged (one net per seed, same shapes). Training: +1 GRU epoch per seed (~+1-2 min).
- Backups of the v4 files: `submission_template_v4.py.bak`, `verify_submission_v4.py.bak`, `make_submission_v4.py.bak`,
  `main_v4.py.bak` (= the v4 build, cloud 0.6474). All files kept CRLF.
**Checks done (assistant, 2026-10-20):** `python make_submission.py` -> main.py regenerated, "standalone import OK: 321
61 106", CRLF. `_train_gru` from main.py on random data (24 series, CPU, 8 epochs): the saved weights equal the manual
mean of the epoch 5-8 state dicts exactly (max diff 0.0; vs the plain epoch-8 net 8.6e-3), 4 models averaged, in-sample
logits produced. `py_compile` of verify_submission.py OK. The two `_wavg` research files: all OOF values finite, sid / step / y
identical to the old best-epoch GRU file. Not run: any training on real data, verify, crunch test.
**Pass bars for v5 (written before the runs):**
1. Smoke `verify_submission.py --n-train 600 --n-val 200 --parity 5`: runs end to end, log shows 8 epochs and
   "weights averaged over epochs 5-8 (4 models)" for both seeds, PARITY OK as v4 (tree ~1e-6, svm ~1e-7, svm_in 0,
   gru 0, gru_in 0, prob ~1e-7), numba vs torch GRU ~1e-15, DETERMINISM identical, no "not part of single contiguous chunk"
   warning; any traceback = stop. Quality
   numbers meaningless at that size.
2. Full `verify_submission.py` (fold 0): **production GRU pair within ~0.002 of the research `_wavg` pair** (no
   selection gap left; v4 was -0.0077 vs the best-epoch pair) and **production blend >= research blend - 0.002**. The
   trees still carry their fixed-200-rounds gap (-0.0048 on the member in v4) at weight 0.4, so if the GRU bar passes
   and the blend misses by a little (-0.002 to -0.003), that is the known trees' gap, not a v5 bug. Report only:
   production blend vs the v4 production blend on the same rows (0.6675); expected higher (the CV said +0.0006 to
   +0.001; the v4 production GRU had the bad fold-0 epoch 7, so possibly more).
3. `crunch test`: retrains `resources/` as `blend-v5-swa`, inference on 100 local series, determinism; us/point in the
   v4 range (~600-870 locally).
4. `prepare_submission.py`: all its checks pass (model set `blend-v5-swa` == main.py CONFIG, train() skips without
   torch, staged outputs == crunch test predictions).
**Commands (PowerShell, from the project folder, one at a time; the owner runs them):**
```
$env:KMP_DUPLICATE_LIB_OK="TRUE"
Copy-Item -Recurse resources resources_v4
python verify_submission.py --n-train 600 --n-val 200 --parity 5
python verify_submission.py
crunch test
python prepare_submission.py
cd submission; crunch push --dry -m "v5 blend-v5-swa: GRU weight average epochs 5-8, blend 0.4/0.4/0.2"
crunch push -m "v5 blend-v5-swa: GRU weight average epochs 5-8, blend 0.4/0.4/0.2"
```
`Copy-Item` keeps the v4 trained models (crunch test overwrites `resources/`). The full verify retrains
`verify_models/fold0` and overwrites the v4 verify models there. Push only from `submission/`, never from the project
folder. On the site: Powerful instance, "Force first train" No.

---

## 30. More hidden-state components for the GRU: `--nbm-hpca 16` and `--nbm-hpca 32` (owner's go 2026-10-20; RESULT: both CLOSED)

**Why:** step 3 of section 24. The 8 PCA channels explain only 80-84% of the NBM hidden state's variance on the full
data, and the hidden state was the largest recent gain (GRU +0.006-0.008 per seed). The question is whether the
remaining 16-20% carries detection signal or is noise.
**What:** no new code. `--nbm-hpca K` already works for any K <= 64 (PCA per fold on the online parts of every 10th
original training series, each component standardised to sd 1, clipped +-6). Both runs use the adopted epoch
averaging (`--swa 5-8`), so the test is the `wavg` line against the current `_swa_wavg` file: same seed, same 8
epochs, same batch order, and only the input channels differ.
**Why not the full 64-d state:** the GPU holds all inputs at once. 61 channels take 8.0 GB; 117 channels (K = 64) would take
~15.4 GB + ~1.1 GB for training, more than the 16 GB card (and Windows would spill into RAM and crawl, section 4).
K = 32 (85 channels, ~11.2 GB + 1.1) fits. On the 300-series smoke, 32 components explain 99.7% of the variance, so
K = 32 is the full state in practice. Caveat: standardising every component to sd 1 blows the near-zero-variance ones
up into unit-scale noise channels. Whether the GRU learns to ignore them is part of what K = 32 tests.
**Checks 2026-10-20:** 300-series smoke, 1 fold, full flags, `--nbm-hpca 32 --swa 5-8`: runs end to end; caches
skipped (`--n-series`); outputs deleted. Cache files on the full runs: `data/gru_nbmh16v1_f*` ~1 GB per fold,
`gru_nbmh32v1_f*` ~2 GB per fold (589 GB free).
**DECISION RULE (written before any run):**
1. Seed 0 for K = 16 and for K = 32, each read on its `wavg ... vs --compare` line (paired vs
   `oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz`). **Early stop per variant:** mean <= 0 -> CLOSED.
2. If both pass, only the one with the larger seed-0 gain goes on (a mild selection between two, noted).
   Seed 1 vs `oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz`.
3. Blend with its two `_wavg` files in `--b` vs the 0.6527 reference (section 6, the 2026-10-20 command). **ADOPT if >=
   +0.0005, >= 4/5 folds, > 2 SE.**
4. If adopted: the SVM with the same K (`cv.py --backend svm ... --nbm-hpca K`) as its own step.
**Expectation (written before any run):** per seed +0.000 to +0.003 for K = 16, K = 32 the same or worse (noise
channels); blend +0.000 to +0.001; ~20-25% chance to pass. The first 8 components hold most of the variance, and
the GRU's own state already compresses its inputs.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one at a time, nothing else on the GPU; my
estimate ~30-35 min each the first time, since the hidden states are extracted once per fold (~5 min), then ~7 min):**
1. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 16 --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca16_swa`
2. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 32 --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca32_swa`
3. Winner K only: the same with `--seed 1 --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --tag aug2_vol_bo_nbm_hpca<K>_swa_s1`
4. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca<K>_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca<K>_swa_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`
**RESULT K = 16 seed 0 (owner's run 2026-10-20; hidden-state extraction ~190 s per fold, 9.1 GB inputs, peak 10.2
GB): CLOSED by the early stop.** 16 components explain 91-93% on the full data (the first 8 ~80%).
| | fold 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| K = 8 wavg (reference) | 0.6578 | 0.6395 | 0.6374 | 0.6402 | 0.6349 | 0.6420 |
| K = 16 wavg | 0.6478 | 0.6331 | 0.6301 | 0.6390 | 0.6377 | 0.6375 |
**wavg -0.0044 +- 0.0023, 1/5; lavg -0.0038; ep7 -0.0128; best epoch -0.0013 (2/5).** Epoch averaging itself still
works here (wavg - ep7 +0.0083, 4/5), so the loss comes from the inputs, not the training.
Reading: training loss at epoch 8 is LOWER than with K = 8 (0.6093-0.6146 vs 0.6137-0.6231) while validation is worse:
the 8 extra unit-scaled components (each 0.7-2.6% of the variance) give the GRU more to memorise and no detection
signal it can use. Consistent with section 8 "data-limited, not capacity-limited". K = 32 adds even smaller
components, so it is expected to be worse; recommended not to run it (owner's call; the rule allowed it).
Not tried, and low prior: K = 4 (fewer channels). It would be tuning K on the same folds, so it would need the full bar.
**RESULT K = 32 seed 0 (owner's run 2026-10-20, stopped by the owner during fold 4): CLOSED.** 32 components explain
97.4-97.9%. 11.2 GB inputs, fine on the GPU. wavg 0.6524 / 0.6358 / 0.6291 / 0.6347 vs K = 8 wavg 0.6578 / 0.6395 /
0.6374 / 0.6402: **-0.0054 / -0.0037 / -0.0083 / -0.0055, 0/4, mean -0.0057** (worse than K = 16's -0.0044).
Training loss at epoch 8 drops further (0.5993-0.6048). Monotone in K: 8 > 16 > 32. The extra hidden-state
directions are memorisable, not predictive. The hidden-state line of work stops at K = 8 for the GRU.

---

## 31. Detectability weights `cv.py --detw 0.2` (owner's go 2026-10-20, option 3 of "sources of new information"; RESULT: CLOSED, two-seed trees -0.0007)

**Why:** some breaks change nothing measurable (4.4% are "quiet" on all 5 simple axes, blend 0.487 on them; old
section 15.2), and bamasa reports ~30% invisible to 49 statistics. Their post-break rows teach the trees that "looks
normal" can be positive, which is label noise for a detector. The training labels hold information we have not used:
the known break position lets us measure how much each training break really changed the series.
**What (`--detw FLOOR`, training rows only):** for every training break series (originals AND re-cut variants, each
against its own history), change size s = max(|log var ratio| / 0.1, |log diff-var ratio| / 0.1, |d rho1| / 0.03,
|d kurtosis| / 0.5, |mean shift| / 2 SE) over the WHOLE post-break part (the "quiet" thresholds of
`diagnostics/error_breakdown.py`; s < 1 = quiet on all axes). Needs >= 30 post-break points, else no change. Positive
rows get multiplier min(1, max(FLOOR, s / 2)); then every step's positives are rescaled to their old total weight,
so class balance and the metric weighting are unchanged and only the mix of positives shifts toward visible breaks.
Validation rows, labels and features are untouched (the OOF stays honest). Production change if adopted: the same
weights in `submission_template.py` training (the break positions are known there too). Inference unchanged.
FLOOR fixed at 0.2 BEFORE any run; no sweep.
**Checks 2026-10-20 (300 series, 2 folds, `--ev compact --vol --aug 2`):** runs end to end; 140 measured break
series: quiet (s < 1) 6.4%, multiplier < 1 (s < 2) 27.9%, median s 3.7; re-cut row layout assert passes; 29-37%
of positive training rows down-weighted, mean multiplier 0.67. `--detw 0` runs the old code path (the
new code only runs when > 0). Backup of the pre-change cv.py: scratchpad only. cv.py still CRLF.
**DECISION RULE (written before any run; standing rule, section 5):** seed 0 paired vs trees seed 0; early stop if
folds 0 and 1 are both <= 0. Otherwise seed 1, then the blend with the new tree pair vs the 0.6527 reference (section
6, 2026-10-20 command); ADOPT if >= +0.0005, >= 4/5 folds, > 2 SE. If adopted: the GRU with the same weights as its
own step (`gru_cv.py` has no --detw yet).
**Expectation (written before any run):** trees +0.000 to +0.002; ~15% chance to pass. Risk: some "quiet" breaks are
detectable on axes the 5 measures miss, so down-weighting them throws away real signal; the floor 0.2 limits that.
**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU; ~6 min per seed):**
1. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --detw 0.2 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_detw`
2. Only if 1 passes the early stop: the same with `--seed 1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --tag aug2_df_bo_nbm_bs24_ep25_detw_s1`
3. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_detw.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_detw_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`

**RESULT seed 0 (owner's run 2026-10-20, ~60 s per fold):** full data: 4,364 measured break series, quiet (s < 1)
4.4% (matches section 15.2), multiplier < 1 for 19.8%, median size 4.1; ~24.5% of positive training rows
down-weighted (mean multiplier 0.67-0.69). Per fold **-0.0052 / +0.0009 / +0.0019 / +0.0026 / +0.0005 -> +0.0001 +-
0.0014, 4/5** (best round in both). Early stop NOT triggered (fold 1 > 0) and the 5-fold mean is > 0, so the rule
says: seed 1, then the blend. Honest read: about zero so far. Side numbers: pooled OOF 0.6433 vs 0.6420; step buckets
0-100 0.5853 vs 0.5807, 100-300 0.6343 vs 0.6339, 300-600 0.6594 vs 0.6595, 600+ 0.6860 vs 0.6791 (gains at the
ends, where the weight of the metric is small); fold-averaged round curve 0.6407 at 200 rounds vs the reference
0.6414 (fixed-round view slightly worse).
**RESULT seed 1 (owner's run 2026-10-20):** per fold -0.0039 / +0.0026 / -0.0003 / -0.0039 / +0.0012 -> **-0.0009 +-
0.0013, 2/5**. **Two seeds averaged vs the two-seed reference (`diagnostics/oof_paired.py`): -0.0007 +- 0.0011, 3/5**
(fold 0 -0.0051 again; pooled 0.6438 vs 0.6443). Small gains at steps 0-100 and 600+ (+0.0011 / +0.0024), losses in
the middle steps that carry most of the metric's weight (100-300 -0.0012, 300-600 -0.0010).
**CLOSED** (decision 2026-10-20): the tree pair is below the reference, so the blend cannot reach +0.0005; blend run
skipped. Reading: down-weighting "quiet" breaks does not clean the labels in a useful way. Part of what the 5 axes
call quiet is still learnable (section 16: dependence traces), and the trees lose it. Consistent with the label
information being mostly used already.

---

## 32. Expected no-break drift of the detectors from richer history inputs `diagnostics/drift_gate.py` (owner's go 2026-10-20, option 2; RESULT: CLOSED by the kill test)

**Idea:** the volatility state (+0.007, our largest feature win) worked because the no-break variance drift is
predictable from the end of the history. Option 2 extends that to the detectors: model each key detector's expected
NO-BREAK value from the history and give the trees "observed minus expected".
**New mechanism required (section 8 closed a lot of this lever: per-series replay null calibration, self-replay blend
calibration, 32 extra history descriptors, hidden-state PCA in the trees):** a drift model built only from columns
the trees already have (ctx_*, vol constants, step) adds nothing the trees cannot learn. It is new only if RICHER
history inputs predict the no-break detector values. So the gate compares, on no-break series at steps 99 / 300 / 600,
out-of-fold R^2 of A = LightGBM on [ctx_* 5, vol constants 4] vs B = A + the fold NBM's end-of-history hidden state (8
PCA channels) + 18 end-of-history window statistics (last 100 / 200 / 500 points: log var ratio, kurtosis, rho1, share
|z| > 2 and > 3, log diff-var ratio). cv.py folds; validation series use the NBM of the fold that never saw them.
**ONE-SIDED KILL TEST (written before the run):** close unless a top detector (nb_lvr, nb_logodds, bo_wu_lvr,
win10_tail3) gains dR^2 >= 0.02 and > 2 SE at >= 2 of 3 steps. A pass would only have meant "build the columns".
**RESULT (2026-10-20, 28 s on the CPU, so I ran it myself; `cv_out/drift_gate.txt`): FAIL -> CLOSED.**
- The striking part is A, not B: **the no-break value of every detector is essentially unpredictable from the
  history** (R^2 A between -0.07 and +0.10; most about 0). The detectors are already calibrated against each
  series' own history by design (ECDF whitening, the frozen GARCH forecast, the NBM's predictive scale), so there is
  no predictable no-break drift left to subtract. The volatility state already took the part that existed.
- B adds at most a few % of variance: nb_lvr at step 300 +0.022 +- 0.008 (the only top-detector pass, 1 of 3 steps),
  bo_wu_lvr 300 +0.022 (1.9 SE), bo_wu_logodds 600 +0.045 (not a top detector), the rest about 0 or negative.
- Reading: the "no-break series wander as much as a break" obstacle (section 3) is now mostly UNPREDICTABLE
  wandering, not history-predictable drift. That bounds every future "condition on the history better" idea.
Inference cost had it passed: one predicted path per series at the start of its online part (negligible).

## 33. Literature sweep: what can be measured in a time series vs what we measure (owner's question 2026-10-20; RESULT: review only, nothing built)

Full report: `reports/measurement_coverage_2026-10-20.md`. Sources were hctsa, catch22, tsfresh, TSFEL, the
change-point literature (spectral, tail dependence), the 2025 2nd place, szukiyu40 and bamasa. The comparison is by the
kind of change a statistic detects, not by library: whole-series library features matter here only as a
history-vs-online change on a cross-series scale.
Verdict: every axis is in production or closed. Those axes are level, scale, roughness, short and long linear
dependence, spectrum and scaling, marginal shape, tail rates, nonlinear dependence, symbolic and ordinal patterns,
entropy, model-based surprise, learned representations, number format, and cross-series.
**Never tested (small prior, about +0.000 to +0.001 blend):**
(1) clustering and timing of tail exceedances (extremal index, catch22 outlier-timing);
(2) quantilogram (tail dependence at lags 1-5);
(3) an explicit tail index (Hill / GPD) on the vol-adjusted u.
Cheapest test, if wanted: add them as a family to `diagnostics/info_oracle.py` under the old §34 kill rule. Not built.

**Oracle built 2026-10-20: `diagnostics/tail_oracle.py` (ready to run, owner's full run pending).**
- Setup: the same as `longrange_oracle.py`. Break position known; L = 100 / 300 / 600; the same 15-stat baseline; the
  segment is compared against history windows of the same length; same kill rule.
- Family `tl_`, 18 statistics, all scale-free (the segment's own thresholds and quantiles), on e and on
  v = e / causal EWMA volatility (half-life 20):
  - extremal index (intervals estimator), gap CV and lag-1 follow probability of the top-10% |x|;
  - quantilogram at the 10% / 90% quantiles, lag 1 and the sum over lags 1-5;
  - Hill index of |x| and the upper-minus-lower Hill index.
- Estimator check on synthetic data (600 points): iid gives extremal index 0.97 and quantilogram ~0; GARCH gives
  0.42 / +0.10; paired shocks at fixed variance give 0.53 / +0.12; t3 raises Hill from 0.22 to 0.41.
- The verdict also tests each subfamily alone (EXC 6 / QG 8 / HILL 4 columns; a subfamily needs > 3 SE on the
  matched slice), and reports each column's max rank correlation with the baseline on the matched rows (the section
  28 lesson). Caveat: v's clustering columns are partly variance detectors on all rows (the EWMA lags a break), so
  the matched slice is the clean read for v.
- Smoke run (300 series, `cv_out/tail_oracle_smoke.txt`) runs end to end. It is unreadable: about 100
  variance-matched segments, SE about +-0.04.
Command (CPU, about 3-6 min): `$env:KMP_DUPLICATE_LIB_OK="TRUE"; python diagnostics/tail_oracle.py` -> `cv_out/tail_oracle.txt`.
Expectation, written before the run: FAIL.

**RESULT (owner's full run 2026-10-20, `cv_out/tail_oracle.txt`): formal PASS on one cell only; CLOSED by the owner 2026-10-20.**
- Incremental over the baseline (15 paired folds):

  | Family | L = 100 (all / matched) | L = 300 (all / matched) | L = 600 (all / matched) |
  |---|---|---|---|
  | whole family | -0.0098 / -0.0204 | -0.0051 / -0.0041 | -0.0090 / -0.0030 |
  | EXC | -0.0044 / -0.0102 | +0.0014 / +0.0054 +- 0.0044 | -0.0044 / -0.0073 |
  | HILL | -0.0062 / -0.0091 | -0.0023 / -0.0008 | -0.0134 / -0.0241 |
  | QG | -0.0025 / +0.0006 | -0.0008 / +0.0000 | +0.0076 +- 0.0049 / **+0.0252 +- 0.0064, 14/15** |

  The bold cell (QG, L = 600, matched slice) is the only one that clears the pre-written subfamily bar (> 3 SE).
- Single statistics are all 0.49-0.57. The strongest is the lower-tail quantilogram at lag 1 on e: 0.56 on the
  matched slice at L = 600. Its max rank correlation with the baseline is 0.36 (b_eac1). Exceedance clustering and
  Hill add nothing at any L.
- **Why close despite the PASS: the metric cannot reward it.** Share of TS-AUC weight on positive rows with at least
  k post-break points (computed on train):

  | k | 100 | 300 | 450 | 600 |
  |---|---|---|---|---|
  | weight share | 0.567 | 0.170 | 0.057 | 0.015 |

  QG is flat at L = 300. Ceiling at face value: 0.015 x ~0.30 (matched share of those breaks) x 0.025 ~ +0.0001;
  +0.0004 if it held on every row with 600+ post-break points. That is before the usual 3-5x shrink into the trees,
  about 2x into the blend, and the dilution of a streaming window that is still partly pre-break. It cannot reach
  the +0.0005 adoption bar.
- Secondary caveats:
  - the 15 folds are 3 repeats of the same data, so the SE is optimistic;
  - there were 24 looks in total;
  - only 151 variance-matched breaks at L = 600;
  - production has much richer lag structure than this baseline;
  - the MMD precedent also passed this oracle and then added nothing.
- The pre-run expectation ("FAIL") was wrong on the letter of the rule; it is right in effect.
- If the owner builds it anyway: a `cv.py` quantilogram family (lower and upper 10% quantiles, lags 1-5, full online
  part and last 600, history-relative), judged by the standing paired-trees rule (section 5).

---

## 34. Small-break detector variants `cv.py --sens both` (owner's "test it" 2026-10-20 after "which knobs make the models more sensitive?"; RESULT: CLOSED, owner stopped in fold 3)

- **Idea.** Two production detectors are tuned toward big breaks; most misses are breaks that barely move the
  variance (section 3). Same kernels, streams and hazard; only the alternatives / the prior change
  (`sbrt/sensdet.py`, no labels used):
  - `ev2_` (6): the evidence kernel on 14 SMALL alternatives (variance x0.85/0.9/0.95/1.05/1.1/1.15, mean +-0.1/0.2,
    dependence +-0.1, Student-t nu 10/20); the `--ev compact` set of columns.
  - `bn_` (6): BOCPD with a narrow post-break variance prior, Inverse-Gamma mean 1.1, sd 0.15 (alpha0 55.78, beta0
    60.26, kappa0 490) vs production's EB prior (mean 1.23, sd 1.21); logodds / logbf / lvr on w and wu.
- **Self-test** (`python -m sbrt.sensdet`, held-out series 5000-6999): wrapper == evidence.py, streaming == batch,
  causal, finite: OK. Single-feature TS-AUC, new vs production: ev2_mix_logodds 0.6075 vs ev_mix_logodds 0.5932
  (+0.014); bn_w_logodds 0.6123 vs bo_w_logodds 0.6046 (+0.008); bn_wu 0.6059 vs bo_wu 0.6004 (+0.006). Rank
  correlations with the production columns 0.78 / 0.89 / 0.90. Past pattern: single-feature gains shrink 3-5x into
  the trees, so expect at most a few 1e-4 in the blend.
- cv.py smoke (300 series, aug 2, 2 folds): runs; `bn_` columns used by the trees. `--sens none` (default) leaves
  every code path unchanged. Cache: `data/sens1both_<stub>.npz`.
- **Test (standing rule, section 5):** paired trees seed 0 = the current best tree command + `--sens both --tag
  aug2_df_bo_nbm_bs24_ep25_sensb`. Early stop if folds 0 and 1 are both <= 0. Otherwise seed 1, then the blend
  with the reference trees replaced. If it passes, ablate `--sens ev` / `--sens bo`; the GRU/SVM versions come only
  after that. Production would need `SD.SensStream` wired into `main.py` (not done).
- **Result (2026-10-20), trees seed 0 paired vs the reference:** fold 0 -0.0037, fold 1 +0.0015, fold 2 -0.0004
  (mean -0.0009 over 3 folds, 1/3); the owner stopped it in fold 3. The early-stop rule was not formally triggered
  (fold 1 > 0), but the fold-0 loss is as large as the best possible gain. CLOSED. Reading: the +0.008-0.014
  single-feature gains were information the trees already had (scans / windows of e10sq and tail flags = small-change
  CUSUMs in disguise); 12 more correlated columns only added noise, the usual 'redundant features hurt' pattern.

---

## 35. Transductive NBM adaptation on the test stream `diagnostics/transductive_gate.py` (owner's idea 2026-10-20; RESULT: default split CLOSED by its rule 2026-10-20)

- **Idea.** `infer` streams all test series through one process per worker, so the shared NBM could be fine-tuned on
  the test histories (A) and on the online segments of series already finished (B, down-weighted 0.3). Score
  recalibration across series (C) was dropped: a per-step monotone transform shared by all series cannot change
  TS-AUC, and the series-specific version is section 32 (R^2 ~0) plus `--xsec` (section 22), both closed.
- **Arms:** base / pool_hist_all (ceiling, not causal) / pool_hist (A, causal, blocks of 250) / pool_hist_online (B).
  Eval = series 5000-6999 as in nbm_sweep; `--shift kurt|rho1|arch` builds a deliberately shifted split;
  `--shards K` = K interleaved streams (production has ~830 series per worker).
- **Decision rule** in the docstring (written before any full run): a causal arm needs paired nb_logodds >= +0.003,
  > 2 SE, on both seeds, with no-break whitened variance no worse than base. Then `--shards 2`, then a cv.py build.
- **Limits:** in the default split CV can only measure "more same-distribution unlabelled data" (section 15: ~0).
  Shift probe in the smoke (not size-dependent): all 6 history statistics of X_test.reduced match training (KS
  p >= 0.16), so no visible shift in the 100 test series.
- **Blocker before anything could ship:** the 10% determinism re-run (tolerance 1e-8). Cross-series state makes a
  series' output depend on which series came before it in its worker. Unknown how the runner splits and re-runs
  series: confirm that protocol first.
- **RESULT default split (owner's run 2026-10-20, 528 s, `cv_out/transductive_gate.txt`):** base reproduces
  nbm_sweep (0.6316 seed 0). Paired nb_logodds vs base, seed 0 / seed 1: ceiling pool_hist_all +0.0016 +- 0.0010 /
  +0.0025 +- 0.0008; pool_hist (A) +0.0013 +- 0.0007 / +0.0008 +- 0.0005; pool_hist_online (B) +0.0022 +- 0.0006 /
  -0.0014 +- 0.0009. All below the +0.003 bar; the ceiling fails, so A and B are CLOSED for this split. Most block
  fine-tunes chose epoch 0 (no improvement on held-out histories): the NBM is saturated, as section 15 found. A is
  consistently slightly positive, but single-feature gains shrink 3-5x into the trees (~+0.0003). Shift probe
  silent (KS min p 0.159; NLL gap +0.9 SE). By the rule a `--shift kurt` pass could not justify a build while the
  probe is silent, so that run is optional (informational only).
- **Commands** (one at a time, GPU, ~20-30 min each):
  `$env:KMP_DUPLICATE_LIB_OK="TRUE"; python diagnostics/transductive_gate.py` -> `cv_out/transductive_gate.txt`;
  then `python diagnostics/transductive_gate.py --shift kurt`; `--shards 2` only if the default passes.
  Check: base at seed 0 should reproduce nbm_sweep.csv bs24_ep25 nb_logodds 0.6316 (+-0.001).

## 36. Break transplant: synthetic breaks learned from real ones `sbrt/breakgen.py` + `diagnostics/breakgen_gate.py` (owner's idea 2026-10-20; RESULT: CLOSED, gate failed and the override cv.py run hit the early stop)

- **Idea (owner):** a matched simulator: learn how real breaks change a series and generate post-break data close to
  the real distribution. Implemented without generating histories (plenty of real break-free data; NBM data curve
  flat) and without sampling post-break points from a model (NBM-surprise features would learn the generator's
  fingerprint). Instead **transplant**: whiten a real break's post-break segment under its pre-break normal model
  (AR(10) + frozen GARCH variance path + ECDF map, as bocpd `wu`), summarise the change as a smooth quantile map
  (Hermite cubic: scale, skew, tails; no location, real breaks have no mean shifts) + AR(2) dependence, and apply it
  to another series' REAL break-free continuation through that series' own inverse normal model. Donors come from
  the parent's own CV fold (no label leak); geometry as synth.py, history lengths from the real distribution.
- **What the smoke runs taught (400-1,500 series):**
  1. A donor's measured change carries NOISE as large as most real changes: placebo donors (no-break series cut at
     a fake break) have sd(log scale) ~0.10 regardless of post length (no-break variance wander vs the frozen
     forecast). Transplanted as is, placebo "breaks" were MORE visible than real ones (oracle 0.604 vs 0.555 at
     L = 300). Fix (VERSION 3): **empirical-Bayes deconvolution** per component (noise = KDE of placebo values,
     true-change prior = NPMLE on the real donors, each transplant uses a posterior draw). Deconvolved placebo
     transplants are near-invisible (0.507 / 0.521 / 0.521): the control works.
  2. Real breaks are instantaneous: real-minus-placebo whitened log variance is flat over windows 0-10 / 10-30 /
     30-100 / 100-300 (+0.11 / +0.08 / +0.09 / +0.11); no ramp to model.
  3. A real-vs-synthetic fingerprint classifier is useless here: real no-break series vs re-cut no-break variants
     (no break anywhere) already give AUC 0.597 (vol_lvh1000, history-length related). That floor applies to the
     adopted re-cuts too. Replaced by **train-on-synthetic / test-on-real (TSTR)**.
- **Smoke numbers (NOT a result).** Calibration (1,500 parents), oracle average L = 30/100/300: real breaks in
  re-cut geometry (the rule-C reference) 0.504/0.521/0.539 (original series 0.515/0.532/0.555); deconv
  0.532/0.554/0.563, lam 0.4 0.523/0.536/0.551, old synth 0.523/0.537/0.558, deconvolved placebo
  0.507/0.521/0.521: every synthetic set is TOO VISIBLE, nothing passes C. TSTR (400 parents, real training set
  = all ~8,000 real series outside fold 0): real-only 0.644; real + deconv 0.649, real + old synth 0.646, real +
  placebo 0.644. The proxy does not reject old synth (known CV loser), so D is informational; deconv TSTR - old synth
  +0.008 +- 0.012. Reading so far: the transplant is not closer to the real breaks than the old hand-made types.
- **Decision rule** in the gate docstring (written before any full run, amended five times on smoke evidence only,
  thresholds unchanged): C = oracle distance < 0.0162 and |gap| <= 0.005 at each L vs real re-cuts; D = TSTR (mean >= old synth +
  0.01, >= TRTR - 0.02 at each L, real + synthetic >= TRTR). C and D pass -> wire `--bgen` into cv.py (same plumbing
  as `--synth`) and run the standing pass rule. Not wired into cv.py yet.
- **Commands** (CPU, ~20-40 min):
  `$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -m sbrt.breakgen` (self-test, ~1 min) then
  `python diagnostics/breakgen_gate.py` -> `cv_out/breakgen_gate.txt` (~40-70 min, CPU).
- **FULL GATE RESULT (owner's run 2026-10-20, 2,087 s, `cv_out/breakgen_gate.txt`, 3,000 parents):**
  - Rule C FAILS for every arm. Reference (real re-cuts) 0.511/0.527/0.543. Distances: old synth 0.0165 (closest),
    lam 0.4 0.0255, lam 0.25 0.0286, deconv 0.0821, lam 1 0.150. Every transplant arm is too visible on the
    variance statistics at L = 30 and grows too slowly with L: the same shape failure as the old synth.
  - Rule D (TSTR, test = real fold 0, real training set ~8,000 series): real only 0.644. Synthetic only: deconv
    0.627 (L = 300 0.686 vs real-only 0.681), old synth 0.601, lam 1 0.615, placebo deconv 0.554. Real + it:
    deconv 0.652, lam 1 0.647, old synth 0.6436, placebo 0.638 / 0.643. Proxy check passed only by 0.0005 (old synth
    0.6436 vs 0.6441). Deconv TSTR - old synth +0.0258 +- 0.0112 (2.3 SE), but deconv misses "TSTR >= TRTR - 0.02 at
    every L" at L = 30 (0.570 vs 0.577) and L = 100 (0.626 vs 0.634), so D FAILS too.
  - Reading: by the written rules, CLOSED. The deconvolved transplants carry more real break signal than the old
    synth (2.3 SE) and real + deconv is +0.008 on the proxy (no paired SE for that line), so an explicit owner
    override would be one paired cv.py trees run (needs `--bgen` wired into cv.py like `--synth`; early stop on
    folds 0 and 1). Not wired.
- **Owner's override (2026-10-20): one paired trees run.** `cv.py --bgen N` wired (same plumbing as `--synth`:
  the transplants are an extra training source, every feature family is computed for them, rows train only in
  folds where the parent trains; donors, placebo noise and the deconvolution come from the parent's own fold;
  needs `--folds 5`). Flags: `--bgen-mode deconv|lam`, `--bgen-lam`, `--bgen-min-post 100`, `--bgen-seed`,
  `--bgen-rel`. `--bgen 0` (default) leaves every path unchanged. Smoke (400 series, 2 folds): runs end to end.
  Command (seed 0, paired vs the reference trees; early stop if folds 0 and 1 are both <= 0, then CLOSED):
  `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --bgen 2 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_bgen2`
  If the 5-fold paired mean is > 0: seed 1 (`--seed 1`, compare `..._s1.npz`), then the standing pass rule.
- **RESULT of the override (owner's run 2026-10-20, trees seed 0, `--bgen 2`, 20,000 transplants, 10,107 with a
  break):** fold 0 0.6621 -> 0.6504 (**-0.0116**, best round 150), fold 1 0.6329 -> 0.6288 (**-0.0042**); folds 0
  and 1 both <= 0 -> **CLOSED by the early stop** (owner stopped it in fold 2). Worse than the old hand-made synth
  (-0.0022 over 5 folds, section 14 of the full history). Reading: the gate's calibration was right and the TSTR
  proxy misled; the transplants are too visible, the trees learn their easier breaks (fold 0 peaks at round 150)
  and rank real breaks worse. With two independent generators failing the same way (too clear early, too faint
  late vs real breaks), synthetic breaks on real backgrounds are a closed direction; real re-cuts stay the only
  augmentation that works.
- **Pre-check (full data, 1,781 real / 3,512 placebo donors, 300 post points):** change predictable from the history?
  R^2 (LightGBM, 8 descriptors) log variance 0.10 real vs 0.04 placebo, all other components <= 0; lag-1 autocorr.
  of |w| has signal share ~0 (breaks do not change vol clustering). So a model "history -> break shape" has
  little to condition on: the learned conditional generator (owner's arm 2) was not built.

---

## 37. Time-reversal augmentation `cv.py --aug-rev N` (assistant's "out of the box" idea, owner's go 2026-10-20; RESULT: CLOSED by the fingerprint gate)

- **Idea:** the series are time-reversible (backcasting = forecasting, irreversibility features no signal), and the
  model is data-limited. Reverse each REGIME on its own: no-break series = reverse(history ++ online); break series =
  reverse(pre-break block) ++ reverse(post-break block), cut at the original history length. Same geometry, break
  at the same position, exact labels, real values. Unlike more re-cuts (`--aug 4` only +0.0007), the new history
  ENDS with the original's first values: a new history-end state for the volatility-state features.
- **Code:** `sbrt/augment.py` `reverse()` / `make_reversed(n)` (shift-0 reversed copy + n - 1 re-cuts of it; own
  random stream); self-test extended (involution, exact labels by position tracking, ranges). `cv.py --aug-rev N`
  adds source "reversed" (same plumbing as re-cuts / `--bgen`: trains only with its parent's fold; cache stub
  `rev{N}v{REV_VERSION}_s{seed}_m{min_shift}`), `--aug-rev-rel` (default = `--aug-rel`). `--aug-rev 0` (default) changes nothing.
- **Gate (cheap, before CV):** `diagnostics/rev_fingerprint.py`: real vs reversed copy vs the re-cut floor on the
  production features, grouped by parent, no-break and break series separately. Rule (written before any run): GO if
  reversed <= re-cut + 0.02 at every k = 30 / 100 / 300 on both, and the real-vs-real control is 0.50 +- 0.03.
- **Commands:**
  1. self-test: `python -m sbrt.augment`
  2. gate smoke: `python diagnostics/rev_fingerprint.py --n-series 600 --out cv_out/rev_fingerprint_smoke.txt`
  3. gate full: `python diagnostics/rev_fingerprint.py` -> `cv_out/rev_fingerprint.txt`
  4. cv.py smoke (same flags as the full run, tiny): `python cv.py --n-series 400 --folds 2 --gpu --rounds 100 --threads 6 --aug 2 --aug-rev 1 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 2 --tag smoke_rev`
  5. paired trees, seed 0 (early stop on folds 0 and 1): the current best tree command plus `--aug-rev 1`, `--compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_rev1`
- **Expectation, written before any run:** +0.000 to +0.004 trees; the flat `--aug 4` curve is the main reason to doubt it.
- **Reading the gate:** the 0.60 re-cut floor is mostly history length (`vol_lvh1000`); shift-0 reversals have the
  same geometry, so their honest floor is the control (~0.50). A reversed AUC near the re-cut level would be a
  dynamics fingerprint (GARCH irreversibility in vol_ / bo_): read it against the control and its top features.
- **Not built:** adoption would also need `--aug-rev` in `gru_cv.py` and in `make_submission.py` (training only).
- **GATE RESULT (owner's full run 2026-10-20, `cv_out/rev_fingerprint.txt`): FAIL, CLOSED.** AUC real vs arm at
  k = 30 / 100 / 300: no-break control 0.495 / 0.511 / 0.518, re-cut 0.505 / 0.500 / 0.491, **reversed 0.647 / 0.651 /
  0.665**; break series re-cut 0.507 / 0.493 / 0.484, **reversed 0.673 / 0.670 / 0.664** (rev + re-cut the same).
  Top features of the reversed arm: `vol_lvh1000`, `vol_lvh200`, `ctx_skew`, `vol_persist`. So the series are NOT
  time-reversible in their volatility path and increment asymmetry (the earlier "time-reversible" reading held for
  the linear / residual predictability only). Side finding: in the same geometry, the re-cut fingerprint is ~0.50,
  so the 0.597 floor of section 36 came from geometry (history length), not from re-cutting. Not pursued: trimming
  the series start before reversing (the warm-up idea) cannot fix `ctx_skew`, an irreversibility of the dynamics.

---

## 38. Test-time views `cv.py --tta 51,150` (assistant's idea, owner's go 2026-10-20; RESULT: CLOSED 2026-10-21, bar out of reach after 3 folds)

- **Idea:** no new data. Score each series several times with the history/online boundary moved EARLIER by m points
  (the last m history points become pseudo-online; causal: view step t + m uses only points up to step t) and average
  the logits over the views. The trees already train on such re-cuts (`--aug 2`), so the views are in-distribution.
  Target: the no-break false alarms that come from WHERE the reference happens to end (history-end vol level, GARCH
  and NBM end state); a real break is in every view's online part. Seeds / extra members see identical inputs
  (blend +0.0002-0.0004); views change the inputs.
- **Code:** `sbrt/augment.py make_views(series, tau, shift)` (online cut to 999; histories that would fall below
  1,000 get no view; self-test extended). `cv.py --tta 51,150` adds one score-only source per shift (`view+51`, cache
  stub `view51v1`): features, NBM and hidden-state columns per fold like any extra source, but `sel` is empty, so
  they NEVER train. After each fold, `tta_fold` scores the validation series with the SAME fold model at its best
  round, plain and averaged over the views that exist for each (series, step), with the plain score weighted 1 and
  2 (`TTA_W0`). Every logit is one predict call over rounds 0..k, so XGBoost's per-call base score is one shared
  offset. Report: per-fold and 5-fold paired difference vs plain, by step bucket; files `cv_out/oof<tag>_tta.npz`
  and `..._tta_w2.npz` (same layout as OOF files, for blend.py). Shifts must be multiples of 3 (rows every 3rd step).
  Trees only (lgb / xgb binary; refuses svm, rank, prune, base-margin, xsec, pc, emb, femb, eqf, lratio).
  `--tta ""` (default) changes nothing.
- **Why the test is clean:** plain and views share one model per fold, so the paired difference has NO seed
  noise. One run of the current best tree command answers it; its plain OOF should reproduce the reference
  (the `--compare` line is ~0).
- **Rule (written before any run):** worth a production build if the trees' paired TTA gain is >= +0.0010, >= 4/5
  folds, > 2 SE, then the standing blend rule with the `_tta` trees file as `--a` (GRU / SVM views would be a
  second step). Inference cost x (1 + views) on the trees' features (cloud ~11 min today: inside the 15 h budget).
- **Commands:** smoke `python -m sbrt.augment`; then
  `python cv.py --n-series 400 --folds 2 --gpu --rounds 100 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 2 --tta 51,150 --tag smoke_tta`;
  full: the current best tree command plus `--tta 51,150 --tag aug2_df_bo_nbm_bs24_ep25_tta`.
- **Expectation, written before any run:** +0.000 to +0.003 on the trees; views share most of their data.
- **Choice fixed before the run:** the blend uses the plain-weight variant (x1 or x2) with the larger 5-fold trees
  mean, both seeds with views (seed 1: `--seed 1 --compare ..._s1.npz --tag ..._tta_s1`); one blend look only.
  Early stop: both variants <= 0 in folds 0 and 1.
- **RESULT seed 0 (owner's run 2026-10-21, folds 0-2 shown):** plain OOF reproduces the reference exactly (paired
  +0.0000 in every fold: sanity check passed). TTA vs plain, same fold model (no seed noise):
  | Fold | round | plain | x1 + views | x2 + views |
  |---|---|---|---|---|
  | 0 | 200 | 0.6621 | -0.0045 | -0.0026 |
  | 1 | 650 | 0.6329 | -0.0003 | +0.0006 |
  | 2 | 150 | 0.6450 | -0.0009 | +0.0004 |
  x1 failed the early stop (folds 0 and 1 both <= 0); x2 survived it on fold 1's +0.0006. Mean after 3 folds: x1
  -0.0019, x2 -0.0005. **CLOSED:** reaching the +0.0010 5-fold mean would need folds 3-4 to average +0.0033 for x2,
  5x the best fold so far, and these differences carry no seed noise to bridge it. Reading: the views are
  in-distribution for the trees (re-cuts) but shorten the effective online part, so a real break is diluted by
  views in which it sits closer to the start (view+150 has no positives before step 100); the history-end false
  alarms they were meant to average out are not the dominant error. Inference cost would have been x3 on trees.

---

## 39. Query-attention transformer `gru_cv.py --arch qattn` (owner's idea 2026-10-21; RESULT: CLOSED; folds 0-2 wavg -0.0136 vs the GRU, warmup retry -0.0075 on folds 0-1)

**What (owner's design):** a transformer in place of the GRU recurrence. Learned query tokens ask "did a break
happen?" and attend over everything the model has: the raw series channels and every statistical feature.
`QAttn` in `gru_cv.py`, with the same inputs, loss, weights, folds and output layout as the GRU (61 channels):
1. **Feature tokens:** each channel group becomes its own token at every step: raw z / residuals, history
   descriptors, time, vol, BOCPD, NBM, NBM hidden PCA, and the 20 kernel features in 4 chunks of 5
   (`--qa-group`, 1 = one token per feature). With today's flags that is 11 tokens per step.
2. **Feature queries:** `--qa-fq 2` learned queries per step attend over that step's tokens ("which statistics
   matter now").
3. **Causal transformer over the steps:** `--qa-layers 2`, width `--qa-d 64`, 4 heads with ALiBi recency slopes
   1/8, 1/32, 1/128 and 0. The heads act like ~8 / 32 / 128-step windows plus one over the whole stream. Without
   a recency bias, causal attention tends to a running mean, which is what the closed TCN did.
4. **Break queries:** `--qa-queries 4` learned queries, each conditioned on the current state, attend over all
   steps <= t. The state plus the 4 answers go to the usual head. 139k parameters (the GRU has ~33k).

**Causality:** strict masks, LayerNorm per token only. At model creation in fold 0 a built-in check replaces
all inputs after step t and requires the earlier outputs to stay the same. Smoke: max change 0.00e+00, OK.
The run exits if the check fails.

**What it does NOT see:** the whole raw history. `G["X"]` already holds 8 GB on the GPU (100 burn steps +
online), so a longer `--burn` does not fit. The full history reaches the model only through the
ctx / vol / BOCPD / NBM channels. Follow-up, only if v1 shows anything: 32-64 history-summary tokens per series in
their own tensor, which every online query may see (no leakage: the history is complete before step 0).

**Smoke (2026-10-21):** `--n-series 300 --folds 1 --epochs 2 ... --nbm-hpca 8 --arch qattn` trains end to end,
3.1 GB peak GPU (full data: expect ~8 GB inputs + ~3 GB; if it crawls or OOMs, add `--batch 32`). GRU path
re-run on the same smoke: runs, unchanged code path (the new modules are created only for qattn). Outputs
deleted; `gru_cv.py` still CRLF.

**Expectation (written before any run):** small. Strong prior from the same data: TCN = one GRU seed, honest blend
-0.0002 (0/5); bigger GRUs and hpca 16/32 overfit (data-limited). Most likely: alone about one GRU seed or a bit
below, blend about +0.000 to +0.0005. It could gain if the break queries pick up multi-scale
"recent vs earlier" contrasts the GRU state blurs.

**Decision rule (written before any run):**
1. Seed 0, all 5 folds, `--swa 5-8 --fixed-ep 7`; its `wavg` line vs the GRU s0 `_wavg` file (printed as
   "vs --compare"). No two-fold early stop (sequence-net folds vary ~+-0.01).
2. Whatever the alone result: `blend.py --honest-paired` with qattn as a 4th member (`--d`) vs the 0.6527
   reference. Also judge it in the GRU slot once seed 1 exists (only if step 1 is >= 0).
3. ADOPT at >= +0.0005, >= 4/5 folds, > 2 SE (section 5). Production would need a numpy causal transformer with a
   KV cache (O(t) per step, about 1,100 x 64 x layers per point): slower than the GRU, inside the budget.

**Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
1. `python gru_cv.py --folds 5 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 5-8 --fixed-ep 7 --arch qattn --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_qa`
2. `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --d cv_out/oof_qattn_aug2_vol_bo_nbm_hpca8_swa_qa_wavg.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz --honest-paired`
3. Seed 1 (only if step 1 >= 0): command 1 plus `--seed 1`, `--compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz`, `--tag aug2_vol_bo_nbm_hpca8_swa_qa_s1`.

---

## 40. Sign-reflection augmentation, fingerprint gate `diagnostics/rev_fingerprint.py --mode flip` (teammate research G08, owner's go 2026-10-21; RESULT: CLOSED by the fingerprint gate)

**Source:** teammate's `TEAM_GAP_RESEARCH_2026-10-21.md` (written without access to this repo). Its eight G-cards were
checked against this repo on 2026-10-21: G07 (history-conditioned blend gate) fails its own gate (stacking WITH history
context was tried, section 8 / [§23]: memorises series); G01 (SSA reconstruction residuals) and G03 (stochastic-volatility
particle filter) overlap heavily with the AR(10) residual / variogram / closed Rio-CPD and with the NBM (a learned
Student-t volatility filter) / closed MS and GARCH twins; G04 (conditional quantiles) is mostly the NBM's ECDF-whitened
error, except that `disttest.py` never runs on the NBM stream (cheap variant, not built); G05 / G06 low prior (ordinal,
irreversibility, long-range, MiniRocket closed); G02 (NBM loss gradients) open, moderate build. The report's code
(`bank004f.py`, `bank004c2.py`) is not in this repo. G08 picked first: cheapest real gate.

- **Idea:** train on the sign-reflected copy (-history, -online) too. Negation is invertible, keeps every change in
  scale, tails and dependence, keeps the history exactly z-scored and keeps the ORDER of the volatility path (what
  failed reversal, section 37). The model is data-limited (section 3: doubling data ~+0.015; re-cuts +0.009).
  What it changes: every signed odd-moment statistic (skew, z^3 terms, mean drift, leverage).
- **Code:** `sbrt/augment.py` `flip()` / `make_flipped(n)` (shift-0 flipped copy + n - 1 re-cuts of it, own random
  stream `[seed, i, 23]`), `FLIP_VERSION = 1`; self-test extended (involution, exact labels, ranges): passes.
  `diagnostics/rev_fingerprint.py --mode flip` runs the section 37 gate with arms control / re-cut / flipped /
  flip+re-cut (same features: 270 kernel + 13 vol + 10 BOCPD, geometry columns dropped; no NBM). `--mode rev` (default)
  is the old gate unchanged. `cv.py --aug-flip` is NOT built: only after a GO.
- **Gate rule (written before any run, same thresholds as section 37):** GO to a paired `cv.py --aug-flip` trees run
  if, on the no-break AND the break series, flipped <= re-cut + 0.02 at every k = 30 / 100 / 300, and the control is
  0.50 +- 0.03. Otherwise CLOSE and record the top features of the flipped arm. An owner override would still be the
  paired cv.py run (which would then need `--aug-flip` built first).
- **Expectation (written before the full run):** likely FAIL. The 300-series smoke (meaningless for the verdict: its
  control was 0.39-0.47, too few rows) already put `ctx_skew` and the z^3 columns at the top of the flipped arm,
  i.e. the series look systematically skewed one way. If the full gate passes: trees +0.000 to +0.004 (prior from
  re-cuts, and the flat `--aug 4` curve).
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; CPU only):**
  1. self-test (seconds): `python -m sbrt.augment`
  2. gate smoke (~1 min): `python diagnostics/rev_fingerprint.py --mode flip --n-series 600 --out cv_out/flip_fingerprint_smoke.txt`
  3. gate full (~10-25 min, 12 threads): `python diagnostics/rev_fingerprint.py --mode flip` -> `cv_out/flip_fingerprint.txt`
- **GATE RESULT (owner's full run 2026-10-21, `cv_out/flip_fingerprint.txt`): FAIL, CLOSED.** AUC real vs arm at
  k = 30 / 100 / 300: no-break control 0.495 / 0.511 / 0.518, re-cut 0.505 / 0.500 / 0.491, **flipped 0.679 / 0.677 /
  0.677** (flip+re-cut 0.674 / 0.673 / 0.680); break series re-cut 0.507 / 0.493 / 0.484, **flipped 0.650 / 0.649 /
  0.658** (control 0.494 / 0.501 / 0.450; the 0.450 at k = 300 has only 868 rows per class, irrelevant next to a
  +0.14 to +0.19 gap). Top features of the flipped arm: `ctx_skew` (0.12-0.15), `vol_persist`, `vol_lvh1000`,
  `ctx_kurt`, z^3 columns. The vol_ columns are exactly sign-invariant (checked: max |real - flipped| = 0.0 on 8
  series), so they enter only through the JOINT with skew: the history skew has a preferred sign that depends on
  the volatility dynamics (leverage-like), and a flipped copy breaks that relation. Not pursued: dropping or
  abs()-ing the skew columns would delete real information and the interaction would still leak through z^3 terms.
  Like reversal (section 37), the series are not symmetric under this group; data-limited remains unsolved by
  label-preserving transforms other than re-cuts.

**RESULT seed 0 (owner's run 2026-10-21, stopped in fold 3; no OOF file saved, so no blend look):**
| Fold | qattn wavg | GRU s0 wavg | Paired | qattn best epoch |
|---|---|---|---|---|
| 0 | 0.6426 | 0.6578 | -0.0152 | 0.6441 (ep 8) |
| 1 | 0.6243 | 0.6395 | -0.0152 | 0.6228 (ep 7) |
| 2 | 0.6269 | 0.6374 | -0.0105 | 0.6296 (ep 6) |
Mean -0.0136, 0/3: well past GRU seed noise. **Diagnosis: it underfits, unlike the bigger GRUs / hpca 16-32, which
overfit.** Train loss at epoch 8 is 0.644-0.646 vs the GRU's 0.617-0.621, epoch 1 0.671-0.673 vs 0.654-0.655, and
validation jumps around (fold 0: 0.6402 -> 0.6254 -> 0.6441). It is not learning the training data as well, so this run
does not test the idea fairly. Likely causes: no warmup, lr 2e-3 is high for attention (GRU-tuned), and 8 epochs is
short for a 139k-parameter transformer. ~45 s/epoch (GRU ~1.5 s).
**Fix BUILT 2026-10-21 (`--warmup N`, linear over the first N optimiser steps on top of the epoch cosine; 0 = off, as before; smoke OK), not run:** warmup 375 steps (= 1 epoch) + lr 5e-4, folds 0-1 only. Kill test: train loss reaches the GRU's ~0.62 AND
the paired wavg is > -0.005 in both folds, otherwise CLOSED.
Command: `python gru_cv.py --folds 2 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 5-8 --fixed-ep 7 --arch qattn --lr 5e-4 --warmup 375 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_qa_lr5w`

**RESULT of the warmup retry (owner's run 2026-10-21, `cv_out/qattn_summary_aug2_vol_bo_nbm_hpca8_swa_qa_lr5w.txt`): FAILS the kill test, CLOSED.**
| Fold | qattn wavg | GRU s0 wavg | Paired | train loss ep 8 | ep 1 |
|---|---|---|---|---|---|
| 0 | 0.6514 | 0.6578 | -0.0064 | 0.6351 | 0.6750 |
| 1 | 0.6309 | 0.6395 | -0.0086 | 0.6314 | 0.6716 |
Mean wavg -0.0075 +- 0.0011 (0/2); best-epoch line -0.0071, lavg -0.0069. Both parts of the rule fail: train loss
0.631-0.635 (GRU ~0.62) and paired < -0.005 in both folds. The warmup and lower lr HALVED the gap (-0.0152 -> -0.0075
per fold) and the model still improves at epoch 8 (fold 0: 0.6443 -> 0.6511), so it remains slower to fit than the
GRU, not a different signal: wavg - ep7 +0.0048 (2/2), as for the GRU. Step buckets 0.575 / 0.638 / 0.650 / 0.676.
Not pursued without an owner override: more epochs (~45-60 s/epoch, 5 folds x 2 seeds of a member that at best
matches one GRU seed; TCN, the other architecture swap, gave honest blend -0.0002, section 17).

**Owner override 2026-10-21: longer schedule, "does it fit better than the GRU given the same data?"** Both runs
above stopped at epoch 8 of a 25-epoch cosine (`--swa 5-8` ends training at max(B, --fixed-ep)), i.e. with the
learning rate still ~85% of its peak: the transformer never got the low-lr end of the schedule. Retry: 20 epochs,
cosine over 20 (fully annealed), weights averaged over epochs 13-20, warmup 375, lr 5e-4, folds 0-1. Not
schedule-matched to the GRU reference (epochs 5-8 of 25): this gives qattn the more favourable schedule, which is
the point of the question. ~40-60 s/epoch -> about 35-40 min.
Rule (written before the run), wavg vs GRU s0 wavg, folds 0 and 1:
  - both > -0.002 (matches the GRU within seed noise, or better): run all 5 folds with the same flags, then
    `blend.py --honest-paired` with qattn as a 4th member (`--d`) and in the GRU slot; standing adoption bar.
  - otherwise CLOSED for good (no further schedule tries).
  Also read: train loss at epoch 20 vs the GRU's ~0.62 (does it fit the training data at all as well?).
Command: `python gru_cv.py --folds 2 --epochs 20 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 13-20 --fixed-ep 20 --arch qattn --lr 5e-4 --warmup 375 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_qa_lr5w_e20`

---

## 41. Distribution tests on the NBM's whitened stream `cv.py --nbm-dist` (teammate research G04-lite, owner's go 2026-10-21; RESULT: CLOSED by the early stop)

- **Idea:** the NBM turns each online point into w_t = Phi^-1(ECDF_hist(v_t)), v_t = the NBM's standardised one-step
  error: a CONDITIONAL PIT given the whole past. Until now w only feeds a Gaussian BOCPD (`nb_` columns), which sees
  its mean and variance. A change in the SHAPE of the conditional distribution (tails, asymmetry, quantile coverage)
  with unchanged mean and variance is invisible to it, and `--dist` tests shape only on the MARGINAL e and u. This is
  the cheap remainder of the teammate's G04 card (a separate quantile head would duplicate the NBM's density).
- **Code:** `sbrt/nbdist.py` (8 columns `nbd_ks, nbd_ad, nbd_cvm, nbd_w1, nbd_ks100, nbd_ad100, nbd_ksscan,
  nbd_adscan`; the `disttest.py` statistics via `disttest._seg`). Reference = the history's own whitened values,
  exactly Phi^-1((k + 0.5) / (m + 1)), m = len(hist) - AR - 51 (checked against `NbmStream.ref`). Input = the NBM
  sums' first differences (`nbm.w_of`), so NO extra GRU pass; production would push C1[n] - C1[n-1] from the
  `NbmStream` (bit-identical to the batch). `cv.py --nbm-dist` (needs `--nbm`): placeholder columns after the
  `nb_` / pshape columns, filled per fold inside `nbm_fold`, cached as `data/nbd1_f{f}_{cfg}_{stub}.npz`. Trees
  only for now (SVM / GRU inputs would be a second step).
- **Checks done (2026-10-21):** `python -m sbrt.nbdist`: streaming == batch (all rows and every 3rd), causal,
  deterministic, reference = the NBM's in-history whitened values; shape-only change (unit-variance t3 vs Gaussian,
  600 points): AD 0.73 -> 10.3, KS 0.59 -> 2.08. cv.py smoke (300 series, 2 folds): runs, 8 columns appended and
  used; flag OFF gives an OOF bit-identical to the pre-patch cv.py (NaN-aware). CRLF kept.
- **First full run cost:** the fold caches for `nb_` exist but the w stream is not cached, so each fold re-runs the
  NBM pass over originals + re-cuts (weights loaded from the fold cache, no retraining; the `nb_` caches are
  rewritten with identical values), then the trees. Estimate ~1-1.5 h per seed (an estimate: let the run decide).
- **Expectation (written before any run):** trees +0.000 to +0.002. For: `--dist` on e/u gave trees +0.0011, and a
  conditional test is a different question. Against: nb_lvr / nb_logodds already carry most of what w holds; the
  tail oracle (section 33) found little shape-only signal; the variance-matched slice has resisted everything.
- **Rule (section 5 standing rule, written before any run):** 1. paired trees seed 0 vs the reference trees;
  early stop if folds 0 AND 1 are both <= 0 -> CLOSE. 2. Seed 1, compare the seed pair with
  `diagnostics/oof_paired.py`. 3. If the 5-fold pair mean > 0: blend with the new trees in the trees slot vs the
  0.6527 reference; ADOPT at >= +0.0005, >= 4/5 folds, > 2 SE.
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
  1. self-test (seconds): `python -m sbrt.nbdist`
  2. trees seed 0: `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-dist --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_nbd`
  3. trees seed 1 (only if 2 passes the early stop): command 2 plus `--seed 1`, `--compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`, `--tag aug2_df_bo_nbm_bs24_ep25_nbd_s1`
  4. seed pair: `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nbd.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nbd_s1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
  5. blend (only if 4 > 0): `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nbd.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nbd_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`
- **RESULT trees seed 0 (owner's run 2026-10-21, stopped in fold 2): CLOSED by the early stop.** Paired vs the
  reference trees: fold 0 0.6621 -> 0.6598 (-0.0023, best round 250), fold 1 0.6329 -> 0.6327 (-0.0002, round 450).
  Both <= 0. The 8 columns are a real shape detector (self-test AD 0.73 -> 10.3 on a pure shape change) but add nothing
  the trees do not already get from nb_ / dist_e / dist_u: like the tail families (section 33), shape-only change of
  the conditional distribution is not where the missing breaks are. NBM-stream cost per fold ~330-370 s (no retrain).
  Caches `data/nbd1_f0_*`, `f1_*`, `f2_*` (~490 MB) can be deleted; the flag and module stay (off = bit-identical).

---

## 42. Joint NBM fine-tuning through the GRU detector `gru_cv.py --nbm-joint 0.1` (assistant's idea A of the 2026-10-21 "new ideas" list, owner's go 2026-10-21; RESULT: CLOSED by its gate 2026-10-21)

- **Idea:** today the NBM is pretrained self-supervised (next AR residual, no labels) and then frozen; the GRU reads
  8 PCA channels of its hidden state (section 24, the largest blend step since the volatility state). More channels
  of the same frozen state overfit (hpca 16 / 32, section 30): the extra dimensions are unsupervised. Here the
  detector's label loss also trains the NBM, so the state learns to show what separates breaks, not only what predicts
  the next point. The NBM's own Student-t NLL stays as an anchor loss so it remains a normal-behaviour model.
- **Why this is not a repeat:** per-series NBM fine-tuning ([§40], no labels, one series) and transductive adaptation
  on the test stream (section 35) adapted to single series; hpca 16 / 32 added unsupervised channels. Nothing so far
  used the training labels to shape the NBM.
- **What the flag does (`--nbm-joint M`, needs `--nbm-hpca`; 0 = off):**
  - The fold's pretrained NBM (`data/nbm_v1_cv_f{f}_e25_...pt`, the same file the frozen run uses) becomes a
    trainable submodule `model.nbmj`. Its learning rate is M x `--lr` (0.1 -> 2e-4), with weight decay `--nbm-wd`, in
    its own AdamW group on the same cosine.
  - Every batch (train and validation) the 8 hidden-state channels are recomputed from the CURRENT NBM weights and
    replace the cached frozen ones. The PCA basis (mean, components, sd) is the fold's frozen basis, fixed.
  - Truncated backprop: the history except its last `--burn` (100) rows runs without gradient (packed sequence ->
    state h0); the window of the last 100 history rows + the online part runs with gradient from h0.
  - Anchor: `--nbm-joint-nll` (default 0.2) x the NBM's Student-t NLL of the next residual on the window's
    PRE-BREAK rows (history rows, no-break online rows, break series before tau). The printed train loss is the
    detector loss alone; the epoch line adds `NBM NLL`.
  - The 6 `nb_` channels (BOCPD on the NBM stream) stay on the FROZEN NBM, and so do the trees and the SVM.
  - Epoch averaging (`--swa`) averages the NBM's weights with the rest of the model.
  - Input rows for the NBM (z, AR residual e, 6 descriptors; same arithmetic as `sbrt/nbm.build_seq`) are built once
    and cached fold-independently: `data/gru_nbmseq_v1_aug2_s0_m20.npz` (~0.8 GB, ~1-3 min the first time).
  - New log lines per fold: the start check (torch channels vs the cached numba channels) and, at the end, how far
    the NBM weights moved (relative L2).
- **Checks (2026-10-21, 300-series smokes, 1 fold, GPU):**
  - flag OFF: OOF files (best, wavg, lavg, ep7) bit-identical to the unpatched script (max |diff| 0.0).
  - flag ON (`--nbm-joint 0.1`): runs end to end. Start check max |diff| 8.9e-3 on the standardised channels
    (float32 / TF32 torch vs float64 numba over thousands of steps; an off-by-one alignment would give O(0.1-1)).
    NBM NLL flat at 1.40-1.41 over 8 epochs; NBM weights moved 3.2%; ~2.3 s per epoch for 720 training series
    (frozen run ~0.15 s). Deterministic: repeat run gave the same scores. Smoke numbers say nothing about the effect.
  - Pairing fixes after review (2026-10-21): (a) the NBM submodule is built inside `torch.random.fork_rng`, so
    later folds' detector init is unchanged: 2-fold smoke, fold 1 epoch-1 train loss 0.6849 off and on (before the
    fix fold 1 started from a different init, i.e. effectively another seed); (b) the NBM batch length is rounded up
    to a multiple of 512 (the Windows VRAM-spill pattern of section 4; history lengths vary 1k-5k within a batch);
    (c) start check sign-aligns the PCA components to the cached channels (the full run recomputes the basis, since
    the frozen channels come from the cache; an SVD sign flip would otherwise feed the GRU sign-flipped channels).
    2-fold smoke: off bit-identical to the unpatched script on all four OOF files; on: 0 flips, max |diff| 8.9e-3 /
    9.9e-3, NBM moved 3.2% / 3.6%.
  - NOT verified: full-size time and GPU memory, and the cache-hit path of the start check (smokes never use caches).
  - CRLF kept.
- **Stop signs while the gate runs:** the start-check line should say max |diff| ~1e-2 (after any sign flips); above
  ~0.1 means the basis or alignment is wrong: stop. Epoch 1 far slower than ~2 min means VRAM spill: stop, check
  the sysmem-fallback setting (section 4).
- **Cost:** about +75 s per epoch at full size (scaled from the smoke: an estimate), so ~10-12 min per fold for the
  8 epochs of `--swa 5-8`, ~25-35 min for folds 0-1. Inference (if adopted): a second `NbmStream` pass per series with
  the fine-tuned weights (hidden state only, no posterior), the numba port unchanged; about one more NBM pass of
  inference time, well inside the 15 h budget. Production build NOT done.
- **Expectation (written before any run):** GRU member +0.000 to +0.006 per fold, blend +0.000 to +0.003. For: the
  NBM state is the one family still paying, and this is the supervised way to get more from it without more channels.
  Against: hpca 16 / 32 showed the GRU overfits easily from this state, and a trainable 30k-parameter NBM is more
  capacity of the same kind; the GRU's seed noise is ~+-0.01 per fold, so two folds can only kill, not confirm.
- **Rule (written before any run):**
  1. Gate: GRU seed 0, folds 0-1, `--nbm-joint 0.1`, `wavg` paired vs the `_wavg` reference (same seed, same 8
     epochs). **Both folds <= 0 -> CLOSED** (one variant allowed only if the log shows the NBM barely moved, < 0.5%,
     or its NLL rose by > 0.05: then `--nbm-joint 0.3` resp. `--nbm-joint-nll 1.0` once, same rule).
  2. Otherwise all 5 folds seed 0, then seed 1 (`--seed 1`, compare to the `_s1_wavg` reference).
  3. Blend with the joint GRU pair in the GRU slot vs the 0.6527 reference: ADOPT at >= +0.0005, >= 4/5 folds,
     > 2 SE (section 5).
  Also read: train loss vs the reference run (overfitting shows as lower train loss with worse validation, as
  hpca 16 / 32 did), and the step buckets.
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
  1. Gate, folds 0-1 (~30-40 min including the one-time input build):
     `python gru_cv.py --folds 2 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --nbm-joint 0.1 --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_jt01`
     Read the `fold k: epochs 5-8 averaged` lines and the final `wavg ... vs --compare` line in
     `cv_out/gru_summary_aug2_vol_bo_nbm_hpca8_swa_jt01.txt`.
  2. Only if the gate passes: command 1 with `--folds 5` (same tag; it reruns folds 0-1, ~1-1.5 h in total), then
     seed 1: command 1 with `--folds 5 --seed 1 --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_jt01_s1`.
  3. Blend: `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_jt01_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_jt01_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`
- **RESULT of the gate (owner's run 2026-10-21, `cv_out/gru_summary_aug2_vol_bo_nbm_hpca8_swa_jt01.txt`): CLOSED.**

  | Fold | joint wavg | reference wavg | Paired | joint ep7 / ref ep7 | train loss ep 8 joint / ref | NBM moved |
  |---|---|---|---|---|---|---|
  | 0 | 0.6524 | 0.6578 | -0.0054 | 0.6461 / 0.6473 | 0.6180 / 0.6194 | 12.8% |
  | 1 | 0.6379 | 0.6395 | -0.0016 | 0.6365 / 0.6350 | 0.6177 / 0.6170 | 13.0% |

  Mean wavg -0.0035 +- 0.0019, 0/2: both folds <= 0, so CLOSED by the pre-written rule. No variant is allowed: the
  NBM moved 13% (well above the 0.5% "barely moved" trigger) and its NLL did not rise (1.377 flat in every epoch,
  vs a 0.05 trigger). Start checks passed (0 sign flips, max |diff| 9.2e-3 / 2.6e-3: the cache-hit path works).
  Cost as estimated: ~60-70 s per epoch, peak GPU 9.4 GB, NBM input rows built in 18 s.
- **Reading:** the label gradient moved the NBM a lot, in directions that leave its forecasts unchanged (flat NLL),
  and the detector fits the training data exactly as well as with the frozen NBM (train loss within +-0.0014). So
  the supervised state is not more informative to the GRU; validation is a little worse, and in both folds. This
  agrees with sections 30 and 39: what the frozen 8-PC state carries the GRU already uses; reshaping it does not help.
  Not tried (no mechanism to expect a different sign): larger NBM learning rates, a trainable PCA projection.
- **Cleanup:** `data/gru_nbmseq_v1_aug2_s0_m20.npz` (~0.8 GB) can be deleted. The flag stays (off = bit-identical).

---

## 43. Score-test features from the NBM `cv.py --nbm-score` (assistant's idea B of the 2026-10-21 "new ideas" list = teammate research card G02, owner's go 2026-10-21; RESULT: BUILT, not run)

- **Idea:** the NBM reduces each online point to ONE number, the standardised error v_t, and the nb_ / nbd_ features
  ask whether its mean, variance or shape moved. A break can instead move the NBM's prediction in a direction that
  depends on the state (e.g. the scale is too small only in some region of the state), which averages out in v_t.
  A score (Lagrange-multiplier) test asks exactly this: is the gradient of the model's log-likelihood, which has
  mean zero while the model is right, drifting away from zero in some direction?
- **Why it is not a repeat:** sections 41 (`--nbm-dist`) and the nb_ BOCPD test the scalar stream only; hpca (24 / 27)
  gives the state itself, not the state x error interaction. Section 42 closed the supervised route to the state;
  this is the unsupervised, test-statistic route.
- **What it computes (`sbrt/nbscore.py`, 9 columns, prefix nsc_):**
  - Per point, the Student-t scores of the prediction that the fold NBM made for it (z = (e - m) / s):
    location (nu + 1) z / (nu + z^2), log scale (nu + 1) z^2 / (nu + z^2) - 1, tail shape -d nll / d nu; and
    location x p_k, log scale x p_k for the 8 hidden-state PCs p of the state that made the prediction (basis fitted
    per fold like `--nbm-hpca`). 19 dimensions.
  - Each series' HISTORY scores (rows >= 51, as nbm.py's reference) give a mean and covariance (+ 1e-3 I); online
    scores are whitened with them, then summed.
  - Columns: signed CUSUM / sqrt(n) of location, scale and tail shape; LM statistic ||sum||^2 / n for the 3 base
    directions and the 16 state directions, over all online points, the last 100, and a scan over start times
    (segment >= 20; computed on kept rows only, as nbdist).
  - One more NBM GRU pass per series (no posterior); numba float64, one step function for batch and stream.
- **Checks (2026-10-21):**
  - `python -m sbrt.nbscore` (21 s, gate NBM `data/nbm_v1.pt`): streaming == batch (all rows and every 3rd),
    causal, finite, deterministic on 12 held-out series; the GRU state after the first online point equals
    `nbm.run_hidden` (so the scores use exactly the state the hpca channels use).
  - Null check, 167,774 online points of held-out no-break series: whitened score means -0.03..+0.01 (calibrated);
    variances 1.00 / 1.01 for location / scale, 0.14 for tail shape (nearly collinear with the scale score: its
    residual is dominated by a few heavy-tailed history points), 1.1-2.0 for the 16 state directions (the online
    state wanders more than in the history). Ranks across series are what the trees use, so not fixed; noted.
  - `cv.py` smoke (300 series, 2 folds, 300 rounds, GPU): flag OFF gives an OOF bit-identical to the unpatched
    script; flag ON runs, 9 columns appended (330 in total), all 9 used by the trees; ~9 s per 900 series on 6
    threads -> ~5-6 min per fold at full size (an estimate), then cached (`data/nbsc1_f{f}_e25_..._aug2_s0_m20.npz`).
  - CRLF kept in `cv.py` and `sbrt/nbscore.py`.
  - Caveat (shared with every nb_ column): for TRAINING series the NBM saw the history, so their history scores are
    in-sample; validation series' are not. The whitening reference is therefore a little tighter for training rows.
- **Expectation (written before any run):** trees +0.000 to +0.001. For: a genuinely new direction (state x error),
  cheap, the NBM family is the one still paying. Against: the trees already have nb_lvr / nb_mu (the base scores'
  information) and the hpca columns, and they did not use hpca (section 27, trees closed); section 41's shape tests
  on the same stream added nothing; section 42 showed the state carries nothing more the GRU can use.
- **Rule (section 5 standing rule, written before any run):** 1. paired trees seed 0 vs the reference trees; early
  stop if folds 0 AND 1 are both <= 0 -> CLOSE. 2. Seed 1, compare the seed pair with `diagnostics/oof_paired.py`.
  3. If the 5-fold pair mean > 0: blend with the new trees in the trees slot vs the 0.6527 reference; ADOPT at
  >= +0.0005, >= 4/5 folds, > 2 SE. SVM / GRU inputs only as a second step after an adoption.
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
  1. self-test (~20 s): `python -m sbrt.nbscore`
  2. optional look (~5-10 min, CPU, information only, not a gate): `python -m sbrt.nbscore test` (single-feature
     TS-AUC of the 9 columns on held-out series 5000-6999 with the gate NBM)
  3. trees seed 0: `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 1200 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-score --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_nsc`
     Read each `fold k: ... paired difference` line; stop after fold 1 if both are <= 0.
  4. trees seed 1 (only if 3 passes the early stop): command 3 plus `--seed 1`, `--compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`, `--tag aug2_df_bo_nbm_bs24_ep25_nsc_s1`
  5. seed pair: `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nsc.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nsc_s1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
  6. blend (only if 5 > 0): `python blend.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nsc.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_nsc_s1.npz --b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --c cv_out/oof_svm_nbm_hpca8.npz --ref-a cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz --ref-b cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz --ref-c cv_out/oof_svm_nbm_hpca8.npz`

- **RESULT trees seed 0 (2026-10-22):** per fold **-0.0043 / +0.0022 / +0.0005 / -0.0004 / +0.0015**, 5-fold
  **-0.0001 +- 0.0011 (SE), 3/5**; averaged curve best 0.6420 at 200 rounds (reference 0.6417 at 150). Not stopped
  early (fold 1 > 0). Importance: `nsc_lm_base_scan` rank 11 (2.5% of gain), while `nb_logodds` fell from ~3.3% to
  1.0%: the trees swap the new column in for the NBM's own BOCPD column (the section 9 substitution pattern). Run
  time ~1 min per fold with the columns cached. Next by the rule: seed 1, then the seed pair (`oof_paired.py`);
  blend only if the pair mean > 0.

## 44. GRU distillation from the blend `gru_cv.py --distill` (owner's go 2026-10-22, option 1 of "boosting methods for the blend"; RESULT: CLOSED 2026-10-22, the clean teacher gives nothing: the leaky gain was the leak)

- **Idea:** train the GRU on soft targets, (1 - a) x label + a x p, where p is the 0.6527 blend's score. Why it
  might help: the labels are noisy in a way no model can fix (about 4.4% of breaks change nothing measurable, and
  the first points after tau look like no-break) and we are data-limited (section 3). Ensemble soft targets are the
  standard way to get more out of each row. The other blend-level methods (stacking, per-step weights, residual
  boosting from nb_logodds, 4th members, rank:pairwise) are closed (section 8), so this is boosting across members:
  the GRU learns from what the trees + SVM know.
- **Build:**
  - `distill_teacher.py` combines the reference OOF files exactly like `blend.py` (each file / sd, groups summed,
    group / sd, 0.4 / 0.4 / 0.2) -> `cv_out/teacher_blend.npz` (z per sid/step row; reproduces 0.6527).
    `--constant` writes z = 0 -> `cv_out/teacher_const.npz`, the label-smoothing control.
  - `gru_cv.py --distill a [--distill-file F]`: per fold, Platt calibration of z on that fold's ORIGINAL training rows
    (features z, z x log1p(step)/7, log1p(step)/7; class-balanced per-step weights like the GRU's own BCE weights),
    logit interpolated between the every-3rd-step rows, mixed into the BCE target on original training series only.
    Re-cuts keep hard labels (they carry ~2/3 of the loss weight, so the effective distillation weight is about a/3:
    if the probe reads flat, a = 0.7-0.8 is the next try). Rank / aux losses still use hard labels.
  - Smoke (300 series, 1 fold, 2 epochs, `--aug 2 --vol --bocpd`, GPU): runs; Platt coefficients sensible; mean
    |p - label| on training steps 0.45 (the teacher is weak per point, AUC ~0.65, so p sits near the balanced 0.5:
    a = 0.5 is largely label smoothing plus a little teacher signal -> hence the constant control).
    The `--nbm --nbm-hpca --swa` path with `--distill` was checked by reading only (nothing there touches the
    targets). Flag off = bit-identical by construction (every change is behind `args.distill > 0`, no RNG calls);
    no pre-edit copy was diffed. CRLF kept.
- **LEAK (why this is a probe):** a fold-k training row's OOF score came from member models trained on fold k's
  validation series, so the probe's gain is an UPPER BOUND. A clean CV teacher needs an inner CV per outer fold
  (at least trees: 4 inner fits per outer fold). Production is cheap and clean anyway: the production GRU trains on
  all 10k series and the teacher is the existing 5-fold OOF, which is out-of-sample for every training series.
- **Expectation (written before any run):** GRU wavg +0.000 to +0.004 on the leaky probe, blend less.
- **Rule (written before any run):** read the `wavg ... vs --compare` line under EPOCH AVERAGING (the per-fold
  "paired difference" line uses the best epoch, selection-optimistic). Same-seed GRU paired sd is ~0.005 per fold.
  1. Probe a = 0.5, seed 0, folds 0-1. Both folds <= 0 -> CLOSE. Mixed -> run folds 2-4 before building anything.
     Pass to step 2 only at mean >= +0.002 with both folds > 0 (higher bar because the probe is leaky).
  2. Control: the same run with `--distill-file cv_out/teacher_const.npz`. Build the clean nested teacher only if
     teacher - constant > 0. If the constant (pure smoothing) wins by itself, it is leak-free: judge it under the
     standard rule (5 folds, both seeds, blend >= +0.0005, >= 4/5, > 2 SE), no nesting needed.
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
  0. teachers (2 s each; already written 2026-10-22): `python distill_teacher.py` and
     `python distill_teacher.py --constant --out cv_out/teacher_const.npz`
  1. probe: `python gru_cv.py --folds 2 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --distill 0.5 --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_dist05`
  2. control (only if 1 passes): command 1 with `--distill-file cv_out/teacher_const.npz --tag aug2_vol_bo_nbm_hpca8_swa_dist05c`
- **RESULT step 1, leaky probe a = 0.5, seed 0 (2026-10-22):** Platt coef (z, z x ls, ls) ~ (0.34, 0.47, 0.44), so the
  teacher sharpens with the step; mean |p - label| 0.46. wavg 0.6596 / 0.6421 vs reference 0.6578 / 0.6395:
  **+0.0018 / +0.0026, mean +0.0022 +- 0.0004 (SE), 2/2** -> passes the pre-written +0.002 bar (only just).
  lavg +0.0023. ep7 -0.0032 (0/2) and wavg - ep7 +0.0053: the soft targets make single epochs noisier, averaging
  removes it. Best-epoch line +0.0012 (selection-optimistic, not the gate). Gain is in every fold but it is an upper
  bound (leak). OOF: `cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_dist05_wavg.npz` (folds 0-1 only).
- **RESULT step 2, constant-teacher control (2026-10-22):** wavg 0.6543 / 0.6390 vs reference 0.6578 / 0.6395:
  **-0.0035 / -0.0005, mean -0.0020 +- 0.0015, 0/2**. Pure label smoothing HURTS; teacher - constant
  **+0.0053 / +0.0031**. Verdict by the pre-written rule: the gain is the teacher's information, not smoothing ->
  build the clean teacher. (The leak can still be part of it: that is what the clean run decides.)
- **Clean teacher, BUILT 2026-10-22:**
  - `cv.py --lpo K` (leave-pair-out): for each inner fold i != K, train on the 3 folds outside {K, i} (6,000 series)
    and score fold i, using outer fold K's NBM / hpca columns (the NBM is label-free, so no leak). The OOF file covers
    fold K's 8,000 training series; fold K's rows stay NaN; nothing saw fold K's labels. Round / lambda selection
    happens on fold i (allowed: fold i is training data for the outer fold K). Smoke (400 series): NaN rows ==
    fold K's rows exactly; flag off = bit-identical OOF to the pre-edit `cv.py` (checked by a run, 2 folds).
  - `distill_teacher.py`: empty groups skipped (`--b` with no files), `--allow-nan` for partial files.
  - `gru_cv.py --distill-file` with `{f}` in the name loads one teacher per outer fold and refuses a file with NaN on
    the fold's training rows or scores on its validation rows. Full-size smoke with a fake per-fold file (the
    leaky teacher with fold 0 blanked): same Platt coefficients and same epoch-1 loss / score as the leaky probe.
  - The clean CV teacher is built from 6,000-series members, so it is weaker than production's (8,000-series OOF on
    10k): CV will slightly UNDER-state the production gain.
  - The GRU is NOT in the clean teacher (`gru_cv.py` has no --lpo). Command 3 below checks whether trees + SVM carry
    the gain; if they do not, a GRU --lpo is the next build.
- **Rule (written before these runs):**
  - Command 3 (leaky, no GRU in the teacher): if wavg mean < +0.001, the trees + SVM teacher does not carry the gain:
    STOP and build GRU --lpo before spending ~2 h on commands 4-6.
  - Command 8 (clean probe, folds 0-1): both folds > 0 and mean >= +0.001 -> extend: `--lpo 2/3/4` teachers, 5 folds,
    seed 1, then `blend.py` with the distilled GRU pair in the GRU slot vs the 0.6527 reference; ADOPT by the
    standing rule (>= +0.0005, >= 4/5, > 2 SE). Both folds <= 0 -> CLOSE (the leaky gain was leak). Mixed -> folds 2-4.
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
  3. leaky teacher WITHOUT the GRU (~3 min; teacher file already written, in-sample TS-AUC 0.6486):
     `python gru_cv.py --folds 2 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --distill 0.5 --distill-file cv_out/teacher_blend_nogru.npz --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_dist05ng`
  4. clean trees, outer folds 0 and 1, seeds 0 and 1 (4 commands, each 4 fits, ~20-30 min each, estimate):
     `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 600 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --lpo 0 --tag aug2_df_bo_nbm_bs24_ep25_lpo0`
     then the same with `--lpo 1 --tag aug2_df_bo_nbm_bs24_ep25_lpo1`, and both again with `--seed 1` and tags
     `..._lpo0_s1` / `..._lpo1_s1`.
  5. clean SVM, outer folds 0 and 1 (~5 min each):
     `python cv.py --backend svm --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --svm-top 100 --svm-gmul 0.5 --svm-imp cv_out/importance_aug2_df_bo_nbm_bs24_ep25.csv --lpo 0 --tag svm_nbm_hpca8_lpo0`
     and the same with `--lpo 1 --tag svm_nbm_hpca8_lpo1`
  6. teachers (seconds): `python distill_teacher.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_lpo0.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_lpo0_s1.npz --b --c cv_out/oof_svm_nbm_hpca8_lpo0.npz --w 0.4 0 0.2 --allow-nan --out cv_out/teacher_lpo_f0.npz`
     and the same with lpo1 / `teacher_lpo_f1.npz`
  7. (optional, seconds) nothing: gru_cv.py prints each clean teacher's TS-AUC on its training rows.
  8. clean probe (~3 min): command 3 with `--distill-file "cv_out/teacher_lpo_f{f}.npz" --tag aug2_vol_bo_nbm_hpca8_swa_dist05lpo`
- **RESULT command 3, leaky teacher without the GRU (trees + SVM, in-sample 0.6486; 2026-10-22):** wavg 0.6590 /
  0.6418 vs 0.6578 / 0.6395: **+0.0012 / +0.0023, mean +0.0018 +- 0.0006, 2/2** (with the GRU in the teacher:
  +0.0022). Above the +0.001 stop line: trees + SVM carry most of the gain, so the clean teacher needs no GRU --lpo.
- **Command 4/5 progress:** trees `--lpo 0` seed 0 done (each inner fit only ~30 s, not 20-30 min: ~2.5 min per
  command); inner fits on 6,000 series score 0.6309 at their best averaged round (vs ~0.642 for 8,000-series trees):
  the clean teacher is weaker, as expected. SVM `--lpo 0` done (0.6342; ~1 min per fit). Still to run: trees lpo0 s1,
  trees lpo1 s0 / s1, SVM lpo1.
- **RESULT command 8, CLEAN probe (2026-10-22):** clean teachers (trees s0+s1 + SVM, 6,000-series members) TS-AUC on
  the training rows 0.6386 / 0.6460. GRU wavg 0.6566 / 0.6393 vs reference 0.6578 / 0.6395: **-0.0012 / -0.0002,
  mean -0.0007 +- 0.0006, 0/2** (lavg -0.0005). By the pre-written rule (both folds <= 0): **CLOSED.**
- **Reading:** the same teacher recipe went from +0.0018 (leaky, trees + SVM) to -0.0007 (clean): about +0.0025 of
  the leaky gain was the members having seen the validation fold. Caveat, recorded honestly: the clean teacher is
  also weaker (6,000- vs 8,000-series members, roughly 0.640 vs 0.649 on the training rows), so leak and teacher
  strength are not fully separated; but a production teacher would be only one notch stronger than the leaky one's
  out-of-sample part, and nothing here suggests a gain worth a submission. Lesson for section 9: an OOF score used
  as a TRAINING target inside the same CV is leaky in a way that inflates CV by ~0.002-0.003 on the GRU; always
  test teachers / stacked targets with `cv.py --lpo`.
- Files: `cv_out/oof_*_lpo0*.npz`, `*_lpo1*.npz` (trees s0/s1, SVM), `cv_out/teacher_lpo_f0/1.npz`,
  `teacher_blend*.npz`, `teacher_const.npz`; GRU outputs `oof_gru_aug2_vol_bo_nbm_hpca8_swa_dist05*`.

## 45. XGBoost DART trees `cv.py --dart` (owner's go 2026-10-22, option 3 of "boosting methods for the blend"; RESULT: CLOSED by the early stop 2026-10-22)

- **Idea:** DART = boosting with dropout. Each round a random share of the existing trees is left out while the new
  tree is fitted, so late trees cannot just patch the last few trees' residuals; it fights over-specialisation of
  late trees. The only XGBoost booster variant never tried (the section 44.3 sweep, monotone, base-margin and lambda
  were all gbtree). Inference cost is unchanged (dropout is training-only; prediction uses every tree).
- **Build:** `--dart RATE` sets booster dart, rate_drop RATE, `--dart-skip` (default 0.5) skip_drop, uniform
  sampling, tree normalisation; everything else as the reference trees. Guard: xgb backend, binary objective, no
  --base-margin. Flag off = the gbtree code path, untouched.
  - Checked: on xgboost 3.4.1 (the cloud pin too) chunked `iteration_range` sums equal the full DART prediction
    up to the constant base score (spread 3e-6); prediction is deterministic. XGBoost warns that `booster=dart` is
    deprecated in favour of dropout parameters on the tree booster: it still works; harmless.
  - Caveat: the per-round curve is the first k trees with their FINAL weights, not the model as it was at round k,
    so "best round" means less than for gbtree. Production would train exactly R rounds and use all of them (NOT the
    `m[:200]` slicing used for gbtree).
  - Smoke (1,000 series, 2 folds, 600 rounds, GPU): runs; ~2.6x slower than gbtree (8 s -> 21 s per fold); the
    DART curve peaks later (rounds 150 / 600 vs 50 / 300), hence `--rounds 800` below.
- **Expectation (written before any run):** trees -0.002 to +0.002; prior small (the section 44.3 sweep says the
  day-one settings are a local optimum). Run time an estimate: ~15-25 min per fold.
- **Rule:** standing pass rule (section 5): seed 0 paired vs the reference trees, early stop if folds 0 AND 1 <= 0;
  then seed 1, the seed pair, the blend (>= +0.0005, >= 4/5, > 2 SE).
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time):**
  1. `python cv.py --gpu --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 800 --threads 6 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --dart 0.1 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz --tag aug2_df_bo_nbm_bs24_ep25_dart01`
     Read each `fold k: ... paired difference` line; stop after fold 1 if both are <= 0.
  2. seed 1 (only if 1 passes): command 1 plus `--seed 1`, `--compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`, `--tag aug2_df_bo_nbm_bs24_ep25_dart01_s1`
  3. `python diagnostics/oof_paired.py --a cv_out/oof_aug2_df_bo_nbm_bs24_ep25_dart01.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_dart01_s1.npz --ref cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz`
  4. blend (only if 3 > 0): the section 6 paired blend command with the DART pair in `--a`.
- **RESULT seed 0 (2026-10-22):** fold 0 0.6584 vs 0.6621 (**-0.0037**, best round 500, last 0.6568); fold 1 0.6320 vs
  0.6329 (**-0.0009**, best at the last round 800). Both <= 0: **CLOSED by the early stop** (owner stopped in fold 2).
  Run time ~250 s per fold (faster than the estimate). Note: fold 1 was still rising at round 800, so more rounds
  might narrow fold 1, but fold 0 had already peaked at 500 and lost 0.0037; not worth a rerun (prior was small, and
  the reference picks its own best round too).

## 46. NBM history hidden states in the GRU burn-in `--nbm-hist` and NBM gate activity `--nbm-gates` (owner's go 2026-10-22, ideas 1 and 3 of "what is similar to the earlier-layers idea"; RESULT: BUILT, not run)

- **Why:** section 24's gain came from handing the GRU the label-free NBM's internal state. Reading `gru_cv.py`
  showed a blind spot: the 8 hidden-state channels are ZERO in the GRU's 100-step history burn-in, so the GRU never
  sees what this series' NBM state looked like before the online part, only the online state on a scale shared by
  all series. The main obstacle (section 3) is series-specific wandering of no-break series.
- **Build:**
  - `sbrt/nbm.py run_hidden_full(hist, online, M)`: the NBM's state and gate activity after every history residual
    and every online point. The history loop mirrors `_init_core` (checked: its final state equals NbmStream's,
    asserted in every call); the online part equals `run_hidden()` bit for bit; truncating the online part leaves
    earlier rows unchanged (causal). Gate activity after each step from the step's own pre-activations:
    `nbg_upd` = mean over the 64 units of (1 - update gate) (share of memory overwritten), `nbg_rst` = mean reset
    gate, `nbg_ldh` = log ||h_t - h_t-1||.
  - `gru_cv.py --nbm-hist raw` (needs --nbm-hpca): the K hidden-state channels in the burn-in rows get the real
    history states (burn-in row j = history point len(h) - 100 + j, same PCA and scaling as the online rows).
    `--nbm-hist white`: as raw, and all hidden-state rows (burn-in and online) are re-expressed relative to the
    series' own history states from residual 50 on (minus their mean, over their sd floored at 0.2, in the
    globally standardised units), clipped +-6. Replaces the global channels; still 8 channels (16 overfit, section 30).
  - `--nbm-gates` (needs --nbm): 3 channels after the hidden-state channels, standardised per fold on the online
    rows of every 10th original training series, clipped +-6; in the burn-in only when --nbm-hist is not none.
  - Cached per fold: `data/gru_nbmx{K}{mode}_g{0|1}_b100v1_f{f}_...npz`. Build time ~12 s per 900 series on 8
    threads -> about 6-7 min per fold on first use (an estimate), then seconds.
  - Not combinable with --nbm-joint (guarded). Flag off: OOF bit-identical to the pre-edit gru_cv.py (300-series
    smoke, run). Smokes (300 series, 1-epoch NBM): raw + gates and white both run.
  - Inference cost: the NBM already runs over the whole history at inference; recording the last 100 states and
    the gate means is negligible.
- **Expectation (written before any run):** raw +0.000 to +0.004 (GRU wavg); white -0.003 to +0.004 (it removes
  the global level the GRU used, which may hurt); gates -0.001 to +0.002 (gates are a function of the state, so they
  may repeat what the 8 PCs carry).
- **Rule (written before any run):** each probe = GRU seed 0, folds 0-1, paired `wavg` vs the reference `_wavg` file
  (read the EPOCH AVERAGING `wavg ... vs --compare` line). Both folds <= 0 -> CLOSE that variant. Both > 0 -> the
  best variant (raw and gates may be combined in one run) goes to 5 folds, then seed 1, then `blend.py` with the new
  GRU pair in the GRU slot vs the 0.6527 reference; ADOPT at >= +0.0005, >= 4/5, > 2 SE. Mixed -> folds 2-4 first.
  Same-seed GRU paired sd is ~0.005 per fold, so a 2-fold probe can only flag clear effects.
- **Commands (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; GPU, one job at a time; ~15-20 min each the first
  time because of the channel build, ~3 min after):**
  1. raw: `python gru_cv.py --folds 2 --epochs 25 --feat-top 20 --aug 2 --vol --bocpd --nbm --nbm-batch 24 --nbm-epochs 25 --nbm-hpca 8 --nbm-hist raw --swa 5-8 --fixed-ep 7 --threads 8 --device cuda --compare cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz --tag aug2_vol_bo_nbm_hpca8_swa_hraw`
  2. white: command 1 with `--nbm-hist white --tag aug2_vol_bo_nbm_hpca8_swa_hwhite`
  3. gates: command 1 with `--nbm-gates` instead of `--nbm-hist raw`, `--tag aug2_vol_bo_nbm_hpca8_swa_gates`
