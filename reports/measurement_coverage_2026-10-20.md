# What can be measured in a time series, and what we already measure (2026-10-20)

Question (owner): check the papers and other sources for every kind of information that can be extracted from a time
series, and compare it with what the project already measures.

Reference convention: "§N" = a section of the current `HANDOFF.md`; "old §N" = a section of `HANDOFF_full_2026-10-17.md.bak`.

## Sources checked

- **hctsa** (Fulcher & Jones): over 7,700 features, grouped as distribution, correlation, entropy, stationarity,
  scaling, model fits and forecasting. pyhctsa ports about 4,500 of them.
- **catch22** (Lubba et al. 2019): 22 features chosen from hctsa, in 7 groups: distribution, simple temporal,
  linear autocorrelation, nonlinear autocorrelation, successive differences, fluctuation scaling, and other.
- **tsfresh**: about 75 calculators, among them energy, changes, autocorrelation/PACF, AR coefficients, ADF,
  entropies (approximate/sample/permutation/binned/Fourier), c3, CID, crossings, peaks, strikes, CWT, FFT
  aggregates, Langevin/Friedrich, Lempel-Ziv, matrix profile, Benford, duplicates and time-reversal asymmetry.
- **TSFEL**: 4 domains: statistical, temporal, spectral (fundamental frequency, roll-off, wavelet energy) and
  fractal (DFA, Higuchi, Hurst, Petrosian).
- **Change-point literature**: CUSUM of squares (Inclán-Tiao), spectral and Whittle change points (Lavielle-Ludeña),
  KS on the spectral distribution, ratio tests under GARCH errors. Also tail-dependence break tests: Bücher-Jäschke-Wied
  2015, Hoga 2018, MOPED 2025. These tail tests are multivariate; this search found no univariate extremal-index break test.
- **Competition write-ups**:
  - 2025 2nd place: 2,408 features. These are moments, quantiles, autocorrelation and the coefficient of variation on 6
    transforms (z, cumsum, rank, abs, moving average, moving sd), plus F / Levene / KS tests and TabPFN. TabPFN is
    forbidden here.
  - szukiyu40 (Real-Time, 0.6429): level, variability and lag correlation on raw values and AR residuals; Bayesian
    and adaptive-variance evidence; AR(12) innovations; quantile heads trained on normal histories (PIT, z, pinball);
    Chronos-2/Bolt patch embeddings vs the history; NLinear, GRU and TCN.
  - bamasa (0.6299): whitening, CUSUM/GLR/SR, spectral bands, BOCPD, TCN. Already reviewed in old §15.1.

## The key difference from those libraries

hctsa, catch22, tsfresh and TSFEL describe a **whole series** so that series can be classified. Here a statistic is
useful only as a **change**, measured at every online step: the online part against the series' own history. It
must also be put on a scale that is comparable across series, because TS-AUC ranks series against each other at each
step. Many library features do not carry over directly.

- Some grow with segment length, so their raw values are meaningless online. Examples: longest strike, number of
  peaks, sum of values, length.
- Some only describe the history, and history statistics alone have AUC ~0.5. Examples: periodicity, first ACF zero,
  histogram mode.

So the comparison below is by **what kind of change a statistic detects**, not by library.

## Coverage table

States: **PROD** = in the shipped models; **CLOSED** = built and tested, rejected (section reference); **NEW** =
never tested.

| Axis of change | Library examples | Our status |
|---|---|---|
| Level / mean | mean, median, linear trend, agg trend, cumsum transform, count above mean | PROD: z running mean, windows, scans, EWMA (`full_z`, `win*_z`, `glrmax_z`, `ewma_z`). Mean shifts are almost absent in the data (§3) |
| Scale / variance | variance, sd, energy, RMS, CoV, F-test, Levene, moving sd | PROD and very strong: `z2`, `e10sq`, `abse10`, the vol state (GARCH frozen path), BOCPD log variance, NBM `nb_lvr`. Levene and F are robust or 2-sample versions of the same thing |
| Successive differences / roughness | abs sum of changes, mean abs change, 2nd derivative, CID, pNN40, Hjorth mobility/complexity | PROD: variogram lags 1/2/4/8/16 (`vg*`), `abs_abs1`. CLOSED: higher derivatives (old §46) |
| Linear dependence (short) | ACF, PACF, AR coefficients, ACF 1/e decay, first min, Welch centroid / low-band power, rolling-mean forecast error | PROD: lag products at 1/2/3/5/10, AR(1)/AR(10) residuals, `ctx_rho1`/`ctx_phi`. CLOSED: Rio-CPD lag-covariance geodesic (§21), AR-coefficient change in BOCPD2 (old §28) |
| Linear dependence (long) / spectrum / scaling | FFT, Welch bands, spectral entropy, roll-off, DFA, Hurst, R/S, Higuchi, variance ratios | CLOSED, online change version tested: variance ratios to m = 64, DFA-1, low-frequency periodogram share, ACF lags 20/50, residual lags 11-50 (old §34, FAIL). Also as history descriptors (old §12). Short spectral shape = the lag products above |
| Marginal shape | skew, kurtosis, quantiles, histogram mode, binned entropy, symmetry, KS/AD/CvM/W1 | PROD: `z3`, `z4`, decile chi-square, ECDF ranks `u`, KS/AD/CvM/W1 on e and vol-adjusted u (`--dist`). CLOSED: MMD on single residuals / pairs / triples (old §20 oracle, old §21 `--mmd`) |
| Tails (rate) | ratio beyond r sigma, count above/below, abs max, mean of n largest | PROD: tail flags at 1 / 1.5 / 2 / 3 / 4 sigma on z and e10, in all windows and scans |
| Tails (shape at fixed scale) | Hill / tail index, extreme-value fits | Mostly PROD indirectly: AD on the vol-adjusted u is tail-weighted, and tail rates at 5 thresholds trace the tail shape. **NEW as an explicit tail index** |
| Timing / clustering of extremes | catch22 `DN_OutlierInclude_*` (gaps between extreme events), extremal index | **NEW.** Closest tested: squared-residual (volatility) clustering at lags 2-20, no signal (old §20) |
| Nonlinear dependence | automutual information, c3, time-reversal asymmetry, `CO_trev`, leverage, Langevin/Friedrich, Embed2 distances | CLOSED: leverage, vol clustering lags 2-20, irreversibility on e and z (old §20); MMD on pairs and triples; nonlinear mean R1 (old §39); backcasting shows the series are reversible (old §31) |
| Quantile / tail dependence | quantilogram, cross-quantilogram, tail-copula breaks | **NEW** (the univariate quantilogram at lags 1..k). Partly overlaps the closed vol clustering |
| Symbolic / ordinal / motifs | permutation entropy, MotifThree, transition matrix, strikes, crossings, peaks, turning points | CLOSED: ordinal patterns (old §20), symbol transitions `--shape` (§20), matrix profile / FLUSS (old §24), conv exceedance (§28). PROD: sign persistence `signpers` (= zero-crossing rate), `centre`. Turning points = lag-1 dependence of differences ~ `vg1`/`vg2` |
| Regularity / complexity | approximate / sample / spectral / permutation entropy, Lempel-Ziv, CID | CLOSED: entropy family (old §30, §31b): no rise at breaks; `--pshape` |
| Stationarity tests | ADF, KPSS, hctsa `SY_` sliding-window stats | Level part: no mean shifts. Variance part: PROD (windows / scans). Persistence: covered by AR(10) and the closed variance ratios (old §34) |
| Model-based / predictive | AR fits, forecast errors, PIT, pinball, model-fit residuals, foundation-model surprise | PROD: AR(10) residuals, GARCH frozen forecasts, NBM Student-t PIT-like stream through BOCPD, hidden-state PCA. CLOSED: Chronos-2/Bolt (old §29.4 / old §35.1, over budget and gate blind), richer one-step density (old §39), NBM fine-tune |
| Learned representations | Rocket/MiniRocket kernels, embeddings, TabPFN | CLOSED: conv exceedance (§28), contrastive embeddings (old §33). TabPFN is forbidden by the rules |
| Number format / metadata | Benford, duplicates, reoccurring values, value count | CLOSED: float32-exact, no duplicates, no splices (old §20, §12) |
| Cross-series | (none in the libraries) | CLOSED: per-step cross-sectional normalisation `--xsec` (§22); no shared histories |
| Change-detection evidence | CUSUM, SR, GLR, BOCPD, Bayesian mixtures | PROD: `--ev compact`, GLR scans, BOCPD on w / wu / NBM streams |

## Verdict

**Almost every axis the literature measures is already in production or was tested and closed.** That includes
everything in catch22 and every family that hctsa, tsfresh and TSFEL group by. The libraries add estimators of
properties we already measure, not new properties. The two strongest recent write-ups (szukiyu40's PIT / pinball heads
and Chronos embeddings, and the 2025 2nd place's F / Levene / KS on transforms) use the same information as our vol /
dist / NBM / BOCPD stack.

The section 9 lesson still holds: gains came from new information, and the data facts bound what is left.

- The variance-matched slice looked undetectable in every oracle (old §20, §16).
- Section 32 showed the no-break wandering is unpredictable from the history.

### What is genuinely untested: three tail-centred families

All three target breaks where the variance stays the same and the tails change. The prior for each is small: about
+0.000 to +0.001 on the blend, because the tail rates and the AD on u already cover much of it.

1. **Clustering and timing of extremes:** gaps between |w| > 2 or 3 exceedances; an extremal-index estimate (runs or
   intervals estimator) online vs the history value. Nearest negative result: the convolution-exceedance oracle (§28) was
   history-calibrated exceedance statistics on w / wu. Those collapsed into signed variance detectors and failed on
   the FLAT slice (-0.0287). So also check the new columns' correlation with the variance columns on the matched slice.
2. **Quantilogram:** correlation of 1{w_t < q} with 1{w_{t-k} < q} for q = 0.05 / 0.95 at k = 1..5, online minus
   history. This is dependence in the tails, which the squared-residual clustering does not isolate.
3. **Explicit tail index:** Hill estimate on |u| over the top ~10% of the history vs the online exceedances, or a
   GPD log-likelihood ratio.

**Cheapest test (do not build features first):** add these as a fourth family to `diagnostics/info_oracle.py`
(break position known, L = 50 / 100 / 300, variance-matched slice, incremental AUC over the baseline). Use the same
kill rule as old §34: PASS only if the gain is > +0.0066 and > 3 SE on all series, or >= +0.005 and > 2 SE on the
variance-matched slice. It is a CPU run of a few minutes. Expected result: FAIL, which would close the "properties of
the points" list for good.

Not worth testing: Hurst / Higuchi / Petrosian (covered by old §34), Levene / F (covered by variance), FFT / Welch /
wavelet energies (lags 1-16 and old §34), peaks / strikes / crossings (length-dependent or covered by `signpers` /
`vg1`), Benford and duplicates (old §20).

## Links

- hctsa / phenotyping: https://arxiv.org/pdf/1612.05296 ; Fulcher & Jones 2014: https://arxiv.org/pdf/1401.3531 ; pyhctsa: https://www.theoj.org/joss-papers/joss.10581/10.21105.joss.10581.pdf
- catch22: https://ar5iv.arxiv.org/html/1901.10200
- tsfresh list: https://tsfresh.readthedocs.io/en/main/text/list_of_features.html
- TSFEL: https://tsfel.readthedocs.io/
- Spectral change points: https://projecteuclid.org/download/pdfview_1/euclid.bj/1081282692 ; serial correlation and change-point tests: https://www.projecteuclid.org/euclid.aos/1176349042 ; ratio tests (GARCH errors): https://arxiv.org/pdf/0805.2473v1
- Tail-dependence breaks: https://arxiv.org/pdf/2509.00585 ; https://ideas.repec.org/a/oup/biomet/v105y2018i3p627-643..html ; https://wisostat.uni-koeln.de/sites/statistik/Dominik_Wied/Buecher_Jaeschke_Wied_2015_JoE.pdf
- 2025 2nd place: https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
- szukiyu40: https://github.com/szukiyu40/adia-structural-break-realtime ; bamasa: https://github.com/bamasa/adia-structural-break

## Result of the oracle (2026-10-20, `diagnostics/tail_oracle.py`, `cv_out/tail_oracle.txt`)

Exceedance clustering and the Hill index add nothing at any segment length. The quantilogram is the one formal PASS:
L = 600, variance-matched slice, +0.0252 +- 0.0064 (14/15). It is flat at L = 300 and below.

Only 1.5% of the TS-AUC weight sits on positive rows with at least 600 post-break points, so the face-value ceiling is
about +0.0001 on the blend before shrinkage. **CLOSED by the owner (2026-10-20).** With this, every axis in the literature is in production or closed. Details are in HANDOFF section 33.
