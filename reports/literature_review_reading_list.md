# Reading list for a literature review: real-time structural-break detection

For the ADIA Lab Structural Break Challenge, Real-Time Edition. Compiled 2026-10-16.

**How the entries were checked.** I checked every entry against a primary metadata source: a Crossref DOI record, the arXiv API, PMLR, NeurIPS or OpenReview pages, or the publisher's own page. I did not rely on memory or search snippets. Corrections found along the way are already applied. Where a page range could only be found on a secondary site, I left it out and kept the link. The one-line notes under each entry are my own summaries of why the paper matters here, not quotations, so check the paper before citing a note. ★ marks the core papers, 48 of them. A review built only from those would already cover the field.

---

## 0. How this problem maps onto the literature (a framing for the review)

The task has four features. Each one points to a different body of work:

| Feature of the task | Closest literature | Section |
|---|---|---|
| A score after every point saying a break *has already happened*. This is the posterior P(τ ≤ t \| data). | Bayesian quickest detection (Shiryaev), Shiryaev-Roberts, CUSUM, GLR | 2, 6 |
| "Normal" is known only through a finite, break-free history of 1,000 to 5,000 points. | Monitoring with a training sample (Chu-Stinchcombe-White and successors), and how estimated parameters degrade control charts | 3 |
| Breaks sit mostly in variance, tails and dependence, and break-free series drift on their own (volatility clustering). | Variance and GARCH change detection, and the confound between breaks, long memory and GARCH persistence | 4 |
| Thousands of labelled series, so a detector can be *learned*. | Density-ratio and classifier detectors, neural CUSUM, supervised and simulation-trained detectors | 9 |
| Scored by TS-AUC: an AUC across series at each step, then averaged. | Time-dependent ("cumulative/dynamic") ROC from survival analysis, a close analogue | 12 |

**A point the review can make.** Classical detection theory judges one stream at a time. It trades the false-alarm rate (ARL) against detection delay (EDD, Lorden's worst-case delay) and needs an alarm threshold. TS-AUC has no threshold and only ranks *different series at the same step*. Optimality results for CUSUM and SR therefore do not transfer directly. A score that is comparable across heterogeneous series matters as much as a score that is powerful within one series. The nearest formal tool for this kind of evaluation is the cumulative/dynamic time-dependent AUC (Heagerty et al. 2000; Heagerty & Zheng 2005; Zheng & Heagerty 2007), a close analogue rather than an exact match (see §12).

---

## 1. Start here: surveys and textbooks

- ★ **Xie, L., Zou, S., Xie, Y. & Veeravalli, V.V. (2021).** Sequential (quickest) change detection: classical results and new directions. *IEEE J. Selected Areas in Information Theory* 2(2):494–514. https://doi.org/10.1109/JSAIT.2021.3072962
  The best modern entry point. It covers the classical theory and the newer data-driven and unknown-distribution settings.
- ★ **Aue, A. & Kirch, C. (2024).** The state of cumulative sum sequential changepoint testing 70 years after Page. *Biometrika* 111(2):367–391. https://doi.org/10.1093/biomet/asad079
  A recent review of online CUSUM-type monitoring. It treats the training-sample and estimated-parameter issues that this task faces.
- ★ **Tartakovsky, A., Nikiforov, I. & Basseville, M. (2014).** *Sequential Analysis: Hypothesis Testing and Changepoint Detection.* Chapman & Hall/CRC. https://doi.org/10.1201/b17279
  The reference textbook.
- **Basseville, M. & Nikiforov, I.V. (1993).** *Detection of Abrupt Changes: Theory and Application.* Prentice Hall. ISBN 0-13-126780-9. https://openlibrary.org/books/OL1738695M
  The classic engineering textbook.
- **Poor, H.V. & Hadjiliadis, O. (2008).** *Quickest Detection.* Cambridge University Press. https://doi.org/10.1017/CBO9780511754678
  Short and rigorous. Some catalogues give the year as 2009, the copyright year.
- **Polunchenko, A.S. & Tartakovsky, A.G. (2012).** State-of-the-art in sequential change-point detection. *Methodology and Computing in Applied Probability* 14(3):649–684. https://doi.org/10.1007/s11009-011-9256-5
- ★ **Aminikhanghahi, S. & Cook, D.J. (2017).** A survey of methods for time series change point detection. *Knowledge and Information Systems* 51(2):339–367. https://doi.org/10.1007/s10115-016-0987-z
  A machine-learning-oriented survey covering supervised and unsupervised methods.
- **Truong, C., Oudre, L. & Vayatis, N. (2020).** Selective review of offline change point detection methods. *Signal Processing* 167:107299. https://doi.org/10.1016/j.sigpro.2019.107299
  Offline methods, but its cost / search / penalty taxonomy is useful for organising the review.
- ★ **Casini, A. & Perron, P. (2019).** Structural breaks in time series. *Oxford Research Encyclopedia of Economics and Finance.* https://doi.org/10.1093/acrefore/9780190625979.013.179 (arXiv: https://arxiv.org/abs/1805.03807)
  The econometric meaning of "structural break", surveyed.
- **Aue, A. & Horváth, L. (2013).** Structural breaks in time series. *J. Time Series Analysis* 34(1):1–16. https://doi.org/10.1111/j.1467-9892.2012.00819.x
- **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014).** A survey on concept drift adaptation. *ACM Computing Surveys* 46(4), Article 44. https://doi.org/10.1145/2523813
  The data-stream ("concept drift") view of the same problem.

---

## 2. Quickest-detection theory: CUSUM, Shiryaev-Roberts, GLR

- ★ **Page, E.S. (1954).** Continuous inspection schemes. *Biometrika* 41(1-2):100–115. https://doi.org/10.1093/biomet/41.1-2.100
  The origin of CUSUM. Every running-sum feature in this project descends from it.
- ★ **Shiryaev, A.N. (1963).** On optimum methods in quickest detection problems. *Theory of Probability & Its Applications* 8(1):22–46. https://doi.org/10.1137/1108002
  Bayesian quickest detection with a prior on the change time. Its central quantity, the posterior probability that the change has already happened, is exactly what the competition asks us to score.
- **Roberts, S.W. (1966).** A comparison of some control chart procedures. *Technometrics* 8(3):411–430. https://doi.org/10.1080/00401706.1966.10490374
  Introduces the Shiryaev-Roberts statistic, a sum over start times where CUSUM takes the maximum, and compares it with EWMA.
- ★ **Lorden, G. (1971).** Procedures for reacting to a change in distribution. *Annals of Mathematical Statistics* 42(6):1897–1908. https://doi.org/10.1214/aoms/1177693055
  Defines the worst-case detection delay and shows that CUSUM is asymptotically optimal.
- **Pollak, M. (1985).** Optimal detection of a change in distribution. *Annals of Statistics* 13(1):206–227. https://doi.org/10.1214/aos/1176346587
- **Moustakides, G.V. (1986).** Optimal stopping times for detecting changes in distributions. *Annals of Statistics* 14(4):1379–1387. https://doi.org/10.1214/aos/1176350164
  Exact, non-asymptotic optimality of CUSUM.
- ★ **Siegmund, D. & Venkatraman, E.S. (1995).** Using the generalized likelihood ratio statistic for sequential detection of a change-point. *Annals of Statistics* 23(1):255–271. https://doi.org/10.1214/aos/1176324466
  The GLR scan for when the size of the change is unknown. It is the basis of the "scan over all start times" features.
- ★ **Lai, T.L. (1995).** Sequential changepoint detection in quality control and dynamical systems. *JRSS B* 57(4):613–644 (discussion to p. 658). https://doi.org/10.1111/j.2517-6161.1995.tb02052.x
  Window-limited GLR with practical guidance. A read-paper with discussion.
- **Lai, T.L. (1998).** Information bounds and quick detection of parameter changes in stochastic systems. *IEEE Trans. Information Theory* 44(7):2917–2929. https://doi.org/10.1109/18.737522
  Delay bounds for *dependent* observations, so the theory covers autocorrelated series.
- ★ **Lai, T.L. & Xing, H. (2010).** Sequential change-point detection when the pre- and post-change parameters are unknown. *Sequential Analysis* 29(2):162–175. https://doi.org/10.1080/07474941003741078
  The classical setting closest to ours: the "before" regime is estimated and the "after" regime is unknown.

---

## 3. Monitoring with a break-free training sample (the closest formal match)

These papers fix a historical, break-free sample of size m and then monitor new observations one at a time. That is exactly the history/online split of the competition.

- ★ **Chu, C.-S.J., Stinchcombe, M. & White, H. (1996).** Monitoring structural change. *Econometrica* 64(5):1045–1065. https://doi.org/10.2307/2171955
  The founding paper of the monitoring paradigm: train on a stable historical period, then monitor with a controlled false-alarm probability.
- ★ **Horváth, L., Hušková, M., Kokoszka, P. & Steinebach, J. (2004).** Monitoring changes in linear models. *J. Statistical Planning and Inference* 126(1):225–251. https://doi.org/10.1016/j.jspi.2003.07.014
- **Aue, A., Horváth, L., Hušková, M. & Kokoszka, P. (2006).** Change-point monitoring in linear models. *The Econometrics Journal* 9(3):373–403. https://doi.org/10.1111/j.1368-423X.2006.00190.x
  The distribution of the detection time, i.e. the theory of the delay.
- **Zeileis, A., Leisch, F., Kleiber, C. & Hornik, K. (2005).** Monitoring structural change in dynamic econometric models. *J. Applied Econometrics* 20(1):99–121. https://doi.org/10.1002/jae.776
  Monitoring in autoregressive models. Implemented in R's `strucchange`.
- **Fremdt, S. (2015).** Page's sequential procedure for change-point detection in time series regression. *Statistics* 49(1):128–155. https://doi.org/10.1080/02331888.2013.870568
  A Page-CUSUM form of monitoring, which is faster for late changes.
- ★ **Kirch, C. & Weber, S. (2018).** Modified sequential change point procedures based on estimating functions. *Electronic J. Statistics* 12(1):1579–1613. https://doi.org/10.1214/18-EJS1431
  A unified monitoring framework that covers AR and other time-series models.
- **Dette, H. & Gösmann, J. (2020).** A likelihood ratio approach to sequential change point detection for a general class of parameters. *JASA* 115(531):1361–1377. https://doi.org/10.1080/01621459.2019.1630562
- ★ **Gösmann, J., Kley, T. & Dette, H. (2021).** A new approach for open-end sequential change point monitoring. *J. Time Series Analysis* 42(1):63–84. https://doi.org/10.1111/jtsa.12555
  A monitoring statistic that scans over candidate start times. It matches the project's finding that "a scan beats history-vs-all-so-far".
- **Gösmann, J., Stoehr, C., Heiny, J. & Dette, H. (2022).** Sequential change point detection in high dimensional time series. *Electronic J. Statistics* 16(1):3608–3671. https://doi.org/10.1214/22-EJS2027
- ★ **Hlávka, Z., Hušková, M., Kirch, C. & Meintanis, S.G. (2012).** Monitoring changes in the error distribution of autoregressive models based on Fourier methods. *TEST* 21(4):605–634. https://doi.org/10.1007/s11749-011-0265-z
  Monitors the *whole* residual distribution of an AR model, through its empirical characteristic function, with a bootstrap. This is very close to the project's "whiten, then test the distribution" streams.
- ★ **Kojadinovic, I. & Verdier, G. (2021).** Nonparametric sequential change-point detection for multivariate time series based on empirical distribution functions. *Electronic J. Statistics* 15(1):773–829. https://doi.org/10.1214/21-EJS1798
  ECDF-based detectors for *dependent* data, with a dependent multiplier bootstrap.
- **Pape, K., Wied, D. & Galeano, P. (2016).** Monitoring multivariate variance changes. *J. Empirical Finance* 39:54–68. https://doi.org/10.1016/j.jempfin.2016.08.007

### 3b. When "normal" is only estimated: the effect of a finite reference sample

- ★ **Jensen, W.A., Jones-Farmer, L.A., Champ, C.W. & Woodall, W.H. (2006).** Effects of parameter estimation on control chart properties: a literature review. *J. Quality Technology* 38(4):349–364. https://doi.org/10.1080/00224065.2006.11918623
  Estimating "normal" from a finite Phase I sample changes how a chart behaves. Here the history is 1,000 to 5,000 points of exactly such a sample.
- **Jones, L.A., Champ, C.W. & Rigdon, S.E. (2004).** The run length distribution of the CUSUM with estimated parameters. *J. Quality Technology* 36(1):95–108. https://doi.org/10.1080/00224065.2004.11980254
- **Gandy, A. & Kvaløy, J.T. (2013).** Guaranteed conditional performance of control charts via bootstrap methods. *Scandinavian J. Statistics* 40(4):647–668. https://doi.org/10.1002/sjos.12006
  Calibrates each chart *conditional on its own training sample*. This is the per-series calibration idea, done properly.
- **Chatterjee, S. & Qiu, P. (2009).** Distribution-free cumulative sum control charts using bootstrap-based control limits. *Annals of Applied Statistics* 3(1):349–369. https://doi.org/10.1214/08-AOAS197

---

## 4. Variance, volatility and dependence breaks, and what imitates them

In this data the signal sits in variance, tails and dependence, and break-free series drift as much as broken ones. This literature explains both.

- ★ **Inclán, C. & Tiao, G.C. (1994).** Use of cumulative sums of squares for retrospective detection of changes of variance. *JASA* 89(427):913–923. https://doi.org/10.1080/01621459.1994.10476824
  CUSUM of squares, the standard variance-change test.
- ★ **Berkes, I., Gombay, E., Horváth, L. & Kokoszka, P. (2004).** Sequential change-point detection in GARCH(p,q) models. *Econometric Theory* 20(6):1140–1167. https://doi.org/10.1017/S0266466604206041
  Sequential detection with a training sample when the null itself has volatility clustering. This is the formal version of the project's `--vol` stream.
- ★ **Lamoureux, C.G. & Lastrapes, W.D. (1990).** Persistence in variance, structural change, and the GARCH model. *J. Business & Economic Statistics* 8(2):225–234. https://doi.org/10.1080/07350015.1990.10509794
  Unmodelled breaks make GARCH look highly persistent. It is the same confound seen from the other side.
- ★ **Mikosch, T. & Stărică, C. (2004).** Nonstationarities in financial time series, the long-range dependence, and the IGARCH effects. *Review of Economics and Statistics* 86(1):378–390. https://doi.org/10.1162/003465304323023886
- **Diebold, F.X. & Inoue, A. (2001).** Long memory and regime switching. *J. Econometrics* 105(1):131–159. https://doi.org/10.1016/S0304-4076(01)00073-2
- **Berkes, I., Horváth, L., Kokoszka, P. & Shao, Q.-M. (2006).** On discriminating between long-range dependence and changes in mean. *Annals of Statistics* 34(3):1140–1165. https://doi.org/10.1214/009053606000000254
- **Andreou, E. & Ghysels, E. (2002).** Detecting multiple breaks in financial market volatility dynamics. *J. Applied Econometrics* 17(5):579–600. https://doi.org/10.1002/jae.684
- **Gombay, E. (2008).** Change detection in autoregressive time series. *J. Multivariate Analysis* 99(3):451–464. https://doi.org/10.1016/j.jmva.2007.01.003
- **Shao, X. & Zhang, X. (2010).** Testing for change points in time series. *JASA* 105(491):1228–1240. https://doi.org/10.1198/jasa.2010.tm10103
  Self-normalised tests that are robust to serial dependence and need no bandwidth choice.
- **Hamilton, J.D. (1989).** A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2):357–384. https://doi.org/10.2307/1912559
  Markov regime switching, the "recurring regimes" alternative to a single permanent break.
- **Pesaran, M.H. & Timmermann, A. (2007).** Selection of estimation window in the presence of breaks. *J. Econometrics* 137(1):134–161. https://doi.org/10.1016/j.jeconom.2006.03.010

---

## 5. Econometric structural-break tests (background and terminology)

- ★ **Chow, G.C. (1960).** Tests of equality between sets of coefficients in two linear regressions. *Econometrica* 28(3):591–605. https://doi.org/10.2307/1910133
  A break test at a *known* date. The 2025 offline edition of the challenge is essentially this problem.
- **Brown, R.L., Durbin, J. & Evans, J.M. (1975).** Techniques for testing the constancy of regression relationships over time. *JRSS B* 37(2):149–163 (discussion to p. 192). https://doi.org/10.1111/j.2517-6161.1975.tb01532.x
  CUSUM and CUSUM-of-squares of recursive residuals.
- ★ **Andrews, D.W.K. (1993).** Tests for parameter instability and structural change with unknown change point. *Econometrica* 61(4):821–856. https://doi.org/10.2307/2951764
  Sup-Wald/LM/LR tests over all candidate dates. Correction: *Econometrica* 71(1):395–397 (2003), https://doi.org/10.1111/1468-0262.00405.
- ★ **Bai, J. & Perron, P. (1998).** Estimating and testing linear models with multiple structural changes. *Econometrica* 66(1):47–78. https://doi.org/10.2307/2998540
- **Bai, J. & Perron, P. (2003).** Computation and analysis of multiple structural change models. *J. Applied Econometrics* 18(1):1–22. https://doi.org/10.1002/jae.659

---

## 6. Bayesian online change-point detection

- ★ **Adams, R.P. & MacKay, D.J.C. (2007).** Bayesian online changepoint detection. arXiv:0710.3742. https://arxiv.org/abs/0710.3742
  The run-length posterior (BOCPD). The project's `bocpd.py` specialises it to at most one break with a known pre-break regime.
- ★ **Fearnhead, P. & Liu, Z. (2007).** On-line inference for multiple changepoint problems. *JRSS B* 69(4):589–605. https://doi.org/10.1111/j.1467-9868.2007.00601.x
- **Saatçi, Y., Turner, R.D. & Rasmussen, C.E. (2010).** Gaussian process change point models. *ICML 2010.* https://icml.cc/Conferences/2010/papers/170.pdf
  BOCPD with dependent (GP) segments.
- **Knoblauch, J. & Damoulas, T. (2018).** Spatio-temporal Bayesian on-line changepoint detection with model selection. *ICML 2018*, PMLR 80:2718–2727. https://proceedings.mlr.press/v80/knoblauch18a.html
  BOCPD with model selection over autoregressive models.
- ★ **Knoblauch, J., Jewson, J. & Damoulas, T. (2018).** Doubly robust Bayesian inference for non-stationary streaming data with β-divergences. *NeurIPS 2018.* https://proceedings.neurips.cc/paper_files/paper/2018/hash/a3f390d88e4c41f2747bfa2f1b5f87db-Abstract.html
  BOCPD made robust to heavy tails and outliers, a known failure mode in this data.
- **Altamirano, M., Briol, F.-X. & Knoblauch, J. (2023).** Robust and scalable Bayesian online changepoint detection. *ICML 2023*, PMLR 202:642–663. https://proceedings.mlr.press/v202/altamirano23a.html
- **Sellier, J. & Dellaportas, P. (2023).** Bayesian online change point detection with Hilbert space approximate Student-t process. *ICML 2023*, PMLR 202:30553–30569. https://proceedings.mlr.press/v202/sellier23a.html
  Heavy-tailed segments are modelled directly.
- ★ **Tsaknaki, I.-Y., Lillo, F. & Mazzarisi, P. (2025).** Bayesian autoregressive online change-point detection with time-varying parameters. *Communications in Nonlinear Science and Numerical Simulation* 142:108500. https://doi.org/10.1016/j.cnsns.2024.108500
  BOCPD on AR regimes whose variance and correlation drift over time, aimed at volatile financial series. This is the "null drifts on its own" problem.

---

## 7. Nonparametric and kernel online detection (whole-distribution changes)

- ★ **Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. & Smola, A. (2012).** A kernel two-sample test. *JMLR* 13:723–773. https://jmlr.org/papers/v13/gretton12a.html
  MMD, the basic tool of this section.
- **Harchaoui, Z., Bach, F. & Moulines, E. (2008).** Kernel change-point analysis. *NeurIPS 21 (2008).* https://proceedings.neurips.cc/paper_files/paper/2008/hash/08b255a5d42b89b0585260b6f2360bdd-Abstract.html
- **Arlot, S., Celisse, A. & Harchaoui, Z. (2019).** A kernel multiple change-point algorithm via model selection. *JMLR* 20(162):1–56. https://jmlr.org/papers/v20/16-155.html
  KCP. The project reviewed it in HANDOFF §24.
- ★ **Li, S., Xie, Y., Dai, H. & Song, L. (2019).** Scan B-statistic for kernel change-point detection. *Sequential Analysis* 38(4):503–544. https://doi.org/10.1080/07474946.2019.1686886 (arXiv: https://arxiv.org/abs/1507.01279; early version at NIPS 2015 as "M-statistic for kernel change-point detection")
  An online kernel scan against a *reference pool drawn from historical data*, which fits the history/online design.
- ★ **Wei, S. & Xie, Y. (2026).** Online kernel CUSUM for change-point detection. *JRSS B* 88(4):1251–1277. https://doi.org/10.1093/jrsssb/qkag020 (arXiv: https://arxiv.org/abs/2211.15070)
- **Wei, S. & Huang, C. (2022).** Optimal sub-sampling to boost power of kernel sequential change-point detection. arXiv:2210.15060. https://arxiv.org/abs/2210.15060
  How to sub-sample the historical reference for Scan-B and kernel CUSUM.
- **Keriven, N., Garreau, D. & Poli, I. (2020).** NEWMA: a new method for scalable model-free online change-point detection. *IEEE Trans. Signal Processing* 68:3515–3528. https://doi.org/10.1109/TSP.2020.2990597
  Two EWMAs of random features, O(1) per step.
- ★ **Kalinke, F. & Gavioli-Akilagun, S. (2025).** Optimal online change detection via random Fourier features. *NeurIPS 2025* (vol. 38), pp. 10919–10954. https://doi.org/10.52202/085713-0333 (arXiv: https://arxiv.org/abs/2505.17789)
  Random-Fourier-feature MMD at dyadic split points, O(r log n) per step.
- **Kalinke, F., Heyden, M., Gntuni, G., Fouché, E. & Böhm, K. (2025).** Maximum mean discrepancy on exponential windows for online change detection. *TMLR* (Feb 2025). https://openreview.net/forum?id=OGaTF9iOxi (arXiv: https://arxiv.org/abs/2205.12706)
- **Matteson, D.S. & James, N.A. (2014).** A nonparametric approach for multiple change point analysis of multivariate data. *JASA* 109(505):334–345. https://doi.org/10.1080/01621459.2013.849605
  E-divisive, built on energy distance.
- ★ **Hawkins, D.M., Qiu, P. & Kang, C.W. (2003).** The changepoint model for statistical process control. *J. Quality Technology* 35(4):355–366. https://doi.org/10.1080/00224065.2003.11980233
  The "changepoint model" (CPM): a sequential GLR when no parameters are known.
- ★ **Ross, G.J., Tasoulis, D.K. & Adams, N.M. (2011).** Nonparametric monitoring of data streams for changes in location and scale. *Technometrics* 53(4):379–389. https://doi.org/10.1198/tech.2011.10069
- **Ross, G.J. & Adams, N.M. (2012).** Two nonparametric control charts for detecting arbitrary distribution changes. *J. Quality Technology* 44(2):102–116. https://doi.org/10.1080/00224065.2012.11917887
  CPMs built on Kolmogorov-Smirnov and Cramér-von Mises statistics, the online versions of the project's `--dist` tests.
- **Ross, G.J. (2015).** Parametric and nonparametric sequential change detection in R: the cpm package. *J. Statistical Software* 66(3):1–20. https://doi.org/10.18637/jss.v066.i03
- **Chen, H. (2019).** Sequential change-point detection based on nearest neighbors. *Annals of Statistics* 47(3):1381–1407. https://doi.org/10.1214/18-AOS1718
- **Kifer, D., Ben-David, S. & Gehrke, J. (2004).** Detecting change in data streams. *VLDB 2004*, pp. 180–191. https://doi.org/10.1016/B978-012088469-8.50019-X
- **Bifet, A. & Gavaldà, R. (2007).** Learning from time-changing data with adaptive windowing. *SIAM SDM 2007*, pp. 443–448. https://doi.org/10.1137/1.9781611972771.42
  ADWIN.
- **Gharghabi, S., Ding, Y., Yeh, C.-C.M., Kamgar, K., Ulanova, L. & Keogh, E. (2017).** Matrix Profile VIII: domain agnostic online semantic segmentation at superhuman performance levels. *IEEE ICDM 2017*, pp. 117–126. https://doi.org/10.1109/ICDM.2017.21
  FLUSS. The project found no signal from it on noise-like series (HANDOFF §24), which is itself a reportable negative result.

---

## 8. Conformal martingales and e-detectors (assumption-light sequential tests)

- **Vovk, V., Nouretdinov, I. & Gammerman, A. (2003).** Testing exchangeability on-line. *ICML 2003.* https://cdn.aaai.org/ICML/2003/ICML03-100.pdf
- **Volkhonskiy, D., Burnaev, E., Nouretdinov, I., Gammerman, A. & Vovk, V. (2017).** Inductive conformal martingales for change-point detection. *COPA 2017*, PMLR 60:132–153. https://proceedings.mlr.press/v60/volkhonskiy17a.html
- ★ **Vovk, V. (2021).** Testing randomness online. *Statistical Science* 36(4):595–611. https://doi.org/10.1214/20-STS817
  CUSUM and Shiryaev-Roberts applied to conformal p-values. The project found these equivalent to its history-rank (ECDF) whitening plus CUSUM/SR.
- **Vovk, V., Petej, I., Nouretdinov, I., Ahlberg, E., Carlsson, L. & Gammerman, A. (2021).** Retrain or not retrain: conformal test martingales for change-point detection. *COPA 2021*, PMLR 152:191–210. https://proceedings.mlr.press/v152/vovk21b.html
  Detects when new data stop being exchangeable with a *reference/training* set.
- ★ **Shin, J., Ramdas, A. & Rinaldo, A. (2024).** E-detectors: a nonparametric framework for sequential change detection. *New England J. Statistics in Data Science* 2(2):229–260. https://doi.org/10.51387/23-NEJSDS51 (arXiv: https://arxiv.org/abs/2203.03532)

---

## 9. Learned detectors: density ratios, classifiers, neural statistics, supervised training

The competition gives 10,000 labelled series, so the detector itself can be learned. This is the most direct link to the project's trees, GRU and SVM.

- ★ **Li, J., Fearnhead, P., Fryzlewicz, P. & Wang, T. (2024).** Automatic change-point detection in time series via deep learning. *JRSS B* 86(2):273–285. https://doi.org/10.1093/jrsssb/qkae004 (arXiv: https://arxiv.org/abs/2211.03860)
  Shows that CUSUM is a small ReLU network and trains detectors on *labelled/simulated series*. It beats CUSUM under autocorrelated or heavy-tailed noise. This is the key paper for the "learned detector" argument.
- ★ **Kawahara, Y. & Sugiyama, M. (2012).** Sequential change-point detection based on direct density-ratio estimation. *Statistical Analysis and Data Mining* 5(2):114–127. https://doi.org/10.1002/sam.10124
- ★ **Liu, S., Yamada, M., Collier, N. & Sugiyama, M. (2013).** Change-point detection in time-series data by relative density-ratio estimation. *Neural Networks* 43:72–83. https://doi.org/10.1016/j.neunet.2013.01.012
  RuLSIF. The relative ratio keeps scores bounded and stable.
- **Hushchyn, M. & Ustyuzhanin, A. (2021).** Generalization of change-point detection in time series data based on direct density ratio estimation. *J. Computational Science* 53:101385. https://doi.org/10.1016/j.jocs.2021.101385
  A classifier trained to separate "recent" from "past" windows serves as the density-ratio estimator.
- ★ **Hushchyn, M., Arzymatov, K. & Derkach, D. (2026).** Online neural networks for change-point detection. *Machine Learning* 115(3), article 56. https://doi.org/10.1007/s10994-026-07000-6 (arXiv: https://arxiv.org/abs/2010.01388)
  ONNC/ONNR: neural classifiers and regressors trained online.
- ★ **Londschien, M., Bühlmann, P. & Kovács, S. (2023).** Random forests for change point detection. *JMLR* 24(216):1–45. https://jmlr.org/papers/v24/22-0512.html
  Turns classifier probabilities into a log-likelihood-ratio change statistic (`changeforest`), the tree-based cousin of the project's approach.
- **Gong, T., Lee, J., Cheng, X. & Xie, Y. (2022).** Neural network-based CUSUM for online change-point detection. arXiv:2210.17312 (preprint only). https://arxiv.org/abs/2210.17312
  Peer-reviewed short version: **Lee, J., Xie, Y. & Cheng, X. (2023).** Training neural networks for sequential change-point detection. *ICASSP 2023*, pp. 1–5. https://doi.org/10.1109/ICASSP49357.2023.10095005
- **Lopez-Paz, D. & Oquab, M. (2017).** Revisiting classifier two-sample tests. *ICLR 2017.* https://arxiv.org/abs/1610.06545
- **Liu, F., Xu, W., Lu, J., Zhang, G., Gretton, A. & Sutherland, D.J. (2020).** Learning deep kernels for non-parametric two-sample tests. *ICML 2020*, PMLR 119:6316–6326. http://proceedings.mlr.press/v119/liu20m.html
- **Chang, W.-C., Li, C.-L., Yang, Y. & Póczos, B. (2019).** Kernel change-point detection with auxiliary deep generative models. *ICLR 2019.* https://arxiv.org/abs/1901.06077
  KL-CPD, a learned kernel for change detection.
- **Deldari, S., Smith, D.V., Xue, H. & Salim, F.D. (2021).** Time series change point detection with self-supervised contrastive predictive coding. *The Web Conference (WWW '21)*, pp. 3124–3135. https://doi.org/10.1145/3442381.3449903
- **De Ryck, T., De Vos, M. & Bertrand, A. (2021).** Change point detection in time series data using autoencoders with a time-invariant representation. *IEEE Trans. Signal Processing* 69:3513–3524. https://doi.org/10.1109/TSP.2021.3087031
- ★ **Romanenkova, E., Stepikin, A., Morozov, M. & Zaytsev, A. (2022).** InDiD: Instant disorder detection via a principled neural network. *ACM Multimedia (MM '22)*, pp. 3152–3162. https://doi.org/10.1145/3503161.3548182
  A training loss that rewards early detection and penalises false alarms. It is the closest neural analogue to optimising a per-step metric like TS-AUC.
- **Hocking, T.D., Rigaill, G., Vert, J.-P. & Bach, F. (2013).** Learning sparse penalties for change-point detection using max margin interval regression. *ICML 2013*, PMLR 28(3):172–180. http://proceedings.mlr.press/v28/hocking13.html
  Supervised learning from labelled change annotations.
- **Lin, S., Mengaldo, G. & Maulik, R. (2023).** Online data-driven changepoint detection for high-dimensional dynamical systems. *Chaos* 33(10):103112. https://doi.org/10.1063/5.0160312
  Compares an unsupervised and a supervised online detector.
- **Dudley, C., Bi, Y., Liu, X. & Oymak, S. (2026).** In-context learning under regime change. arXiv:2604.16988 (preprint). https://arxiv.org/abs/2604.16988
  Shows that transformers can learn change-point detection in context. Theory plus synthetic experiments.

---

## 10. Normal-behaviour (forecast-residual) models and time-series foundation models

The idea is to model the break-free process very well and score departures from it. The project's NBM and the competitor szukiyu40's NLinear/GRU/Chronos models belong here.

- **Malhotra, P., Vig, L., Shroff, G. & Agarwal, P. (2015).** Long short term memory networks for anomaly detection in time series. *ESANN 2015*, pp. 89–94. https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf
- ★ **Hundman, K., Constantinou, V., Laporte, C., Colwell, I. & Soderstrom, T. (2018).** Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *KDD '18*, pp. 387–395. https://doi.org/10.1145/3219819.3219845
  The standard forecast-residual pipeline.
- **Geng, J. & Michailidis, G. (2025).** Neural network-based change point detection for large-scale time-evolving data. arXiv:2503.09541 (preprint). https://arxiv.org/abs/2503.09541
  A residual detector: a network is fitted on one window, its error is calibrated on another, and a change is flagged when the moving-window error departs from that calibration. Comes with consistency theory under dependence.
- **Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023).** Are transformers effective for time series forecasting? *AAAI 2023* 37(9):11121–11128. https://doi.org/10.1609/aaai.v37i9.26317
  Introduces the NLinear/DLinear models a competitor used as normal-behaviour models.
- ★ **Ansari, A.F. et al. (2024).** Chronos: Learning the language of time series. *TMLR* 2024. https://arxiv.org/abs/2403.07815 (TMLR: https://openreview.net/forum?id=gerNCVqqtR)
- **Ansari, A.F. et al. (2025).** Chronos-2: From univariate to universal forecasting. arXiv:2510.15821 (technical report). https://arxiv.org/abs/2510.15821
- **Choi, J., Shook, B. & Dubrawski, A. (2026).** Non-stationarity in the embedding space of time series foundation models. arXiv:2604.16428 (preprint). https://arxiv.org/abs/2604.16428
  *Read with care.* It probes how mean shifts, variance changes and trends show up in foundation-model embeddings under controlled conditions. It is not a change-detector benchmark.
- **Uray, M., Messineo, S., Kwitt, R. & Huber, S. (2026).** Exploring zero-shot foundation models for multivariate time series anomaly detection. arXiv:2607.12454 (to appear, EUROCAST 2026, LNCS). https://arxiv.org/abs/2607.12454
  *Read with care.* According to its abstract, this is an anomaly-detection study (TimesFM, zero-shot, on the SWaT industrial benchmark) and the approach is not competitive there. The model adapts so well that errors stay low inside anomalous windows. It only *observes* that forecast errors peak at anomaly boundaries and concludes the approach is "promising for change-point detection", without running a change-detection test. `reports/research_summary.html` §4.4 cites it as evidence about breaks; it is indirect evidence at best.

---

## 11. Tools the pipeline uses (cite for the methods section)

- **Christ, M., Braun, N., Neuffer, J. & Kempa-Liehr, A.W. (2018).** Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package). *Neurocomputing* 307:72–77. https://doi.org/10.1016/j.neucom.2018.03.067
- **Lubba, C.H., Sethi, S.S., Knaute, P., Schultz, S.R., Fulcher, B.D. & Jones, N.S. (2019).** catch22: CAnonical Time-series CHaracteristics. *Data Mining and Knowledge Discovery* 33(6):1821–1852. https://doi.org/10.1007/s10618-019-00647-x
- **Chen, T. & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *KDD '16*, pp. 785–794. https://doi.org/10.1145/2939672.2939785
- **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. & Liu, T.-Y. (2017).** LightGBM: A highly efficient gradient boosting decision tree. *NIPS 2017.* https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- **Bai, S., Kolter, J.Z. & Koltun, V. (2018).** An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271. https://arxiv.org/abs/1803.01271
  TCN.

---

## 12. Evaluation: how to score a streaming detector

- ★ **Heagerty, P.J., Lumley, T. & Pepe, M.S. (2000).** Time-dependent ROC curves for censored survival data and a diagnostic marker. *Biometrics* 56(2):337–344. https://doi.org/10.1111/j.0006-341X.2000.00337.x
- ★ **Heagerty, P.J. & Zheng, Y. (2005).** Survival model predictive accuracy and ROC curves. *Biometrics* 61(1):92–105. https://doi.org/10.1111/j.0006-341X.2005.030814.x
  Defines the cumulative/dynamic AUC, where cases are the units whose event happened by t and controls are those still event-free, plus a time-integrated concordance summary. TS-AUC is closely analogous, with the break as the "event" and steps weighted by n_pos·n_neg, like the case-control pair weighting. It differs in two ways. First, the score is updated at every step instead of being a fixed baseline marker. Second, at step t only series that are still running are ranked, so series whose online segment has ended drop out, and the set at each step depends on online length. Censoring weights play no part.
- **Zheng, Y. & Heagerty, P.J. (2007).** Prospective accuracy for longitudinal markers. *Biometrics* 63(2):332–341. https://doi.org/10.1111/j.1541-0420.2006.00726.x
  The same ROC ideas for a marker that is re-measured over time. This is closer to a score updated at every online step.
- ★ **van den Burg, G.J.J. & Williams, C.K.I. (2020).** An evaluation of change point detection algorithms. arXiv:2003.06222. https://arxiv.org/abs/2003.06222
  The TCPD benchmark. Shows that the choice of metric and protocol changes the ranking of methods.
- **Wu, R. & Keogh, E. (2021, early access).** Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. *IEEE Trans. Knowledge and Data Engineering.* https://doi.org/10.1109/TKDE.2021.3112126
  A warning about benchmark artefacts. Useful when discussing the gap between CV and the leaderboard.

---

## 13. Offline change-point methods (for contrast)

- **Killick, R., Fearnhead, P. & Eckley, I.A. (2012).** Optimal detection of changepoints with a linear computational cost. *JASA* 107(500):1590–1598. https://doi.org/10.1080/01621459.2012.737745
  PELT.
- **Fryzlewicz, P. (2014).** Wild binary segmentation for multiple change-point detection. *Annals of Statistics* 42(6):2243–2281. https://doi.org/10.1214/14-AOS1245

(See also Truong et al. 2020 in §1 and Matteson & James 2014 in §7.)

---

## 14. The competition itself (grey literature, not peer-reviewed)

No official paper describes either edition of the challenge or its winning solutions. The 2026 winners are announced at the ADIA Lab symposium on 26–28 Oct 2026. What exists:

- CrunchDAO docs, Real-Time Edition (rules, data, TS-AUC): https://docs.crunchdao.com/competitions/competitions/structural-break-real-time
- ADIA Lab page, Real-Time Edition: https://www.adialab.ae/adia-lab-x-crunch-the-structural-break-challenge-2026
- CrunchDAO docs, 2025 offline edition: https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge · ADIA Lab 2025 page with the winners: https://www.adialab.ae/structural-break-challenge · Open Benchmark (a different, offline, ongoing task): https://www.adialab.ae/structural-break-open-benchmark
- bamasa (Real-Time, public 0.6299): whitening by the series' own history, CUSUM/GLR/SR, LightGBM rankers, causal TCNs, with a full experiment log. https://github.com/bamasa/adia-structural-break
- szukiyu40 / Yu Suzuki (Real-Time, public ~0.643): a statistical ensemble plus an adapted Chronos-2 encoder, with a poster in the repo. https://github.com/szukiyu40/adia-structural-break-realtime
- 2025 offline edition, 2nd-place code: about 2,400 features built from F, Levene and KS tests on transformed series, fed to LightGBM. https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
- CrunchDAO forum thread on the 2025 winning solutions: https://forum.crunchdao.com/t/winning-solutions-of-structural-break-challenge/1070
- **Yuksel, K.A. & Sawaf, H. (2026).** EvoForest: A novel machine-learning paradigm via open-ended evolution of computational graphs. arXiv:2604.19761 (preprint). https://arxiv.org/abs/2604.19761
  Uses the 2025 offline challenge as a benchmark and claims 94.13% ROC-AUC against a reported winning 90.14%. The claim is unreplicated, and it concerns the offline task, not real-time.

---

## 15. Suggested reading order (if you only have time for ten)

1. Xie, Zou, Xie & Veeravalli (2021). The map of the field.
2. Page (1954), Lorden (1971) and Shiryaev (1963). The three ideas everything builds on: CUSUM, worst-case delay, and the posterior that a change has happened.
3. Chu, Stinchcombe & White (1996). Monitoring after a stable training period, which is our exact setup.
4. Aue & Kirch (2024). What 70 years of CUSUM monitoring has learned.
5. Jensen et al. (2006). What happens when "normal" is only estimated.
6. Berkes, Gombay, Horváth & Kokoszka (2004) with Lamoureux & Lastrapes (1990). Volatility clustering against true breaks.
7. Adams & MacKay (2007). BOCPD.
8. Li, Xie, Dai & Song (2019). Kernel scan against a historical reference.
9. Li, Fearnhead, Fryzlewicz & Wang (2024). Learned detectors.
10. Heagerty & Zheng (2005). The ROC framework behind TS-AUC.

## 16. Gaps the review can argue (supported by this list and by the project's own experiments)

- **Metric mismatch.** Every theoretical paper here judges detectors by ARL/EDD on a single stream. None of the papers on this list studies ranking *many heterogeneous series against each other at the same step*, which is what TS-AUC rewards. Per-series calibration (Gandy & Kvaløy 2013) is the closest idea.
- **The null drifts.** Most detectors assume a stationary pre-change process. With volatility clustering and long memory, a break-free series can look like a broken one (Lamoureux & Lastrapes 1990; Mikosch & Stărică 2004). The project's error breakdown says this is the main obstacle.
- **Learned detectors need train/test match.** Li et al. (2024) give guarantees only when the training data look like the test data. That is consistent with the project's finding that synthetic breaks did not help but real re-cut data did.
- **The foundation-model evidence is thin.** Only preprints, and neither of the two often cited runs a change-detection benchmark (§10).

