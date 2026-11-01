# Per-hit Voce-type decay: bounded literature and calibration note

**Reviewed 2026-10-25.** Decision note, not a systematic review. Direct-process **quantitative coverage is partial**: relevant laser-forming abstracts were recovered, but their full texts were inaccessible. No experiments, Zotero writes, code changes or campaign-bound changes were made.

## Bottom line

- **The functional form is defensible as an effective attenuation law, not as an identified Voce material law.** Repeated-pass laser forming can show decreasing bend increments, but geometry, thermal history and microstructure also contribute. Peening supplies explicit exponential saturation analogies, including a linear-tail extension.
- **No transferable, experimentally validated interval for this model’s per-plastic-hit `beta` or `floor` was established.** Strain-based Voce constants, peening-time constants and coverage-per-pass constants have different meanings and cannot be pooled into a “typical beta range.”
- **[PROPOSAL, future campaign only]** Use a fixed reference horizon `N_ref=100` hits and explore **`z=beta*N_ref ∈ [1,30]` logarithmically**, i.e. **`beta ∈ [0.01,0.30] hit⁻¹`**. Concentrate exploitation around the successful existing region, without treating it as a posterior interval. Use **`floor ∈ [0,0.10]` as the main search region, explicitly including `floor=0`**, and retain a smaller, feasibility-aware exploration allocation over **`(0.10,0.50]`** and outside the proposed beta core. These are computationally motivated design choices, **not physical admissibility bounds**. The model’s mathematical domain remains `beta≥0`, `floor∈[0,1]`.
- Separate numerical success from shape mismatch. A successful-point mean floor near 0.06 and failures at high floor do **not** establish a material upper bound or prove that floor causes failure.

## 1. What is, and is not, a Voce parameter

The project convention is

\[
q(n)=f+(1-f)e^{-\beta n},\qquad \Delta g_n=Aq(n),
\]

where `n` counts **local plastic hits**, `f=floor` is dimensionless, and `beta` has inverse-hit semantics. Its exponential argument must be dimensionless. A nominal pass that does not create a counted plastic hit need not advance this history. Neither face-sharing nor this hit-count rule is validated by the sources below.

The usual stress–plastic-strain Voce form is

\[
\sigma(p)=\sigma_0+Q(1-e^{-bp})
=\sigma_{\rm sat}-(\sigma_{\rm sat}-\sigma_0)e^{-bp}.
\]

`σ0`, `Q=σsat−σ0` and `σsat` have stress units; `b` is inverse plastic strain, with strain conventionally a **fraction**, not a percentage. Another parameterization uses the initial hardening slope `θ0=bQ`, so the exponent is `−θ0 p/Q`. Authors may call the exponential coefficient `n`; that symbol is **not necessarily pass number**. In multiaxial isotropic-hardening usage, `p` commonly denotes accumulated equivalent plastic strain; the inspected fitted example below is specifically uniaxial plastic strain, not an independent verification of a multiaxial implementation.

**[INFERENCE]** Only under an assumed constant strain increment `Δp` per hit would `p=nΔp` give `beta=bΔp`. Otherwise the exponent is `−bΣΔp_j`. Even with that mapping, the exponentially decaying **stress deficit or hardening slope** is not automatically proportional to the remaining growth increment. A load/temperature-to-plastic-increment relationship is still needed. Likewise, converting cycles to hits requires the actual count rule: if exactly two hits occur per cycle, the coefficient per cycle would be twice the coefficient per hit.

## 2. Concise evidence table

“Direct” means evidence about the cited process, **not validation of this raster/bilayer model**. `PFT` = selected primary full-text passages inspected; `A` = indexed abstract only; `M` = metadata only. Numerical recommendations below are our proposals, not numbers reported by these studies.

| Source / citation | Inspected evidence and numbers | What transfers; what does not |
|---|---|---|
| **Edwardson et al. (2006), [Geometrical influences on multi-pass laser forming](https://doi.org/10.1088/0022-3727/39/2/021)**. DOI `10.1088/0022-3727/39/2/021`. **A**. | Abstract explicitly identifies decreasing bend angle **per pass** after many irradiations along one track and reports empirical support for a geometrical explanation. | **Direct qualitative support** for diminishing increments. Does not isolate strain hardening or supply a checked exponential coefficient/floor; no fitted range extracted. |
| **Cheng & Yao (2002), [Microstructure Integrated Modeling of Multiscan Laser Forming](https://doi.org/10.1115/1.1459088)**. DOI `10.1115/1.1459088`. **A**. | Abstract identifies strain hardening, dynamic recrystallization and phase transformation; incorporating microstructure improves low-carbon-steel geometry/property predictions. | **Direct caution** against a universal pass-count-only material interpretation. No transferable beta/floor identified. The 2001 conference version was not counted as independent evidence. |
| **Nikolov, Thomsen & Kristiansen (2021), [Hardening in laser forming under the temperature gradient mechanism](https://doi.org/10.1088/1757-899x/1135/1/012006)**. DOI `10.1088/1757-899X/1135/1/012006`. **A**. | Abstract studies 1 and 3 mm stainless steel, reports bend-rate changes with parameter sets, significant bend-arc-length influence and bottom-side hardness increase. | **Direct contextual evidence**: bend rate depends on processing and geometry. It does not establish monotone decay at fixed conditions or a universal rate for a 0.77 mm sheet. |
| **Zhang et al. (2022), [Investigation on Strain Hardening and Failure in Notched Tension Specimens of Cold Rolled Ti6Al4V Titanium Alloy](https://doi.org/10.3390/ma15103429)**. DOI `10.3390/ma15103429`. **PFT**, §4.1, Tables 1–2, Eq. (8), Fig. 10 discussion; native publisher XML via Europe PMC `PMC9143126`. | `σ=C1+C2[1−exp(−C3 εp)]`; fitted `C1=1108.564`, `C2=259.898`, `C3=6.333`. Stress convention is MPa, so first two parameters are MPa and `C3` is inverse strain fraction. Pure Voce underestimates the extrapolated response; a hybrid law is preferred. | **Constitutive analogy**, with one real fitted example—not a typical range. `6.333` cannot be copied into a per-hit model. A saturating stress is not a saturating deformation increment. |
| **Kirk (2005), [Computer-Based Saturation Curve Analysis](https://www.shotpeener.com/library/pdf/2005078.pdf)**, *The Shot Peener* 19(4). **PFT**, printed pp.18,20, Eqs. (1)–(3), (7)–(8), visually checked. Technical article; no DOI established. | Almen arc-height families `h=a(1−exp(−bt))`, `h=a(1−exp(−bt^c))`, and `h=a(1−exp(−bt^c))+dt`. Illustrative fitted example `a=9.86`, `b=0.37`; time/height units are not established in the inspected example, so no per-pass conversion is made. | **Strong structural analogy**. For `c=1`, `h′=ab exp(−bt)+d`, giving a normalized residual rate `d/(ab+d)`. No fitted universal `d`/floor. The Almen 10% doubling-time criterion gives `bT=ln(10)≈2.303`, not complete saturation or a plastic-hit count. |
| **Kirk (2009), [Shot Peening Coverage: Prediction and Control](https://www.shotpeener.com/library/pdf/2009014.pdf)**, *The Shot Peener* 23(2). **PFT**, printed p.30, Eqs. (9)–(12), visually checked. Technical article; no DOI established. | `C%=100[1−exp(−An)]`, for identical passes under constant peening conditions. Example: first-pass coverage fraction 0.58 gives `A≈0.87` **per pass**. This is an illustrative calculation, not a material coefficient range. | **Coverage analogy only**: newly covered area decays with a zero floor, while repeated plastic impacts can continue on already covered surface. Stream passes ≠ local plastic hits. |
| **Al-Hassani (1981), [Mechanical Aspects of Residual Stress Development in Shot Peening](https://www.shotpeener.com/library/pdf/1981050.pdf)**. **PFT**, scanned printed p.595, “Aspects of Repeated Loading,” visually checked. No DOI established. | Describes repeated-impact shakedown as subsequent residual-plus-contact stresses producing entirely elastic response; plastic flow can continue above the shakedown limit. No calibrated exponential/floor. | **Mechanism analogy**, not a parameter bound. Vanishing plastic increments resemble elastic shakedown; persistent increments do not. Positive accumulated equivalent plastic strain alone does not prove net directional ratcheting. |
| **Gardner et al. (2014), [Bayesian Optimization with Inequality Constraints](https://proceedings.mlr.press/v32/gardner14.html)**, ICML/PMLR 32(2):937–945. **PFT**, PDF p.3, §§3.1–3.2, visually checked. No DOI established. | Derives constrained EI as `EI(x) P(feasible|x)` under conditional independence of objective and constraint models; handles multiple constraints. | **Optimization-method transfer**. Its continuously observed constraints are richer than a solver success/failure flag; use a classifier or suitable likelihood for the latter. No guarantee of failure-free sampling. |
| **Gelbart, Snoek & Adams (2014), [Bayesian Optimization with Unknown Constraints](https://arxiv.org/abs/1403.5607)**, inspected arXiv **v1**, 22 March 2014. **PFT**, pp.4–7, §§1.4–3.2; p.6 equations visually checked. | Explicitly discusses simulation crashes/failure to terminate; independent objective/constraint models, binomial observations (one trial = GP classification), bounded-runtime modeling, and weighted EI, Eqs. (7)–(9). | **Closest methodological match** for learned solver feasibility. Classifier uncertainty is not physical safety or material admissibility; independence assumptions must be stated. |

**What numerical ranges are actually supported?** The inspected sources provide **examples on different scales**, not a commensurate range: strain-Voce `b=6.333` per strain fraction; illustrative coverage decay `A≈0.87` per identical stream pass; illustrative Almen `b=0.37` per unspecified time unit. **No empirical floor interval for local plastic-hit growth was recovered.** Zero-tail and linear-tail laws both exist as model forms. No primary local-hit law for incremental sheet forming was verified in this bounded review; that route remains unresolved rather than evidence of absence.

## 3. Horizon, cumulative dose and identifiability

**[ANALYTICAL INFERENCE, not a simulation result]** For `N` increments indexed `n=0,…,N−1`, define

\[
S_N=\sum_{n=0}^{N-1}q(n)
=Nf+(1-f)\frac{1-e^{-\beta N}}{1-e^{-\beta}}.
\]

The limit at `beta=0` is `S_N=N`. If the code applies the first increment at `n=1`, the exponential sum gains a factor `e^(-beta)`; confirm indexing before using these totals operationally. These formulas describe **imposed scalar accumulation**, not final shell shape, which is nonlinear and path dependent.

- **`f` is a residual increment fraction, not a final-shape saturation fraction.** At `f=0`, `S_N` has a finite limit. At `f>0`, it eventually grows linearly as `Nf`. A tiny floor can matter over 100 hits. A zero floor gives asymptotically vanishing prescribed increments, not by itself finite-time elastic shakedown.
- The transient fraction is `(q(n)−f)/(1−f)=exp(−beta*n)` for `f<1`. Its half-life is `ln(2)/beta`; 95% removal takes `ln(20)/beta`. Thus `beta*N=1,3,5` leave approximately **37%, 5%, 0.67%** of the transient. This dimensionless interpretation is more defensible than borrowing a stress-hardening constant.
- At the baseline `beta=0.2231`, the half-life is **3.11 hits**, 95% removal takes **13.43 hits**, and `beta*N≈12.94` at 58 hits and `22.31` at 100. With zero floor, even 50 hits are far into the saturated-dose regime. The existing upper bound `beta=1` instead removes 95% in about **3 hits**; it is not mathematically invalid, but spends search capacity on very short transients.
- At `N=100`, `beta=0.2231`, `S_N≈5.00, 5.95, 9.75, 14.50, 49.65` for `f=0, .01, .05, .10, .47`. Holding amplitude fixed, even `f=.10` nearly triples the total versus zero floor. **This is a plausible numerical-load explanation, not proof of why the solver fails.** Amplitudes and other parameters also change.

### Are beta and floor separately identifiable from final shape?

**Not generically; also not proved impossible for this full-field solver.** The following are analytical confounding mechanisms:

1. Early in the history, `q≈1−(1−f)beta*n`; the leading slope identifies mainly `(1−f)beta`.
2. With `f≈0` and `beta*N≫1`, a scalar final response is governed mainly by `A/(1−exp(−beta))`. Amplitude and beta compensate. Two long-horizon totals at 58 and 100 hits then carry almost the same saturation information.
3. With a nonzero tail, long-horizon totals approximately identify an intercept `A(1−f)/(1−exp(−beta))` and slope `Af`, not all three of `A,beta,f` separately. Two scalar totals cannot generally resolve three free parameters.
4. At `f=1`, beta is exactly irrelevant; at `beta=0`, floor is exactly irrelevant. A separate no-decay model avoids spending many samples on equivalent parameterizations.

Spatially varying hit counts, transient equilibrium/path effects, multiple faces and full-field residuals may break these degeneracies. Conversely, freely varying top/bottom amplitudes and anisotropy can add compensations. **The best BO point and GP objective uncertainty are not parameter-identification evidence.** Future checks should use parameter profiles or posterior correlations and, if new measurements become authorized, early/intermediate shapes spanning roughly 1–15 hits as well as late shapes. No such checks were run here.

## 4. Recommended bounds and initial design

**[PROPOSAL]** For an explicitly authorized *future* search:

| Purpose | Recommended parameterization | Rationale / limitation |
|---|---|---|
| Main decay exploration | Fix `N_ref=100`; log-uniform `z=beta*N_ref∈[1,30]`, so `beta∈[.01,.30] hit⁻¹`. | Covers slow decay through fast saturation and includes observed best beta `.19–.24` with margin. Across actual 50–100-hit histories this gives `beta*N∈[.5,30]`. Do **not** rescale beta separately for each run. The upper endpoint is a design choice informed by the current fit, not derivable from horizon alone. |
| Main floor exploration | `f∈[0,.10]`, with a deliberate **atom/subdesign at exactly zero**. | Preserves the current boundary optimum and explores tails large enough to change 100-hit dose substantially. Not a credible interval or a physical bound. |
| Guard against premature restriction | Smaller, predeclared allocation to `f∈(.10,.50]`, slower/faster beta outside the core, and the no-decay alternative only if scientifically useful. | Learn feasibility near the observed region rather than declare `.47` a universal threshold. Do not systematically delete `f>.50` from the physical hypothesis space solely because current solves fail. |

Latin hypercube sampling stratifies **marginals**, not an unknown feasible region. The classic reference is McKay, Beckman & Conover (1979), [Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code](https://doi.org/10.1080/00401706.1979.10489755), *Technometrics* 21(2):239–245; DOI metadata verified, full methods not re-read here. Its existence does not justify uniform effort over an empirically failure-heavy box.

Use a known-successful baseline, a space-filling design within the declared core, explicit zero-floor points, and a smaller space-filling/boundary-probing design outside it. This is a **stratified mixture design**, not an untouched LHS over the original box. Sample beta in log space; a continuous LHS almost surely misses `f=0`. Choose any allocation percentages as budget decisions, not literature constants. Track cumulative-dose summaries such as `A*S_N` to interpret interactions, without pretending they fully predict shape or convergence.

### Failure-aware BO

- Model the valid shape objective using accepted solves, and model **success under the frozen solver/tolerance/time budget** using all relevant outcomes. Use the full six-parameter input, not floor alone. The current success-only floor distribution is selected by both BO and solver feasibility.
- A starting acquisition is expected feasible improvement, `E[(J_best−J)_+ I_success]`; it factorizes into `EI * P(success)` only with appropriate independence/modeling assumptions. For batch/log-EI implementations, use a consistent constrained acquisition rather than arbitrarily multiplying a log acquisition by a probability.
- Distinguish equilibrium nonconvergence, timeout, registration failure and infrastructure failure. A timeout is a failure of the declared resource budget and, for a runtime model, a **right-censored runtime**, not a measured large shape error. Unrelated infrastructure incidents should not train a physical/parameter feasibility boundary.
- Require both runs to succeed, but do not label Run 10 as failed if Run 6 aborted before Run 10 was attempted. Joint feasibility can be modeled directly; multiplying per-case probabilities requires a justified dependence assumption.
- The existing `1.5 × worst valid J` failure penalty is a transparent heuristic, **not an observed objective**. It mixes feasibility with mismatch and can distort GP scale/smoothness; changing the worst valid value can also change the implied penalty. Preserve raw outcomes and do not interpret the resulting surrogate as a likelihood for material parameters.
- Feasibility-weighted BO can still propose failed points. Hard probability thresholds or trust-region restrictions are additional policies, not safety guarantees. With only dozens of six-dimensional evaluations, classifier calibration near the boundary is uncertain; retain limited boundary exploration.

**No change to the active 60-evaluation protocol is implied.** Its convention note explicitly forbids silent bound/recipe changes during the run. The 42/60 success-floor summary and best beta region in the request are **user-reported, not independently audited**; the local convention note corroborates the model, search box and numerical-failure caveat. Solver failure is not evidence of physical instability.

## 5. Remaining uncertainty and review provenance

Unknowns include the process-specific per-hit strain map; whether geometry/thermal history can be represented by count alone; a nonzero persistent tail; identifiability from actual spatial residuals; and the solver’s full-vector feasibility boundary. The observed `floor≈0` may be a useful reduced model, but does not prove zero late plasticity in the physical sheet.

Both literature providers were attempted without date, citation or OA restrictions. Consensus was blocked by its monthly quota. FastTrack searches required shorter-query repairs to avoid off-field laser results; relevant primary-process abstracts and source URLs were retained. Primary XML/PDF checks support the constitutive, peening and BO equations; direct laser-forming full-text attempts met publisher challenges/403s or unavailable repository copies. No access controls were bypassed. No exhaustive forward-citation sweep or incremental-sheet-forming hit-law search was completed. Optional extra examples were omitted once they would not change the recommendation.

Local recovery/evidence directory: `/home/pxl1051/.pi/agent/literature-review-runs/review-b5CUqB/` (`report.md`, `evidence.md`, `C1.json`, `F1.json`–`F12.json`, primary PDFs/extractions, and child `review-3Q7N3k/`). Two delegated routes could not start; the coordinator completed the essential laser-abstract and BO-method checks directly and integrated the surviving child evidence. This note is the only project file authored by the review.

**Provider access notice (verbatim):**
> You've used all 30 searches this month; resets on November 1st. Upgrade to Consensus Pro to get 500 searches per month, return 20 results per search instead of 10, and include more data like study design and key takeaways for every result. https://consensus.app/pricing/
