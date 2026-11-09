# BO-RUN2-RUN3 — Independent Bayesian Calibration of Physical Specimens Run 2 and Run 3

**Status:** APPROVED (2026-11-01, researcher confirmed "yes lets do that").
**Owner:** researcher. **Drafted:** 2026-11-01.

## Question

Do independent Bayesian calibrations directly optimizing Run 2 (`run9`) and Run 3 (`run10`) converge to the same physical parameter basin ($g_{\text{top}}, g_{\text{bot}}, \text{ortho\_top}, \text{ortho\_bottom}$) as Run 1 (`run6`)?

Specifically:
1. Does each independently manufactured and scanned specimen recover $g_{\text{top}} \approx (2.4 \pm 0.6) \times 10^{-4}$ and $\text{ortho\_top} < 0$?
2. What is the resulting complete $3 \times 3$ cross-calibration matrix (calibration on Specimen $i \to$ RMS error on Specimen $j$)?

## Decision driven

- **Consistent Basin** (Gates G1–G4 pass): Parameter universality across distinct physical specimens is confirmed. The manuscript can report that the material and growth parameters are process-intrinsic, not overfitted to Specimen 1.
- **Discrepant Basin** (Any gate G2–G4 fails): Distinguish specimen-to-specimen printing variability or geometric sensitivity from numerical optimization error.

## Formulas

- **F1. Active Parameters.** Four parameters: $g_{\text{top}}$ (growth top, log-scale), $g_{\text{bot}}$ (growth bottom, log-scale), $\text{ortho\_top}$ (orthotropy top, linear), and $\text{ortho\_bottom}$ (orthotropy bottom, linear).
- **F2. Fixed Parameters.** Hardening exponent $\beta = 0.22314355$, hardening floor $= 0.0$.
- **F3. Search Bounds.**
  - $g_{\text{top}} \in [5 \times 10^{-5},\, 0.01]$
  - $g_{\text{bot}} \in [1 \times 10^{-5},\, 0.001]$
  - $\text{ortho\_top} \in [-1.0,\, 1.0]$
  - $\text{ortho\_bottom} \in [-1.0,\, 1.0]$
- **F4. Acquisition & Surrogate.**
  - Initial Design: 24-point Latin Hypercube Sample (seed `20261030`, maximin separation $d \ge 0.05$ in normalized space).
  - Surrogate: Matérn-5/2 Gaussian Process with input warping and success-aware failure classification.
  - Acquisition: Batch $q = 4$ Log Expected Improvement (`qLogEI`).
- **F5. Stopping Rule.** Beginning at Wave 16 (evaluation 64), terminate early if the relative improvement over the preceding 6 waves (24 evaluations) is $< 0.5\%$. Hard cap at 31 waves (124 evaluations).
- **F6. Cross-Transfer Evaluation.** Upon completion, evaluate the optimal parameters on the two non-target specimens without tuning.

## Known-answer checks

- **K1. Parameter roundtrip.** $\text{encode}(\text{decode}(u)) = u$ to within $10^{-7}$ across the unit hypercube $[0, 1]^4$.
- **K2. Target case fidelity.** Generated configurations for `run9` and `run10` correctly load their respective toolpath parameters ($\alpha = -1.0^\circ$ and $\alpha = -1.5^\circ$) and point clouds (`run9.npz`, `run10.npz`).
- **K3. LHS deterministic seed.** Initial 24 design proposals match the identical design evaluated in `run1_clean`.

## Metrics and gates

- **G1 (Objective Quality):** Optimal calibration RMS for Run 2 and Run 3 achieves $< 0.80$ mm.
- **G2 (Growth Consistency):** Calibrated $g_{\text{top}}$ lies in $[1.8 \times 10^{-4},\, 3.5 \times 10^{-4}]$ (within $\approx \pm 25\%$ of Run 1's $2.40 \times 10^{-4}$).
- **G3 (Orthotropy Direction):** Top orthotropy is strictly negative ($\text{ortho\_top} < -0.5$).
- **G4 (Transfer Robustness):** All off-diagonal elements in the $3 \times 3$ transfer matrix achieve sub-millimeter RMS ($< 0.85$ mm).

## Compute plan

- Run 2: Slurm job `run2_clean` on Quest `normal` partition (1 node, 32 CPUs, 64 GB, 24-hour limit).
- Run 3: Slurm job `run3_clean` on Quest `normal` partition (1 node, 32 CPUs, 64 GB, 24-hour limit).
- Both jobs run concurrently.
- Environment: `OMP_NUM_THREADS=8`, `BLIS_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Outputs: `run/run2_clean_24lhs/` and `run/run3_clean_24lhs/`.
