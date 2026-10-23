# Dome contrasts — independent delta pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers found for collecting these three contrasts.** This is not dome qualification, permission for additional candidates, or approval of retries/restarts.

Reviewed execution scope: one fresh, all-eigenstrains-first, free-boundary equilibrium per configuration:

| Configuration | Repeats A/B/C | Accumulation cycles | Output |
|---|---:|---:|---|
| `middle_repeat4.json` | 4/3/3 | 10 | `run/dome/middle_repeat4_v1` |
| `sides_repeat6.json` | 2/6/6 | 14 | `run/dome/sides_repeat6_v1` |
| `middle_length196.json` | 2/3/3 | 8 | `run/dome/middle_length196_v1` |

For each listed `NAME`:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py \
  --sequence calibration/configs/dome/NAME.json \
  --output run/dome/NAME_v1
```

Separate jobs may run concurrently within available scheduler resources. The parent retains execution authorization; this review does not establish runtime enforcement.

## Authority and reviewed identities

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`; paths below are repository-relative.

Authority inspected:

- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`, version 3, revision 762 as read.
- `docs/reports/dome_toolpath_investigation.md`, particularly the researcher clarification, frozen criterion, baseline results and **First sparse contrasts** section.
- `docs/reports/dome_baseline_preexecution_review.md`, reused only for unchanged implementation/evidence.
- Current explicit requirement: **all eigenstrains accumulated before ONE equilibrium, without physical boundary conditions or intermediate solves**.

The current baseline-results/contrast sections govern this review, not stale prospective wording elsewhere in the investigation.

Recorded SHA-256 identities inspected:

| Artifact | SHA-256 |
|---|---|
| `calibration/configs/dome/middle_repeat4.json` | `924ac6f2832c35c46ea92dfd12d456a6e3e0ce52df476d3854fcc3b08df5a80e` |
| `calibration/configs/dome/sides_repeat6.json` | `0b61a5bfb4273f32230aa16a7eef0b1d411d940de21f568338e9ccca1819307c` |
| `calibration/configs/dome/middle_length196.json` | `ea2a723ad9f1a2b2f69c7762709329c39e5ab66c557e5d81236e6cc96ec09db3` |
| `scripts/prepare_dome_contrasts.py` | `b5b0ab8c320fdd0d593a3fa51bf1af036a137ed56ec35a14840dbe3be9768865` |
| `calibration/analysis/analyze_dome.py` | `0265c805b1846e959c448eaad379734dcdbaa1b649875f38bfc6a6a63bb08ed4` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |
| `scripts/prepare_dome_baseline.py` | `99d4e326297dca894df1ad68d1f3e48fcc1d39a96edab940b018132852f91654` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942b3c01be28e321b839694300f7ec5419d7a9852ea70c7838ed180` |
| `calibration/configs/dome/baseline_attempt209.json` | `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `e86851a42805e1b8de08ae7d5faa2ed7bf42665a6006fc8123c9890cb10f5693` |

Sources: `run/dome/checks/contrast_identity.sha256`, `run/dome/checks/baseline_identity.sha256`; the baseline execution manifest corroborates its runner/binary/config identities.

**Identity limitation:** hashes were read and compared, not independently recomputed. `calibration/analysis/dome_direction_diagnostic.py` was inspected in full but has no identity in these packets. Approval is limited to the reviewed contents and recorded execution identities.

## Checks performed

### Configuration deltas and footprints

Manual comparison of all four JSON files found only the declared changes. Calibration coefficients, hardening, profiles, widths, strip counts, rotations, placement, A/B/C order and disabled boundary conditions remain matched. Sheet and solver settings remain in the unchanged runner.

`scripts/prepare_dome_contrasts.py:19–24` correctly preserves transverse centerline span:

\[
W=(N-2)L_v\tan\alpha,\qquad
\alpha_{\rm new}=\arctan\!\left(\frac{W}{(N-2)\,196}\right).
\]

This agrees with native strip construction in `src/libshell/ZigZagGrowth.hpp:199–220`. The resulting angle is **6.548528106136488°**.

Width-inflated bounds, in millimetres:

| Path | u bounds | v bounds |
|---|---:|---:|
| Baseline/repeat-contrast A | ±93.998231 | ±74 |
| Length-contrast A | ±93.998231 | ±102 |
| B, all contrasts | ±100 | 86.624745 to 149.375255 |
| C, all contrasts | ±100 | −149.375255 to −86.624745 |

All fit the 200 × 300 mm sheet. The length contrast closes the approximately **12.625 mm envelope gap**, producing approximately **15.375 mm envelope overlap** with each side region. These are conservative continuous envelopes, **not proof that zigzag bands fill those regions or give mesh-independent hit coverage**.

The generator deep-copies the baseline and refuses to overwrite generated files. Its use of assertions is not a launch issue for these already-inspected literal configurations.

### Execution and numerical contract

`scripts/run_dome.py:63–76` retains final-only accumulation, disabled adaptive continuation, zero slope feedback, multiplicative updates, disabled checkpoint/control routes and seed escape. Native terminal-only logic remains at `src/simulations/Sim_Bilayer_Growth.cpp:2487–2490,2663–2667`.

The repeat/length changes introduce no new solver route. Reused baseline evidence supports ordered face-shared hardening and SPD-preserving metric updates; additional repeats are **not linear strain multiplication**.

The runner retains:

- Exactly one accepted terminal solve, expected cycle count and ordered terminal-only history checks.
- Native gradient tolerance **5e−14** and positive-pivot acceptance.
- Eight CPU threads, native **600-second cooperative limit**, external **1500-second timeout**.
- Existing-output refusal and isolated execution records.

The three proposed output directories were absent in the inspected listing. No restarts are proposed or certified.

### Scientific relevance and comparator fairness

These are appropriate sparse design contrasts against the failed baseline:

- Middle repeats test improved coherence versus amplification of the central saddle pattern.
- Side repeats test useful cross-region bending versus stronger side anisotropy.
- Middle extension tests continuity versus adverse overlap/coupling.

The baseline scalar fields and inspected spatial map independently support these questions; launch does not depend on trusting the untested direction mapping.

Comparisons are matched **layout interventions**, not equal-dose experiments. Repeats add different treatment amounts in differently sized regions. Length extension jointly changes length, angle, spatial profile placement, overlap and subsequent shared-hardening history. It cannot identify an isolated angle or gap-closure effect.

Ranking by frozen geometric coherent-sign coverage is appropriate. Retain all outcomes; improvement can motivate a separately documented refinement, disagreement remains unresolved, and failure to converge is not evidence that a dome is impossible. No automatic combined candidate or wider grid is covered.

### Analysis interpretation

`calibration/analysis/dome_curvature.py:26–72` retains the frozen 15 mm estimator, support/conditioning rules and fundamental-form generalized eigenproblem. Coverage uses the fixed reference-area ROI denominator, finite values, the 0.1 m⁻¹ weaker-curvature floor, ratio ≥0.1 and **separate coherent signs**. Invalid fits cannot improve coverage.

`calibration/analysis/analyze_dome.py:27–52` checks finite/symmetric/SPD target metrics and matching face topology. Its reconstruction from \(H,K\) uses a scale-aware discriminant and stable product-based smaller root. Inspection of the baseline VTP header and `src/libshell/WriteVTK.hpp` confirms **Float64 curvature fields**, consistent with the roundoff allowance; geometric curvature instead uses native positions.

The direction diagnostic implements the mathematically appropriate least-squares pullback
\[
F=D_xD_m^{-1},\qquad d_{\rm material}\propto F^+d_{\rm physical}.
\]
This static check does not substitute for its missing synthetic mapping test.

## Evidence reused and limitations

Inspected retained baseline evidence:

- `run/dome/baseline_v1/execution.json`
- `run/dome/baseline_v1/sequence_result.json`
- `run/dome/baseline_v1/sequence_history.csv`
- `run/dome/baseline_v1/sequence_convergence.csv`

These record **eight accumulated cycles, one full-load solve, zero retries**, gradient **4.701548684439926e−14** and positive pivots.

Inspected analysis evidence:

- `run/dome/baseline_analysis_v1/metrics.json`: geometry **33.890625%**, director **17.015625%**, both below 80%; valid ROI area 100%.
- `run/dome/baseline_analysis_v1/directions.log`.
- `run/dome/baseline_analysis_v1/baseline_shape_curvature_v2.png` and caption: shallow full shape and spatially alternating central curvature.
- Recorded native-state identity: `b6d913cb0b1ab82325699cc45a72480e95ff1b32a73d27b38b6729d6a23f017f`.

Also inspected existing test implementations and logs: three baseline-input tests and two retained-fixture runner tests passed. Analytic curvature evidence is reused with the prior review’s limits: smooth fixtures, not mesh convergence; the rigid-pose assertion includes NumPy’s default relative tolerance. The coarse fixture analysis has zero valid geometric support and is ingestion evidence only.

**No commands, tests, simulations, builds, installations, hash calculations or array recomputations were performed in this review.** The dirty working tree was preserved.

## Launch-blocking findings

**None for the three specified collection runs.**

## Nonblocking caveats and qualification gates

1. **Analysis status is not dome qualification.**  
   `calibration/analysis/analyze_dome.py:55–60` compares pass/fail booleans and explicitly omits spatial vetoes and numerical verification. Before qualification, inspect coherent-sign consistency between estimators, spatial fields and full shape; resolve classification disagreement; complete the prescribed tighter-tolerance/finer-mesh checks. Report the declared 70/90% coverage sensitivities as well as the implemented curvature-floor sensitivities.

2. **Direction-based mechanism remains provisional.**  
   `calibration/analysis/dome_direction_diagnostic.py:16–24` lacks a synthetic mapping test and eigenvalue-separation/residual diagnostics. Do not interpret near-degenerate directions causally. Its statistics are conditional on valid mapped area; retain that fraction when comparing candidates. Lines 43–45 define **baseline spatial regions**, not candidate treatment masks—especially for the length extension. These limitations do not block the independently motivated scalar/spatial contrasts.

3. **Endpoint audit has a limited scope.**  
   `calibration/analysis/analyze_dome.py:39–41` checks topology, not VTP/native positional correspondence. The inspected analysis records do not independently certify every serialized field. Preserve endpoint provenance and avoid claiming a stronger audit than performed.

4. **Model-conditional conclusions only.**  
   Fixed Attempt209 parameters are effective calibrated inputs, not new physical identification. The 8/4 mm widths and top-only center-peak profile extrapolate beyond uniform 2 mm calibration. Positive pivots support local numerical acceptance, not global optimality. No fabrication validity, cadence equivalence or resumability is established.

Consequential changes to code, configurations, physics or scoring require reassessment. No historical cumulative budget is imported.