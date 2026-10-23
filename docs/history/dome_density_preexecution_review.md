# Dome density — delta pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers for collecting the two specified configurations.** Approval is for collection only—not dome qualification, retries, additional candidates or execution authorization.

## Scope and authority

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`; paths below are repository-relative.

Inspected authority:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`, version 3, revision 1134 as read.
- `docs/reports/dome_toolpath_investigation.md`, especially **First contrasts — observed results** and **Middle-density refinement — declared before execution**.
- `docs/reports/dome_contrasts_preexecution_review.md` and `docs/reports/dome_baseline_preexecution_review.md`, reused for unchanged implementation and evidence.

Current results/refinement sections govern, not stale prospective wording elsewhere. The required contract remains **all eigenstrains before ONE free-boundary equilibrium, with no intermediate solve or physical boundary conditions**.

| Configuration | Repeats A/B/C | Accumulation cycles | Fresh output |
|---|---:|---:|---|
| `middle_n18_repeat2.json` | 2/3/3 | 8 | `run/dome/middle_n18_repeat2_v1` |
| `middle_n18_repeat4.json` | 4/3/3 | 10 | `run/dome/middle_n18_repeat4_v1` |

For each listed `NAME`, the reviewed launch specification is:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py \
  --sequence calibration/configs/dome/NAME.json \
  --output run/dome/NAME_v1
```

Two independent jobs may run concurrently in separate allocations. Both proposed output directories were absent in the inspected listing.

## Recorded identities

SHA-256 records inspected in `run/dome/checks/{density,baseline,contrast}_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `scripts/prepare_dome_density.py` | `c72f4d5d4b2420eee61c7db2a02a82d1a7ab377b885acf4ff8ef2a9f8ca8d5d5` |
| `calibration/configs/dome/middle_n18_repeat2.json` | `ad3f2730396f0b10c04f8bc54aad5e9dc4e20494a25953d6f727350953925960` |
| `calibration/configs/dome/middle_n18_repeat4.json` | `ca14804fd8a5a18e17e09f781db27f5e01fabb2e36bcbb69b6300d2072deccc1` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |
| `calibration/configs/dome/baseline_attempt209.json` | `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862` |
| `calibration/configs/dome/middle_repeat4.json` | `924ac6f2832c35c46ea92dfd12d456a6e3e0ce52df476d3854fcc3b08df5a80e` |

The retained middle-repeat execution manifest corroborates its comparator, runner and binary identities. **Hashes were inspected, not independently recomputed.** Approval is bounded to these recorded identities and reviewed contents; consequential drift requires reassessment.

## Checks performed

### Exact deltas and footprint

Manual comparison against both N10 counterparts found only:
- A `n_strips`: **10 → 18**.
- A `alpha_deg`: **9.13 → 4.594163391736709°**.

Each candidate retains its counterpart’s repeat count. Calibration, face-shared hardening, profiles, widths, lengths, placements, rotations, A/B/C order and disabled BCs remain unchanged.

`scripts/prepare_dome_density.py:10–23` correctly uses

\[
W=(N-2)L_v\tan\alpha,\qquad
\alpha_{18}=\arctan\!\left(\frac{W}{16L_v}\right).
\]

This agrees with native strip construction at `src/libshell/ZigZagGrowth.hpp:199–245`: two vertical end strips plus \(N-2\) inclined strips. Inclined-strip pitch halves from approximately **22.4996 to 11.2498 mm**, while the continuous transverse span remains **179.996461 mm**.

Width-inflated bounds remain:

| Path | u bounds, mm | v bounds, mm |
|---|---:|---:|
| A | ±93.998231 | ±74 |
| B | ±100 | 86.624745 to 149.375255 |
| C | ±100 | −149.375255 to −86.624745 |

All fit the 200 × 300 mm sheet. The footprint helper and native placement check agree. Generator assertions check equal envelopes and exact restoration to baseline after removing declared changes; file creation refuses overwrite. These are continuous-envelope checks, not proof of discrete treatment coverage.

### Execution and numerics

Inspected `scripts/run_dome.py` retains:
- Final-only accumulation, adaptive continuation disabled, zero slope feedback and multiplicative updates.
- Disabled checkpoint/control routes and seed escape.
- Exactly one accepted terminal solve, expected cycles and ordered terminal-only history checks.
- Gradient tolerance **5e−14**, positive-pivot acceptance and matching endpoint proof.
- Eight threads, **600-second cooperative native limit**, **1500-second external timeout**, and existing-output refusal.

Native terminal-only logic remains at `src/simulations/Sim_Bilayer_Growth.cpp:2487–2490,2663–2667`. No new solver route is introduced. Unchanged CPU-double, positive-stretch/SPD and ordered hardening evidence is reused from the baseline review; additional strips/repeats are not linear strain scaling.

The cooperative limit is not hard preemption. External timeout remains below the 30-minute allocation. No restart or weakened acceptance gate is covered.

### Scientific rationale and stopping

Inspected retained metrics confirm geometric qualifying coverage:
- Baseline: **33.890625%**.
- Middle repeats4: **41.927083%**.
- Sides repeats6: **30.229167%**.

The middle-repeat result/history records ten accumulation cycles and one accepted solve, gradient **7.823734545900487e−15**. Its inspected spatial PNG still shows central alternating-sign curvature, supporting the density question without relying on provisional direction mapping.

The failed length case remains unscored: `run/dome/middle_length196_v1/solver.log:3–5` records `time_cap`, `indefinite`, `accepted=0`, gradient **9.6389300283928792e−14**. Its failed execution is preserved; this review does not authorize a restart.

The two N18/N10 comparisons are appropriate matched **layout interventions**, not equal-dose or isolated-count experiments. Count, angle, overlap and shared-hardening effects remain coupled.

Outcome handling is appropriate:
- Improvement without qualification informs a separately documented refinement.
- Degradation in both matched settings triggers reassessment, not blind density escalation.
- Failed equilibrium remains numerically unresolved, without a valid curvature score.
- Apparent qualification proceeds to prescribed verification, not an immediate success claim.

## Launch-blocking findings

**None for the two specified fresh collection runs.**

## Nonblocking caveats and qualification boundaries

1. **Density versus estimator smoothing.**  
   `calibration/analysis/dome_curvature.py:9–26` fixes a 15 mm neighborhood radius, while pitch falls to approximately 11.25 mm. Reduced measured oscillation could partly reflect spatial averaging rather than elimination of finer-scale saddle structure. Retain the frozen primary estimator; use the already-required spatial, scale-sensitivity and finer-mesh checks before claiming resolved smoothness or a verified dome.

2. **Qualification remains separate from acceptance.**  
   Preserve the fixed ROI, reference-area denominator, separate coherent signs, weaker curvature ≥0.1 m⁻¹, ratio ≥0.1 and 80% coverage. Invalid fits remain failures. `calibration/analysis/analyze_dome.py:55–60` does not establish spatial vetoes or numerical convergence; resolve geometric/director classification disagreement and retain declared sensitivity reporting.

3. **Prior interpretation limits remain.**  
   Continuous footprints do not certify mesh-independent hit coverage; topology agreement is not positional correspondence. Direction mapping remains provisional. Widths 8/4 mm and the top-only center-peak profile extrapolate beyond uniform 2 mm calibration. Results are model-conditional local equilibria—not physical identification, global optimality, fabrication validation or cadence equivalence. Exported native state is not restart evidence.

## Evidence and review limitations

Directly inspected configuration/generator/runner contents, relevant native construction and cadence code, analysis helpers, identity packets, comparator metrics, middle-repeat execution/result/history/PNG, and failed-length execution/log.

Saved test logs report three baseline-input tests and two retained-fixture runner tests passing. Unchanged native and analytic fixture evidence is reused only within the earlier reviews’ stated limits.

**No simulations, tests, commands, builds, installations, edits, delegation, independent hash calculations or numerical-array recomputations were performed.** The dirty tree was preserved. No historical cumulative budget was imported. The parent retains authorization; this review does not claim runtime enforcement.