# Dome balance — delta pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers for collecting the two specified configurations.** This approval covers collection, not dome qualification, additional candidates, retries or execution authorization.

## Scope and authority

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`; paths below are repository-relative.

Inspected:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`, version 3, revision 1434.
- `docs/reports/dome_toolpath_investigation.md`: **Density results**, **Estimator-scale diagnostic**, **Lower-repeat balance contrasts**, and frozen protocol.
- Reused unchanged evidence and limitations from `docs/reports/dome_density_preexecution_review.md` and `docs/reports/dome_contrasts_preexecution_review.md`.

The required contract remains **all eigenstrains accumulated before ONE free-boundary equilibrium, without intermediate solves**.

| Configuration under `calibration/configs/dome/` | Repeats A/B/C | Cycles | Fresh output |
|---|---:|---:|---|
| `middle_n18_repeat1.json` | 1/3/3 | 7 | `run/dome/middle_n18_repeat1_v1` |
| `middle_n18_sides1.json` | 2/1/1 | 4 | `run/dome/middle_n18_sides1_v1` |

For each listed `NAME`, the reviewed command is:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py --sequence calibration/configs/dome/NAME.json \
  --output run/dome/NAME_v1
```

Separate allocations may run concurrently. Both proposed output directories were absent from the inspected listing. Native cooperative limit remains 600 seconds; external timeout is 1500 seconds. No restart of the failed length case is covered.

## Recorded identities

Inspected `run/dome/checks/{balance,density,baseline}_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `scripts/prepare_dome_balance.py` | `74f4abda54f272e4050fb8db05a2506bcfb70c50855850474714905fc37dc185` |
| `calibration/configs/dome/middle_n18_repeat1.json` | `9dc9dbc255ae44b09163d9d0d9500f80fa29f2663e3171730fd8b1dccb67d4b1` |
| `calibration/configs/dome/middle_n18_sides1.json` | `cf8468661d0f0cf8400b7a7925a12853ad1e953f90ddd4c8fcb2aa03d2ad6185` |
| `calibration/configs/dome/middle_n18_repeat2.json` | `ad3f2730396f0b10c04f8bc54aad5e9dc4e20494a25953d6f727350953925960` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

`run/dome/middle_n18_repeat2_v1/execution.json` corroborates the comparator, runner and binary identities. **Hashes were inspected, not independently recomputed.** Approval is bounded to these identities and reviewed contents.

## Delta checks

Manual comparison of the complete candidate JSONs against N18-repeat2 found only:
- `calibration/configs/dome/middle_n18_repeat1.json:28`: A repeat **2 → 1**.
- `calibration/configs/dome/middle_n18_sides1.json:53,81`: B/C repeats **3 → 1**.

Growth, orthotropy, hardening, profiles, widths, geometry, placements, ordering and disabled boundary conditions remain identical.

`scripts/prepare_dome_balance.py:8–20` deep-copies the comparator, changes only repeats, checks footprints, asserts exact restoration after restoring repeats, and refuses overwrite. Repeat counts do not enter the footprint calculation; unchanged geometry therefore retains the previously reviewed within-sheet envelopes. Assertion execution was reported in the investigation, not rerun here.

Inspected `scripts/run_dome.py` retains final-only accumulation, disabled adaptive continuation, zero feedback, multiplicative updates, terminal-only history checks and exactly one accepted solve. No new mathematical or solver path is introduced; unchanged numerical/SPD evidence is reused within the prior reviews’ limits.

## Scientific relevance and decisions

Directly inspected comparator result and analysis records confirm eight accumulated cycles, one accepted solve, gradient **1.8478921932612982e−14**, and **52.817708%** qualifying geometric coverage.

`run/dome/checks/scale_sensitivity_v1.json` confirms:
- N18-repeat2 improvement over N10 persists at 10/15/20 mm.
- N18-repeat4 performs worse than repeat2.
- All cases fail 70/80/90% coverage thresholds.
- At 15 mm, N18-repeat2 saddle area is **39.744792%**, versus baseline **31.156250%**.

These findings justify testing treatment balance rather than escalating density. Reduced repeats could improve sign coherence or instead lose necessary curvature magnitude. Compare each candidate directly with retained N18-repeat2 using frozen 15 mm coherent-sign coverage; saddle fraction, magnitude and 10/20 mm sensitivity remain explanatory diagnostics, not replacement scores.

Improvement without qualification can inform a separately documented refinement. If neither improves, do not extrapolate toward zero repeats; reassess geometric allocation. Failed equilibrium remains unscored and unresolved. Apparent qualification requires the existing verification gates.

## Launch-blocking findings

**None for these two fresh collection runs.**

## Nonblocking caveats and limits

- These are nonlinear, unequal-dose layout interventions—not isolated physical-mechanism identification. Ordered hardening and coupled equilibrium preclude interpreting repeat changes as linear strain scaling.
- Retain every existing interpretation gate: frozen ROI/reference-area denominator, separate coherent signs, invalid-fit failures, curvature floor and ratio, spatial/full-shape vetoes, director/geometry disagreement handling, sensitivity reporting, and tighter-tolerance/finer-mesh verification before qualification. Scale robustness alone does not establish resolved smoothness.
- Prior width/profile calibration-transfer, provisional direction-mapping, endpoint-provenance and local-equilibrium limitations remain. No physical fabrication validity, global optimality, cadence equivalence or resumability is established.

No simulations, tests, builds, edits, delegation, independent hash calculations or array recomputations were performed. The existing dirty tree was preserved; unrelated changes were not certified. No historical cumulative quota was imported. The parent retains authorization; this report does not claim runtime enforcement.