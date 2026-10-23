# Dome length168 — delta pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers for the single fresh collection run.** This is not dome qualification, execution authorization, retry approval or approval of additional candidates.

## Scope and authority

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`; paths below are repository-relative.

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`, version 3, revision 1669.
- `docs/reports/dome_toolpath_investigation.md`, especially **Balance results**, **Intermediate-length refinement**, and the frozen protocol.
- `docs/reports/dome_density_preexecution_review.md` and `docs/reports/dome_balance_preexecution_review.md`, reusing unchanged numerical/implementation evidence within their stated limits.

Reviewed command, **not executed**:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py \
  --sequence calibration/configs/dome/middle_n18_length168.json \
  --output run/dome/middle_n18_length168_v1
```

The output directory was absent in the inspected listing. Contract: eight accumulation cycles, **all eigenstrains before ONE free-boundary equilibrium**, no intermediate solves or retry.

## Recorded identities

Inspected `run/dome/checks/{length168,density,baseline}_identity.sha256` and comparator `run/dome/middle_n18_repeat2_v1/execution.json`.

| Artifact | SHA-256 |
|---|---|
| `scripts/prepare_dome_length168.py` | `e5e100b9768b75e455de3367b3869e893d139ef58497a8b793626ff066380609` |
| `calibration/configs/dome/middle_n18_length168.json` | `61d0c6084c85995cce31405ae1239ac9fac16605e34cd492d5f66edf2a4f4ae8` |
| `calibration/configs/dome/middle_n18_repeat2.json` | `ad3f2730396f0b10c04f8bc54aad5e9dc4e20494a25953d6f727350953925960` |
| `scripts/prepare_dome_baseline.py` | `99d4e326297dca894df1ad68d1f3e48fcc1d39a96edab940b018132852f91654` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

**These are inspected recorded hashes, not independently recomputed hashes.** Approval is bounded to these identities and reviewed contents; consequential drift requires reassessment.

## Checks performed

### Delta and footprints

Complete manual comparison against N18-repeat2 found only `calibration/configs/dome/middle_n18_length168.json:31–32` changed:
- A length: **140 → 168 mm**.
- A angle: **4.594163391736709 → 3.830976264231982°**.

Repeats remain **2/3/3**. Calibration, hardening, profiles, widths, counts, ordering, placement, rotations and disabled boundary conditions remain identical.

`scripts/prepare_dome_length168.py:12–21` correctly preserves
\[
W=(N-2)L_v\tan\alpha
\]
by solving for the new angle, checks within-sheet envelopes, and asserts exact restoration after reversing the two changes. Output creation refuses overwrite. Assertion execution is reported in the investigation; it was not rerun here.

The formula agrees with native construction at `src/libshell/ZigZagGrowth.hpp:199–245`. Transverse span remains approximately **179.996461 mm**, with inclined-strip pitch **11.249779 mm**.

| Path | u bounds, mm | v bounds, mm |
|---|---:|---:|
| A | ±93.998231 | ±88 |
| B | ±100 | 86.624745 to 149.375255 |
| C | ±100 | −149.375255 to −86.624745 |

All envelopes fit the 200 × 300 mm sheet. The former **12.624745 mm envelope gap** becomes **1.375255 mm overlap**, versus **15.375255 mm** for the failed Lv196 contrast.

### Rationale and outcome handling

Direct inspection of retained `metrics.json` and `sequence_result.json` records corroborates:
- N18-repeat2: accepted, **52.817708%** geometric coverage.
- Reduced A repeats: accepted, **42.026042%**.
- Reduced side repeats: accepted, **39.880208%**.

Thus spatial redistribution is a justified next question rather than continued repeat reduction. The proposed N18/Lv168 configuration is distinct from the failed N10/Lv196 case. Its log, `run/dome/middle_length196_v1/solver.log:3–5`, records `time_cap`, indefinite curvature and `accepted=0`; that case remains unscored and unresolved.

Improvement should trigger spatial interpretation and prescribed verification; degradation should trigger reassessment, not blind overlap escalation. Failed equilibrium supplies no valid shape score.

### Execution and numerics

Inspected `scripts/run_dome.py` retains final-only accumulation, disabled adaptive continuation, zero feedback, multiplicative updates, terminal-only history checks and exactly one accepted solve. Acceptance requires gradient ≤**5e−14**, positive pivots and matching endpoint proof.

Resources remain eight CPUs/32 GB, native cooperative **600 seconds**, external **1500 seconds**, within the 30-minute allocation. No new solver path or numerical method is introduced. Unchanged numerical/SPD evidence is reused, not re-established.

## Launch-blocking findings

**None for this single specified collection run.**

## Nonblocking caveats and qualification limits

- **Envelope overlap is not treatment continuity.** It does not establish discrete face-hit overlap or mesh-independent coverage. Smaller overlap also does not guarantee convergence.
- **Coupled intervention, not isolated mechanism identification.** Length changes orientation, treated area, normalized profile placement and overlap/hardening. The native profile uses normalized along-segment position (`src/libshell/ZigZagGrowth.hpp:151–168`), so unchanged profile parameters do not mean an unchanged spatial growth field.
- **Collection acceptance is not dome qualification.** Retain the fixed central ROI, reference-area denominator, separate coherent signs, invalid-fit failures, weaker curvature ≥0.1 m⁻¹, ratio ≥0.1 and 80% coverage. Preserve sensitivity reporting, spatial/full-shape vetoes, director/geometry disagreement handling, and tighter-tolerance/finer-mesh verification.
- The 15 mm estimator remains larger than the approximately 11.25 mm strip pitch; scale robustness does not establish resolved smoothness. Prior width/profile calibration-transfer, provisional direction-mapping and endpoint-provenance limits remain.
- Results are exploratory, model-conditional local equilibria—not physical identification, global optimality, fabrication validation, cadence equivalence or restart evidence.

No simulations, tests, builds, installations, edits, delegation, hash recomputation or numerical-array recomputation were performed. Existing dirty-tree changes were preserved and are not certified. No historical cumulative quota was imported. This review does not claim runtime enforcement.