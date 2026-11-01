# Shared-footprint metric repair: independent re-review

## Verdict: `APPROVED_WITH_CAVEATS`

**Both remaining geometry blockers are repaired.** The stale-summary inconsistency is also resolved against the current result files. No launch-blocking findings remain within this review slice.

Two nonblocking qualifications remain: the baseline differs slightly from the number quoted in the request, and the new minimum-area test does not isolate that filter from edge rejection.

## Scope, authority and identities

Read-only inspection covered the requested geometry functions, regression tests, rescoring script, summary/result records, backups and execution logs. No repository edits, tests, rescoring or Slurm submissions were performed.

Authority/context inspected:

- The current task’s explicit repair requirements.
- `docs/calibration/bayesian_calibration/shared_footprint_metric_astra_review.md`, the preceding review.
- `docs/calibration/bayesian_calibration/shared_footprint_metric_review.md`.
- `docs/calibration/bayesian_calibration/convention_corrected_investigation.md`.

The current task specifies the spacing-relative rule; at nominal spacing \(h=2\) mm, its edge threshold remains 3.5 mm.

Current abbreviated blob identities reported by the diff tool:

| File | Identity |
|---|---|
| `calibration/ewcal/geometry.py` | `8f941c0` |
| `calibration/tests/test_direction_frame.py` | `b08c88c` |
| `calibration/ewcal/material_frame.py` | `1cf8afa` |

The material-frame identity matches the preceding review. `calibration/analysis/rescore_shared_footprint_metric.py` remains untracked; this review covers its inspected contents, particularly lines 18–23 and 148–162. No independent hash manifest or source-to-log binding was established.

## Findings resolved

### 1. Unconditional reference-footprint coverage — cleared

**Location:** `calibration/ewcal/geometry.py:164–165`.

The implementation now checks:

```python
if np.any(rf(q_xy[:, 0], q_xy[:, 1]) < 0):
    raise ValueError('Shared footprint points lie outside predicted mesh footprint')
```

This evaluates **all queries**, after either reference-cache path and before either interpolation branch. It is outside the deformed-interpolation exception handler, so the required rejection cannot be swallowed by that fallback.

`test_expanded_deformed_mesh_outside_reference_fails_closed` at `calibration/tests/test_direction_frame.py:165–181` directly exercises the former bypass:

- Reference footprint: `[-1,1]²`.
- Expanded deformed footprint: `[-2,2]²`.
- Query `(1.5,0)` must raise despite being covered by the deformed mesh.
- A contracted mesh still accepts reference-covered queries through barycentric fallback, checking the resulting coordinate `0.4` and height `1.25`.

This is an appropriate regression for the original defect. The exact exception text is verified in source; the test asserts the exception type.

### 2. Spacing-relative edge and minimum-area filters — cleared

**Location:** `calibration/ewcal/geometry.py:190–216`.

Verified:

- Defaults are `max_edge_factor=1.75`, `min_area_factor=1e-6`.
- Nearest-neighbor distances are restricted to finite, positive values before taking their median \(h\); an empty set raises `ValueError`.
- Triangle area is correctly computed as half the absolute planar cross product.
- Retention requires both:
  \[
  \max(e_{01},e_{12},e_{20})\le1.75h,\qquad A>10^{-6}h^2.
  \]
- The fixed 3.5 mm floor is absent.
- If no triangles survive, the function raises:
  `ValueError('No valid shared-footprint triangles within edge/area threshold')`.

`test_finer_grid_hole_and_min_area_filter` at `calibration/tests/test_direction_frame.py:183–197` is present. Its 1 mm grid with a missing interior point checks maximum retained edge `≤1.75 mm`, detecting the former fixed-floor defect. The minimum-area portion has the nonblocking limitation below.

### 3. Summary synchronization and backup — cleared against current records

**Location:** `calibration/analysis/rescore_shared_footprint_metric.py:148–162`.

The script backs up an existing summary, refreshes `baseline_attempt1` from attempt 1, and refreshes `best` from the lowest-objective valid result. `backup_once` at lines 18–23 leaves an existing backup untouched.

Manual comparison found the complete summary entries consistent with their corresponding result records, including parameters, case RMS and directional diagnostics:

| Summary entry | Result record | Joint RMS, mm |
|---|---|---:|
| `best` | `run/joint_run6_run10/results/a0032.json` | 0.6400855307665548 |
| `baseline_attempt1` | `run/joint_run6_run10/results/a0001.json` | 0.6744068587819062 |

The baseline candidate result also contains `0.6744068587819062`.

`run/joint_run6_run10/summary.json.pre_metric_revision.bak` exists and is readable. It preserves the earlier best and baseline objectives, respectively `0.6400321873757964` and `0.6743422965495391`. Backups for both inspected top-level result files are also present. Historical byte-for-byte backup integrity was not independently established.

## Execution evidence inspected

`run/joint_stage2/test_shared_footprint_metric.log` reports:

> `18 passed, 13 warnings in 66.61s (0:01:06)`

The source contains twelve direction-frame tests and six joint-driver tests, consistent with that total. Warnings concern deprecated VTK `SetCells`.

`run/joint_stage2/rescore_shared_footprint_metric.log` reports:

| Campaign | Valid rescored / total |
|---|---:|
| `run/joint_run6_run10` | 29 / 48 |
| `run/joint_stage2/arm_A` | 10 / 18 |
| `run/joint_stage2/arm_B` | 5 / 13 |

Its rounded Stage-1 scores agree with the inspected records. These are supplied execution logs, not independently reproduced or cryptographically version-bound runs.

## Launch-blocking findings

**None for the reviewed repair slice.**

## Nonblocking caveats

1. **Requested baseline literal differs from inspected data.**  
   `run/joint_run6_run10/summary.json:81` and its matching result contain `0.6744068587819062`, not the requested `0.6744068567360283`: a difference of approximately \(2.05\times10^{-9}\) mm. Therefore, exact agreement with the requested literal is **not verified**. The summary itself is synchronized; no scoring or ranking defect is demonstrated. Use the observed value when documenting this snapshot. The discrepancy’s origin was not established.

2. **Minimum-area regression is not independently discriminating.**  
   At `calibration/tests/test_direction_frame.py:193–197`, the tiny triangle also has an edge of approximately \(\sqrt2\), while \(1.75h\approx1.2374\). Edge filtering alone therefore removes it. The test would not detect deletion of the area predicate. Optional improvement: relax the edge factor for this subcase, for example to `2.1`, and assert the spacing-normalized area threshold. The production predicate itself is correct by inspection.

3. **Evidence and execution limits.**  
   No scores were recomputed, no exhaustive campaign reconciliation was performed, and active-worker state, concurrent writes and resumability were not checked. Approval does not establish safe concurrent migration or authorize campaign continuation.

## Scientific claim boundaries

The unchanged scoring path retains actual simulation connectivity, flat-reference rigid alignment and equal-case joint RMS. These repairs address domain validity and scan triangulation; they do not establish a universal zero-error theorem for arbitrary folded surfaces or an argument-exchange-symmetric mathematical metric.

Lower discrepancy demonstrates model-conditional geometric fit, not physical parameter identification or optimization convergence. The inspected best improves Run 10 while worsening Run 6 relative to baseline; report that tradeoff. Rescoring does not retroactively make historical search decisions decisions under the revised objective.

Approval applies only to this inspected slice. It grants no execution permission, budget extension, restart authorization or new-parameter Run-9 release.