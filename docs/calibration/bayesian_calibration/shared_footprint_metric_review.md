# Shared-footprint metric review

## Verdict: `CHANGES_REQUESTED`

**Scope:** Read-only review of the five requested source files, the investigation record, Stage-2 initial results/backups, selected Stage-1 records, and Run-9 isolation.

**Authority:** `docs/calibration/bayesian_calibration/convention_corrected_investigation.md`, particularly the joint-calibration and researcher-directed metric-revision sections. The supervisor clarified that rescoring the **existing Attempt-209 Run-9 baseline** was authorized; evaluating new joint-calibrated parameters on Run 9 remains withheld.

## Findings

### Launch/continuation blockers

1. **Interpolation does not preserve the predicted triangular surface.**  
   `calibration/ewcal/geometry.py:164` and `calibration/ewcal/material_frame.py:58` use `LinearNDInterpolator` without the simulation connectivity. This constructs a new Delaunay triangulation: its interpolated points need not lie on the actual simulation triangles used for forward distances. Consequently, the claimed general zero-floor property is not established.

   **Concrete mathematical counterexample:** opposite diagonals of a 2 mm square with vertex heights given by \(z=xy\) produce center heights of 0 and 2 mm. Retriangulation therefore changes the represented surface, even with unchanged vertices.

   **Clearance:** Interpolate using the actual simulation faces. Validate identical-surface scoring with a nonseparable curved patch and deliberately mismatched Delaunay/mesh diagonals. The existing test at `calibration/tests/test_direction_frame.py:67–79` uses a separable parabola and the same interpolator to construct its scan; it does not expose this defect.

2. **Incomplete coverage silently changes the objective and mishandles the rigid transform.**  
   `calibration/ewcal/geometry.py:165–171` switches from shared-footprint samples to **all simulation vertices** if even one interpolated height is nonfinite. This restores the unequal-domain comparison the revision intended to remove and can make scores depend discontinuously on coverage.

   In addition, `calibration/ewcal/material_frame.py:63–64` does not apply the returned Kabsch translation correctly. `rigid_fit(source, target)` returns \(R,t\) for `source @ R + t`; the implementation instead subtracts `c_sim` before rotating and adds the reference mean. Full centered coverage largely masks this error, but partial coverage generally has a nonzero reference centroid.

   **Clearance:** Apply the returned rigid transforms correctly and use an explicit common-footprint coverage policy—without silent full-mesh fallback. Add production-spacing coverage/hole and asymmetric/tilted identical-surface controls.

3. **The checked-in migration is incomplete, and the worker transition is unverified.**  
   `calibration/analysis/rescore_shared_footprint_metric.py:65–119` does not update `results/a*.json`, although these are the records consumed by `calibration/bo/run_joint_run6_run10.py:89–94`. It also leaves `cases[].case_rms` unchanged when replacing case metrics; its top-level JSONL update changes only selected RMS fields. The entry point at lines 122–130 omits Stage-2 arms.

   Existing records show that an additional migration occurred, but the script alone cannot reproduce their consistency. At inspection, neither arm had `results/a0012.json`; actual completion and scoring-module identity were unverified. An expected timeout is not evidence that an old worker cannot emit an old-metric success.

   **Clearance:** Make migration synchronize objective, case RMS, and directional diagnostics across candidate records and BO input records, preserving original backups. Explicitly cover both Stage-2 arms. Before continuation, reconcile all completed workers—including `a0012`—and verify that subsequent GP fitting and successful evaluations use one corrected metric version. No solver rerun is inherently required for rescoring saved successful evaluations.

## Checks and observed evidence

### Alignment and geometry

- **No scan-to-simulation ICP or residual-minimizing registration occurs in `score_material_frame`.** Both rigid fits target flat reference coordinates. However, the interpolation and partial-coverage defects above prevent approval of the stronger common-gauge/zero-floor claim.
- Reverse weights correctly distribute projected triangle area by thirds and normalize the result. Forward distances use uniform weights, consistent with the mixed-quadrature equation in `docs/calibration/bayesian_calibration/calibration_manuscript_draft.tex:49–58`.
- This is a **bidirectional discrepancy**, not necessarily an argument-exchange-symmetric mathematical metric: the two quadratures and surface representations differ.
- At nominal 2 mm spacing, the edge cutoff is 3.5 mm, retaining 2.83 mm diagonals and rejecting 4 mm bridges. However, `geometry.py:138–140` is not a universal fixed-3.5-mm rule: it expands with estimated spacing and retains unfiltered triangles if every triangle fails. Those generic limitations require documentation or fail-closed handling; they were not demonstrated to affect the current complete scan grid.

### Stage-2 data consistency

The four successful initial points have matching stored objective/case-RMS values in both arms and their candidate records:

| Attempt | Joint RMS | Run 6 RMS | Run 10 RMS |
|---|---:|---:|---:|
| a0001 | 0.674342297 | 0.626313230 | 0.719170915 |
| a0004 | 5.518326657 | 6.146489602 | 4.808796498 |
| a0006 | 4.190593276 | 4.860837499 | 3.390339633 |
| a0008 | 17.457704291 | 21.311035364 | 12.465257717 |

Units: mm. These are **stored results**, not independently recomputed scores.

- Both arms contain `.pre_metric_revision.bak` files for these four top-level results, candidate results, and both cases’ comparison/prediction artifacts.
- Attempts `a0002`, `a0003`, `a0005`, and `a0007` are invalid and have no successful objective requiring conversion.
- Baseline case differences from `BASELINE_CASE_RMS` are approximately 0.000482 mm and 0.000000925 mm, within the existing 0.001 mm gate.
- Selected Stage-1 `a0001` and `a0032` records also reflect revised scores.
- Backup existence and readable historical RMS values were checked; binary backup integrity and every saved diagnostic were not exhaustively verified.

### Run 9

**Run 9 was not untouched.** Its existing baseline results, comparison and prediction artifacts were rescored, with backups present. This matches the clarified authorization.

Static inspection of `run_joint_run6_run10.py` and `run_convention_corrected.py` found that joint configuration loads only Run 6 and Run 10. Merely defining Run-9 paths in the imported module does not read them. No new-parameter Run-9 validation is approved by this review.

## File identities

The supervisor supplied the following SHA-256 manifest. **These hashes were not independently recomputed or bound to the inspected bytes.**

| File | Supplied SHA-256 |
|---|---|
| `calibration/ewcal/geometry.py` | `37e14c7a12e818868e5ae5233988a67a46099624810187b3babaf9242ba439e0` |
| `calibration/ewcal/material_frame.py` | `7ef27c8e89d5105e74c2c23af1ab9a31c0b46cf76f4873bce7b718d91c354244` |
| `calibration/tests/test_direction_frame.py` | `fd96a3daa5d2962784e5373005eb1948d571a4c4ea2010774c50da10cc8a6fca` |
| `calibration/bo/run_joint_run6_run10.py` | `de75e7f06727cb6218cc08629de0d7cd47dab43976d01b159152639b9bea7663` |
| `calibration/analysis/rescore_shared_footprint_metric.py` | `bdb3103a5c020b869a2adc5e7af8c9a8cce6677d95dd2e062fef03a9bda4ce0c` |
| `docs/calibration/bayesian_calibration/convention_corrected_investigation.md` | `9491e5302d08a6a2565a23c00fef644218d03e76b03caceb52cccd093874b5ad` |

## Evidence limits and nonblocking caveats

- Review used source/data reads, searches, directory listings, diffs and mathematical inspection. **No tests, rescoring, scheduler queries or hash commands were executed.**
- The supplied “17 passed” evidence was withdrawn after inspection established **7 direction-frame tests and 6 joint-driver tests**. Fresh, version-matched test evidence is needed.
- `simulation_transform` currently records only the initial direction-frame transform, not the additional leveling transform.
- Rescored Stage-1 rankings do not demonstrate convergence under the revised objective. Fitted parameters remain effective/model-conditional; lower geometric discrepancy does not establish physical identification.
- This review grants no execution permission, budget extension, restart authorization or held-out validation release.