# Dome BO campaign — independent pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers found for the reviewed campaign slice.**

This review covers up to **48 new attempts**, with at most two simultaneous solves, through `scripts/run_dome_bo.sbatch`. It does not authorize execution, extension, restart, recipe changes or numerical-verification runs. A screening pass is **not a verified dome**.

## Scope and authority

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`.

Authority inspected:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — version 3, revision 2204 as read.
- `docs/reports/dome_toolpath_investigation.md`, especially **BO stage authorization** and **BO implementation and pre-execution contract**.
- Current supplied researcher authorization: six bounded geometry variables, 48 new attempts, fixed Attempt209 recipe, all strains before one free-boundary equilibrium.

Reviewed completely:
- `scripts/dome_bo_domain.py`
- `scripts/dome_bo_propose.py`
- `scripts/dome_bo_campaign.py`
- `scripts/run_dome_bo.sbatch`
- `calibration/tests/test_dome_bo_domain.py`
- `calibration/tests/test_dome_bo_propose.py`
- `calibration/tests/test_dome_bo_campaign.py`

Also inspected the baseline configuration, footprint helper, runner, analyzer, curvature helper, relevant native strip/placement/failure code, and all five retained warm-start configurations and execution records.

Unchanged evidence was reused from the baseline, density, length168, side-density and uniform-profile pre-execution reviews. Historical execution permissions and material-calibration BO observations/budgets were **not** reused.

`run/dome/bo_stage1` was absent in the inspected directory listing.

## Recorded identities

From `run/dome/checks/bo_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `scripts/dome_bo_domain.py` | `196f9f3930c00dbc62775d59c2ae77d56bb52365af2d1aca808594dc77681265` |
| `scripts/dome_bo_propose.py` | `970456fc0f5071552795ee44c64e7df1b430740707504af1c65d7c9249850003` |
| `scripts/dome_bo_campaign.py` | `732353ce992164f26678987bfdfdb4941cdd2cd301c8753ee6317018ee47f2ef` |
| `scripts/run_dome_bo.sbatch` | `d1db2ecc88c6d5c3574173ee6db865da0c863ebf1c457a09b9194812013c1edd` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `calibration/analysis/analyze_dome.py` | `0265c805b1846e959c448eaad379734dcdbaa1b649875f38bfc6a6a63bb08ed4` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942b3c01be28e321b839694300f7ec5419d7a9852ea70c7838ed180` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

Additional reused identities from `run/dome/checks/baseline_identity.sha256`:
- `scripts/prepare_dome_baseline.py`: `99d4e326297dca894df1ad68d1f3e48fcc1d39a96edab940b018132852f91654`
- `calibration/configs/dome/baseline_attempt209.json`: `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862`

**These are inspected recorded hashes, not independently recomputed hashes.** The three new test files were inspected but are not included in the supplied identity packet. Approval is bounded to the reviewed contents and identities; consequential changes require reassessment.

## Checks performed

### Domain, mixed integers and fixed recipe

`scripts/dome_bo_domain.py` correctly implements:
- Middle bounds: length 100–196 mm, angle 1–15°, integer count 6–26.
- Tied side bounds: length 120–196 mm, angle 0.5–5°, integer count 4–18.
- Integer decoding before footprint filtering and canonical normalized re-encoding.
- Exact fixed-recipe compatibility by reconstructing the candidate from the baseline and comparing the complete configuration.

The width-inflated local envelope uses
\[
W=(N-2)L_v\tan\alpha,\qquad
x_{\max}=(W+w)/2,\qquad y_{\max}=(L_v+w)/2,
\]
then applies the fixed rotations and placements. This agrees with `src/libshell/ZigZagGrowth.hpp:199–245,663–735`, including odd counts. The Python tolerance of \(10^{-9}\) mm agrees with the native \(10^{-12}\) m placement tolerance for this sheet.

Only the six approved fields change. Profiles, widths, repeats **2/3/3**, placements, calibration, hardening, ordering and disabled physical BCs remain fixed.

### Warm starts, provenance and comparator fairness

Direct inspection corroborates these retained inputs:

| Run under `run/dome/` | Geometric coverage | Treatment |
|---|---:|---|
| `baseline_v1` | 0.338906250 | Accepted training observation |
| `middle_n18_repeat2_v1` | 0.528177083 | Accepted training observation |
| `middle_n18_length168_v1` | 0.525989583 | Accepted training observation |
| `middle_n18_sides_n14_v1` | 0.264531250 | Accepted training observation |
| `middle_length196_v1` | — | Failed coordinate, exclusion only |

Evidence: each run’s `sequence.json` and `execution.json`, and the corresponding `*_analysis_v1/metrics.json` for accepted runs.

All five execution manifests record the reviewed runner and binary, production mesh/settings, and `fixture:false`. Manual configuration comparisons found only permitted geometry differences. The failed run’s `solver.log:3–5` records `time_cap`, indefinite curvature and rejected acceptance.

No changed-repeat or uniform-profile result enters the GP. No old material-calibration observation is imported. Retained results are adaptive design observations, not independent validation samples.

### GP and acquisition numerics

`scripts/dome_bo_propose.py` implements the declared finite-pool heuristic:
- Scrambled six-dimensional Sobol pool of \(2^{14}\) points.
- Feasibility filtering after integer decoding.
- Exact exclusion of previously attempted physical coordinates.
- First eight new points selected by feasible Sobol order.
- Thereafter, a CPU-double `SingleTaskGP` using accepted coverage only, output standardization and supplied variance \(10^{-6}\).
- Analytic `LogExpectedImprovement`, maximizing observed geometric coverage.
- Greedy second selection separated by at least 0.1 in normalized Euclidean distance.

The standardized model returns predictions in the original outcome scale; using original-scale `y.max()` for `best_f` is consistent. The positive observation variance regularizes the covariance; it is not measured physical uncertainty. Chunked acquisition evaluation and finite-score checks are appropriate.

The pair selection is **not joint qEI** and does not condition the second selection on a fantasized first outcome. Documentation correctly limits that claim.

### Scientific scoring and stopping

The objective remains relevant to constructive, **model-conditional shape design**. It does not identify a physical mechanism or recalibrate material parameters.

The controller maximizes the larger of the two separate coherent-sign geometric area fractions. It does not pool convex and concave patches or substitute director coverage. The unchanged estimator retains invalid fits in the fixed ROI denominator and applies the frozen magnitude, ratio and coverage thresholds.

`scripts/dome_bo_campaign.py:123–124` stops on geometric screening qualification regardless of `analysis_status`. Consequently, geometry/director disagreement cannot silently trigger additional optimization after a geometry pass.

Outcome decisions are appropriate:
- Screening pass: stop proposals and assess prescribed verification.
- Improved but nonqualifying coverage: retain as exploratory evidence.
- Failed equilibrium: retain unscored, not as a bad-shape observation.
- Exhaustion or repeated numerical failure: report the limitation; no automatic extension.

The broad search can exploit estimator smoothing or mesh-specific treatment patterns. Spatial, scale, director and tighter-tolerance/finer-mesh verification therefore remain essential.

### Slurm resources, timeouts and attempt accounting

Static inspection supports the requested allocation:

| Activity | Request |
|---|---|
| Batch | One node, 16 CPUs, 64 GB, normal partition, 12 hours |
| Each solve step | Exclusive, one task, 8 CPUs, 32 GB |
| Each analysis step | Exclusive, one task, 1 CPU, 4 GB |
| Proposal | Separate process, one computational thread, 300-second timeout |

The solve step’s explicit CPU/memory requests avoid inheriting the entire batch request. Exclusive steps are intended to use disjoint resources; two solve requests fit the allocation. Analysis replaces its own completed solve rather than adding a third solve.

Local configuration inspection corroborates consumable core/memory scheduling and configured cgroup constraints:
- `/etc/slurm/slurm.conf:46,65,107–108`
- `/etc/slurm/cgroup.conf`
- `/etc/slurm/cli_filter.lua:45–48`, which bypasses submission-policy checks for `srun` inside an allocation.

This is configuration evidence, **not an exercised allocation or enforcement claim**.

The runner restores eight solver threads while keeping BLAS single-threaded. Native 600-second and external 1500-second limits remain unchanged. The controller adds a 1700-second solve-step timeout and 600-second analysis timeout.

The 45-minute admission reserve exceeds the nominal longest wave path:
\[
300+1700+600=2600\text{ seconds}<2700\text{ seconds}.
\]
This assumes the reviewed 12-hour batch script without wall-time overrides.

Accounting and failure handling are sound:
- Twenty-four waves of two reservations bound new attempts by 48.
- Both reservations exist before either evaluation starts.
- Failed launched attempts consume slots.
- Both outcomes are retained, even when one raises an infrastructure exception.
- Two consecutive fully failed numerical waves stop the campaign.
- Infrastructure/analysis exceptions stop further proposals.
- Existing-output refusal prevents automatic replay.
- No resume implementation or restart claim is present.

The retained native failure message used for classification corresponds to an actual failed corrector branch at `src/simulations/Sim_Bilayer_Growth.cpp:2795–2806`. Unrecognized failures stop rather than becoming fabricated coverage values.

## Launch-blocking findings

**None identified for the specified, unchanged launch slice.**

## Nonblocking caveats and targeted improvements

1. **Subprocess integration is not covered by the controller mocks.**  
   `calibration/tests/test_dome_bo_campaign.py:19–39` replaces `evaluate()` completely. Thus the reported tests do not exercise real step allocation, timeout cancellation, analyzer invocation or numerical-versus-infrastructure classification. Focused tests around `scripts/dome_bo_campaign.py:46–65` would improve regression coverage. A bounded no-solver two-step check would provide operational evidence, but this review does not impose a new testing campaign.

2. **Identity checks occur before waves, not after them.**  
   `scripts/dome_bo_campaign.py:75–85` snapshots current files and checks them before proposals. It does not compare startup identities to the review packet, nor recheck after the terminal wave. Parent prelaunch identity verification remains necessary. A post-wave check would make terminal-result provenance stronger. This is not continuous drift enforcement.

3. **Warm-start provenance is linked rather than fully copied.**  
   `scripts/dome_bo_campaign.py:26–43` preserves source paths, sequence hashes and extracted scores, but not metrics/execution hashes in `warm_start.json`. The inspected retained records support this launch. Adding those hashes and the existing native-state identity would improve later auditability without changing observations or requiring recomputation.

4. **Timeout cleanup was not exercised.**  
   The outer timeouts act on `srun`; this review did not establish immediate remote-step cleanup following forced termination of that launcher. The nested solver timeout, stop-on-error behavior and finite batch allocation remain safeguards. Interrupted outputs require assessment, not replay.

5. **Durability is not resumability.**  
   Exclusive file creation prevents overwrite, but JSON writes are not transactional crash checkpoints. An interruption can leave partial records. The explicit no-restart policy is appropriate.

These are evidence limitations or optional hardening, not unresolved launch gates under the reviewed contract.

## Evidence and approval limits

Supplied execution attestations, not independently rerun here:
- Domain: three passing tests.
- Mocked controller/warm starts: four passing tests.
- Proposal fixture: one passing test, job **4322816**, **11.679 seconds**, four observations, one CPU/4 GB, no solver.
- Batch script: Bash syntax check passed.
- Environment: torch 2.6.0, BoTorch 0.16.1, SciPy 1.16.3.

The fixture supports small-data proposal correctness, not full-campaign runtime or larger-training-set behavior. The 300-second proposal cap bounds failure in that path.

No commands, tests, simulations, builds, installations, hash recomputations, edits or delegation were performed by this reviewer. Existing working-tree changes were preserved.

Approval retains all prior limitations: calibration transfer beyond uniform 2 mm bands, estimator smoothing, discretized hit coverage, local rather than global equilibrium, and geometry/director disagreement. BO predictive fit is neither physical identification nor scientific uncertainty. No global optimum, impossibility, fabrication validity, cadence equivalence or verified dome is established.