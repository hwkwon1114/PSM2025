# Dome baseline — independent pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers found for collecting this one baseline.** This does **not** approve dome qualification, subsequent candidates, per-cycle experiments, retries or restarts.

Approval is limited to the reviewed identities and command:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py \
  --sequence calibration/configs/dome/baseline_attempt209.json \
  --output run/dome/baseline_v1
```

The parent retains execution authorization. This review does not establish runtime enforcement.

## Scope, authority and identities

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`. Paths below are repository-relative unless absolute.

Authority inspected:

- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`: version 3, revision 426 as read.
- `docs/reports/dome_toolpath_investigation.md`.
- `docs/reports/dome_toolpath_design_review.md`.
- Current explicit researcher requirement: **all eigenstrains before one free-boundary equilibrium; no per-cycle experiments**. This supersedes the older design review’s cadence-verification suggestion.

Reviewed implementation includes the generator, runner, curvature helper, all three requested Python test files, connected sequence/growth/native-solver source, and existing build/provenance evidence.

Recorded SHA-256 identities from `run/dome/checks/baseline_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `scripts/prepare_dome_baseline.py` | `99d4e326297dca894df1ad68d1f3e48fcc1d39a96edab940b018132852f91654` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942b3c01be28e321b839694300f7ec5419d7a9852ea70c7838ed180` |
| `calibration/configs/dome/baseline_attempt209.json` | `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `e86851a42805e1b8de08ae7d5faa2ed7bf42665a6006fc8123c9890cb10f5693` |

Input provenance inspected:

- Collaborator: `calibration/data/processed/zigzag_doublecurv.json`.
- Calibration: `run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00/sequence.json`.
- Cross-check: `run/run9_transfer_expanded/evaluations/attempt_0209/case_00/sequence.json`.

Their recorded hashes in the investigation are respectively:

```text
5ad61c0a70d624bff88092d43f8e88d5ad5693150a4530de6b901a1cf9029d1b
ae885273b4da727a1815c4104d41328a09f96749e478cbf89dbe2ff58df92bb8
e3fe1bb10eaf4b45858b5902567dd7c6428da814642b1ea1951ed1d1ed4e86b5
```

**Identity limitation:** I inspected these records but did not independently compute hashes. The three Python tests are outside the six-file hash packet; their current contents and available test evidence were reviewed.

The binary/source pair matches `run/two_gripper_comparison/checks/final_only_preparation/preexecution_identity.json` and the prior final-only reviews. Its saved `build.log` records successful compilation/linking; `unit_tests.log` records 33 passing native/checkpoint tests. This supports reuse of unchanged solver evidence—not historical execution permission.

## Checks and findings

### Calibration, footprint and scientific purpose

The generated baseline preserves A/B/C order, repeats **2/3/3**, widths, rotations and profiles. It replaces legacy growth/shared orthotropy with the matching Run6/Run9 coefficients:

- `gtop=0.00028835116227798787`
- `gbot=1.4587994001079146e-05`
- `ortho_top=-0.3989157893255133`
- `ortho_bottom=0.960306323937336`
- Hardening `beta=0.2231435513142097`, floor zero, face-shared plastic hits.

Only the approved side adaptation is made: `Lv=196 mm`, shifts `±118 mm`. The generator’s width-inflated placement calculation agrees with the native check in `src/libshell/ZigZagGrowth.hpp:663–735`. A fits approximately within `u=±94`, `v=±74 mm`; the side union fits within `u=±100`, `v=±149.3753 mm`.

This establishes a **minimally adapted collaborator-layout baseline**, not an exact reproduction. Side footprints touch the sheet’s lateral boundaries; continuous validity does not establish mesh-independent treatment coverage.

The baseline is relevant to testing whether fixed effective growth parameters can produce sustained geometric double curvature through layout changes. An accepted but nonqualifying shape is a negative result for this layout; failed equilibrium or conflicting curvature evidence is unresolved, not proof of impossibility.

### Ordered accumulation and one equilibrium

Source inspection supports the requested mechanics:

- `src/simulations/Sim_Bilayer_Growth.cpp:2366–2668` applies ordered material hits and advances shared history through every repeat.
- Hardening is \(q(n)=e^{-\beta n}=0.8^n\), not linear scaling by repeat count.
- `src/libshell/GrowthHelper.hpp:740–834` implements
  \[
  \bar a_{\mathrm{new}}=T^\mathsf{T}\bar a_{\mathrm{old}}T,\qquad
  T=D_m^{-1}R\operatorname{diag}(1+g_1,1+g_2)R^\mathsf{T}D_m,
  \]
  with triangle-conditioning, positive-stretch and updated-SPD checks.
- The fixed inputs are finite with positive incremental stretches. Multiplicative congruence preserves SPD for the valid material mesh.
- The runner explicitly disables slope feedback, adaptive continuation, seed escape and both checkpointed control routes.
- Skipped cycles bypass optimization. The terminal step installs the exact complete targets; with adaptive continuation disabled, failed acceptance throws rather than subdividing or retrying.

Thus the reviewed configuration requests **eight accumulation cycles and one terminal full-load equilibrium**.

### Free boundary, acceptance and safeguards

`src/simulations/Sim_Bilayer_Growth.cpp:4890–4900` selects the unconstrained rectangle; JSON boundary conditions are disabled. Physical clamps/release branches are not entered. Native rigid-mode removal is numerical gauge handling, not a physical clamp.

The project uses CPU `double` (`src/libshell/common.hpp:37`). Native acceptance requires genuine `gradient_target` termination, finite independently evaluated state/energy/gradient, gradient norm ≤ **5e−14**, and `positive_pivots` endpoint evidence (`src/libshell/NativeHybridShellSolver.hpp:339–358`).

`scripts/run_dome.py:19–39` checks completion, expected cycles, exactly one terminal solve, native acceptance, matching state proof, final-mesh existence, ordered history, nonzero hits and terminal-only solve intent. It therefore does not mistake history labels alone for acceptance.

The runner refuses existing output directories, records command/input/binary identities, checks binary stability, caps threads, and uses a **1500-second process timeout**. The native **600-second** limit is cooperative, with separate verification allowance; it is not a hard preemption boundary.

The inspected `/gpfs/home/pxl1051/bin/srun-here` supports the stated allocation. `.srun_here` activates `DHD`; the command explicitly selects the requested Python executable. Scheduler allocation and runtime dependency resolution were not independently exercised. `run/dome/baseline_v1` was absent in the directory listing inspected.

## Evidence reused and its limits

- `run/dome/checks/baseline_tests.log`: **3 passing tests**, covering approved changes/source immutability/eight cycles, original side overflow and invalid dimensions.
- `run/dome/checks/runner_tests.log`: **2 passing tests**, including retained-fixture acceptance and rejection of corrupted cycle count, acceptance, history and state proof.
- `run/dome/fixture_three_paths_v1/`: inspected execution manifest, history, convergence record, native proof, result and solver log. Job **4287432** records:
  - Three ordered accumulation cycles; history solve flags `[0,0,1]`.
  - One accepted full-load solve, `lambda_trial=1`, no retry.
  - Gradient **4.741639361270636e−15**, `positive_pivots`.
  - The same reviewed binary and baseline-source hashes.

The fixture used earlier runner hash `cd410fbd1d40e76edc0600827b7ba413ad9b5b96dd3fe3dac875989624ae71dc`. Its binary-integration evidence remains useful; current checker changes are covered by retained-artifact regression tests. Its coarse mesh, reduced growth, one repeat per path and five warmup attempts do **not** establish production convergence or footprint coverage.

Analytic job **4287009**, reported in the investigation, passed two curvature tests. I inspected their implementation, but no separate analytic execution log was supplied in `run/dome/checks/`. The fixtures cover plane, sphere, cylinder, saddle, rigid transformation, invalid area and mixed signs.

No commands, tests, builds, simulations, numerical recomputations or independent hash calculations were executed during this review.

## Launch-blocking findings

**None unresolved for this single baseline collection.**

## Nonblocking caveats and interpretation gates

1. **B1 is repaired at protocol/helper level, not end-to-end scoring level.**  
   `calibration/analysis/dome_curvature.py:12–72` uses a fixed 15 mm material neighborhood, Euclidean tangent chart, conditioned quadratic fit and a correct symmetric generalized eigenproblem. Coverage retains invalid area in the fixed ROI denominator and separates positive/negative fractions. The protocol makes geometry primary and disagreement unresolved.

   However, native/VTP ingestion, director-curvature reconstruction, disagreement handling and spatial/shape vetoes are not integrated. **Neither `coverage()['qualifies']` nor runner status `accepted` establishes a successful dome.** Complete and check that analysis before making a qualification claim or using curvature directions for refinement.

2. **The runner is not a complete serialized-endpoint audit.**  
   `scripts/run_dome.py:31–39` compares proof objects and checks artifact existence; it does not recompute finite/SPD target checks from exported arrays, compare VTP/native topology and geometry, or certify full per-face history. Solver-side safeguards support this launch, but these stronger artifact claims remain unverified. Audit the saved endpoint before scientific interpretation.

3. **Analytic evidence is limited.**  
   Smooth fixtures do not establish robustness to folds, discontinuous treatment or noisy meshes. Additionally, `calibration/tests/test_dome_curvature.py:41` leaves NumPy’s default relative tolerance enabled; its check is not strictly an absolute-only `1e−8 m⁻¹` bound. Describe that accurately or tighten it when updating analysis tests.

4. **Preserve model and execution boundaries.**  
   Widths 8/4 mm and A’s top-only center-peak profile extrapolate beyond uniform 2 mm calibration. Positive pivots establish local numerical evidence, not global optimality or physical validation. No fabrication, recalibration, cadence equivalence, optimizer resumability or broader search is certified. Retain failures; do not silently restart or weaken gates.

The existing dirty tree was preserved. Consequential changes to the reviewed slice require reassessment; no historical cumulative budget was imported.