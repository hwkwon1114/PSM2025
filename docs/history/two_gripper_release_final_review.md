# Final independent review: release constraints and forming restraint

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved completion-blocking findings for either reviewed comparison.** The saved evidence supports the report’s bounded conclusions for Run 6 and Run 9.

This approval covers the completed comparisons—not new simulations, a cadence experiment, global uniqueness, exact equality, or identification of physical gripper effects.

## 1. Scope and authority

Reviewed authority:

- `.pi/goals/active_goal_2026101623411509_mvbwsnko-2nl477.md`, goal version 3.
- `docs/reports/boundary_conditions_and_springback_synthesis.md`, particularly lines 86–104 and 134–153.
- `docs/reports/never_clamped_preexecution_review.md`.

Reviewed scientific contrasts:

1. **Release constraints:** Putong A:xyz/B:yz/C:z versus fully free release from each identical final clamped state.
2. **Forming restraint:** never-clamped per-repeat forming versus clamped forming followed by fully free release.

The researcher-proposed `final_only` condition remains outside the authoritative goal and this approval. Historical single-pad claims and superseded handoffs were not treated as current conclusions or execution authority.

Implementation inspection covered both analyzers, all three relevant plotting scripts, analysis/setup/checkpoint tests, the control implementation delta, checkpoint and native acceptance/gauge code, and frozen scan-scoring helpers.

## 2. Identities and evidence linkage

All paths are relative to `/gpfs/projects/p33082/PSM2025-zigzag`.

The principal final identity record is:

`run/two_gripper_comparison/checks/final_identity_recheck.json`

SHA-256:
`2815dd680d2d24e63d0b2fd35270c511b48296ab761d24caac93e82bd021fdd8`

Its inspected script and log record **224 successful identity comparisons**, including all 50 substantive preexecution entries, production execution settings, analysis sources/inputs, plot provenance, preserved paired checkpoints, and PDF/render identities.

Selected identities:

| Artifact | SHA-256 |
|---|---|
| Control `bin/shell` | `d93b053a9bc6dbd5ecb3e964494bb1f0c33dc5d985e021fc1b25da81ef6edf39` |
| Preserved paired-production binary | `d8c2964734bc8afa9b84a8ab88f3172fc901521f267a20e53bb28f072eb45149` |
| `calibration/analysis/analyze_never_clamped.py` | `cc45d72dea09dcb7e3cca9e605c0a75e42510e1e697db3a86a5fe85e776d5c34` |
| `calibration/analysis/analyze_two_gripper_release.py` | `c24890c9e4e39cc25d4cc1a08a15812b98ceafadca31cd65a1b69e1d8129432b` |
| `calibration/analysis/plot_never_clamped.py` | `0d9277c72851787650e0383a4a65bf24e923e96d1b57830d067961156214eed6` |
| Expanded `forming_restraint_metrics.json` | `30f09eafb1990a8df79055bcf0b57b545df8e68d05d2c234cf4ef18320c4d3c5` |
| Expanded `run6_full_field.npz` | `34c96f79f553a2abf6af0bf50de8f8fcb2fdf928af0427da41566980b5247fde` |
| Expanded `run9_full_field.npz` | `cad3e249e6d420cd416bbf12d1a1c76c252361fcfbeb52375494fbcc23fc85e3` |

Expanded numeric outputs reside under `run/two_gripper_comparison/analysis_never_clamped/`.

Additional identity evidence:

- `run/two_gripper_comparison/checks/never_clamped_preexecution_identity.json` binds simulation, runner, batch, configs and connected numerical implementation.
- `run/two_gripper_comparison/checks/final_retained_recheck.json` confirms the original paired analysis/provenance and PDF identities. Both complete paired JSON case dictionaries and all nine original NPZ arrays equal their expanded counterparts exactly.
- The synthesis document’s current recorded hash is `93b6b4659184d96ae8d6f9c2bb33eab005ff5b13c41a1d112601e0f3d75b5c66`; its documented administrative change adds execution/analysis status, not a changed scientific recipe.

**Verification limitation:** Hash computation and NPZ arithmetic were parent-executed. I inspected the scripts, records and results; my tools did not permit independent command execution.

## 3. Scientific and numerical checks

### Matched execution and history — satisfactory

The production execution records identify fresh controls (`resume: null`) with the reviewed binary/runner, disabled physical BCs, and unchanged every-cycle cadence, continuation settings and acceptance tolerance.

`calibration/analysis/analyze_never_clamped.py:21–66` checks:

- Ordered accepted control repeats: 29 and 33.
- Unchanged checkpoint identity and empty vertex/edge masks.
- Final-checkpoint linkage and equality to the last accepted checkpoint.
- Exact matching reference DOFs, topology, edges, UV and reference curvature.
- Exact matching final top/bottom target metrics and accumulated hit counts.

Fresh initialization and zero initial history are supported by the inspected simulation path; controls are not initialized from clamped checkpoints. The reviewed implementation delta changes control/checkpoint plumbing, not the constitutive or native solver algorithms.

Preserved paired evidence in `checks/postexecution_audit.json` covers all 62 clamped checkpoints and final-parent equality. Separate release restoration, unchanged targets/history and no additional growth are enforced in `src/simulations/Sim_Bilayer_Growth.cpp:3019–3096`.

### Native acceptance and constraints — satisfactory

`src/libshell/NativeHybridShellSolver.hpp:339–358` requires genuine `gradient_target` termination, independent finite energy/gradient evaluation, gradient norm ≤ **5e−14**, and `positive_pivots` evidence. A failed/capped solve is not promoted merely because its retained candidate looks acceptable.

`src/libshell/ReferenceHybrid/EndpointCheck.hpp` checks the unshifted reduced Hessian, finite positive scaling, symmetry, pivot threshold and solve residual. This is **local numerical evidence**, not global stability or a shape-error certificate.

Final control gradient norms are:

- Run 6: **6.65944310893e−15**
- Run 9: **1.83471826331e−15**

Controls and free releases retain zero physical fixed coordinates and six rigid modes. Putong releases retain six fixed coordinates exactly, with zero remaining rigid modes. The reported rotational gauge conditioning ratios are approximately **0.666**, comfortably above the diagnostic rank threshold.

### Alignment and full-field statistics — satisfactory

`calibration/analysis/analyze_two_gripper_release.py:22–45` correctly implements reference-area-weighted, matched-vertex Kabsch alignment with determinant correction: proper rotation, translation, **no reflection or scaling**.

Weights distribute each reference triangle’s area equally to its vertices and normalize to unity. The RMS is
\[
\sqrt{\sum_i w_i\|x_i^{aligned}-y_i\|^2}.
\]
Quantiles are explicitly vertex quantiles; cumulative plots use reference-area weights. These are descriptive spatial statistics, not independent replicates.

All **15,251 vertices and 30,000 triangles** are retained. The final recheck confirms finite NPZ arrays, positive normalized weights, and exact reproduction of both contrasts’ saved RMS/maxima from the arrays.

| Contrast | Run 6 RMS / maximum, mm | Run 9 RMS / maximum, mm |
|---|---:|---:|
| Never-clamped versus clamped-then-free | 1.42913106e−7 / 4.48085700e−7 | 6.59321899e−7 / 1.98368157e−6 |
| Putong versus free release | 2.10318672e−5 / 5.43744706e−5 | 2.21528981e−5 / 7.39372845e−5 |

These agree with the current report.

### Energies and reaction-equivalent diagnostics — satisfactory

The implementation uses **E=1 normalization**. The report appropriately avoids interpreting normalized generalized gradients as measured forces.

`src/simulations/Sim_Bilayer_Growth.cpp:3061–3080` removes vertex masks before reaction-gradient evaluation, restores them safely, and checks energy/state invariance. The saved checkpoint test verifies unmasked components against finite differences.

Putong fixed-coordinate gradient norms are **1.21889e−16 / 3.21331e−16**. The normalized energy differences agree with the report:

- Putong minus free: **8.04966e−24 / 2.55917e−23**.
- Never-clamped minus free: **5.61417e−25 / −1.55904e−25**.

These provide compatibility diagnostics, not certified physical-effect magnitudes.

### Frozen scan contract — satisfactory within its stated scope

Inspected:

- `checks/frozen_scan_contract_vtp.json`
- Both analyzers’ scan branches
- `calibration/ewcal/material_frame.py`

The recorded baseline VTP rescoring reproduces the checked metrics within **2.29e−14 mm**, below the unchanged **1e−8 mm** requirement. Source/target hashes are verified before scoring.

Intrinsic comparisons use full-precision states; scan comparisons use the verified VTP serialization and material-reference registration contract. Reported symmetric scan RMS values match the saved output. The small score changes are not meaningful predictive improvements or independent physical identification. This is retrospective frozen-contract scoring, not a newly blinded evaluation.

## 4. Execution, tests and figures

`checks/final_job_accounting.txt` confirms:

- Job **4186540**: completed, exit 0:0, **1:56:59**, 8 CPUs.
- Job **4186689**: completed, exit 0:0, **2:00:35**, 8 CPUs.

Both are within the reviewed six-hour limits, without resumption.

Inspected saved test evidence:

- `checks/never_clamped_preparation/unit_tests_v2.log`: **33/33** passed.
- `checks/never_clamped_preparation/integration_tests_v2.log`: **4/4** passed.
- `checks/final_analysis_tests.log`: **4/4** passed, including invalid acceptance, corrupted history, incorrect rigid rank, resumed-chain rejection and layout guards.

Restart evidence is exercised coarse-fixture equality and terminal-export testing—not merely saved checkpoints. No production-resume claim is needed.

I visually inspected, one image at a time:

- All three PNGs in `calibration/analysis/plots/never_clamped_comparison/`.
- All three corresponding `pdf_check_*/page-0001.png` PDF renders.
- The retained paired-release PNG and `checks/release_pdf_review.png`.

No clipping, label collision, missing condition or unit inconsistency was found. Shared magnitude scales intentionally suppress visible forming-contrast variation; the numeric distributions remain authoritative. Stage figures correctly disclose original solver poses and ×5 height exaggeration. Their apparent pose differences are not intrinsic shape differences.

## 5. Findings and limitations

### Launch-/completion-blocking findings

**None unresolved for the reviewed completed slice.** No additional simulation is required to support its bounded conclusions.

### Nonblocking caveats

1. **Numerical interpretation — preserve the existing qualifications.**  
   `docs/reports/boundary_conditions_and_springback_synthesis.md:102,147` correctly avoids exact equality, numerical uncertainty bounds, global uniqueness and certified physical effects. Retain these qualifications in downstream summaries.

2. **Model-conditional scope.**  
   The comparisons support extremely close computed endpoints for these two prescribed-growth cases. They do not establish that real clamps leave plastic flow unchanged, explain Putong’s unavailable historical experiment, or establish cadence equivalence.

3. **Visual/provenance metadata.**  
   New producer manifests retain `pending`/`visually_verified: false`; manual inspection is recorded in this review and the synthesis. Main PNG hashes were recorded at final review, rather than in original output manifests. PDF source/render hashes have recorded counterparts. Preserve this distinction when citing provenance.

4. **Execution and reuse limits.**  
   Hash/NPZ computations and test runs were parent-executed, not independently rerun by this reviewer. Approval remains tied to the recorded identities. Resumed trajectories or consequential implementation/recipe changes require reassessment.

No simulations, installations, configuration changes, uploads, delegation or source/report mutations were performed by this reviewer. No model override was requested; runtime routing metadata was not independently inspectable.

## 6. Supported conclusion

For both tested cases, **neither comparison shows a macroscopically different computed final intrinsic shape**. The release results support the numerical-gauge interpretation of Putong’s constraints in these configurations. The forming results support endpoint insensitivity to the tested forming restraint within this prescribed-growth model.

The nonzero residuals remain unresolved between numerical error and comparably tiny model effects. The current report respects that boundary. Completion of this reviewed two-comparison scope is supported; the proposed final-only extension is not approved here.