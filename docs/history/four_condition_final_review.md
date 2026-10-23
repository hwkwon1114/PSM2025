# Independent final review: four-condition investigation

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific, operational, or completion-blocking findings.** The saved evidence supports completion of the four-condition investigation and its bounded conclusions for Run 6 and Run 9.

This review does not authorize further execution, recipe changes, new scan scoring, budget extensions, or sealed-test release.

## 1. Scope and authority

Root: `/gpfs/projects/p33082/PSM2025-zigzag`. Paths below are relative to this root.

Authority inspected:

- `.pi/goals/active_goal_2026101623411509_mvbwsnko-2nl477.md`
- `docs/reports/boundary_conditions_and_springback_synthesis.md:155–188`: approved expansion, execution contract, and four-condition results.

Earlier handoffs and single-pad conclusions were treated as historical—not current authority.

The three contrasts correctly answer separate, model-conditional questions:

1. **Cadence:** never-clamped final-only versus never-clamped per-cycle equilibrium.
2. **Forming restraint:** never-clamped per-cycle versus clamped-then-free.
3. **Release constraints:** Putong versus free release from the same clamped parent.

Exact target/history matching and accepted endpoints address unequal loading and incomplete convergence as rival explanations. The remaining small differences cannot be separated into numerical error versus comparably tiny model effects. This is not physical identification of clamp-dependent plastic flow.

Prior reviews were reused within their limits:

- `docs/reports/two_gripper_release_final_review.md`: retained forming-restraint and release comparisons.
- `docs/reports/final_only_preexecution_review.md`: final-only implementation, acceptance, and exercised terminal-export recovery—not final scientific conclusions.

## 2. Reviewed identities

The complete identity inventories are:

- `run/two_gripper_comparison/checks/final_only_preparation/preexecution_identity.json`
- `run/two_gripper_comparison/checks/final_only_preparation/final_identity_audit.json`

The preexecution packet SHA-256 is  
`79f3b601c5894549bbf61d7c23b45f774c842317f785fb0745afb9593b16ba6a`.

Selected identities recorded in the inspected evidence:

| Artifact | SHA-256 |
|---|---|
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `e86851a42805e1b8de08ae7d5faa2ed7bf42665a6006fc8123c9890cb10f5693` |
| `scripts/run_two_gripper_comparison.py` | `f9b948c12b3e7b529b6277389d0a2efb5bb81ae378fa41b063e4006fa291aeee` |
| `scripts/run_two_gripper_comparison.sbatch` | `93b76db6ae2bb6b732329d07f7e192425f32a5203db9e31ecd9f9b10abe84d07` |
| `calibration/analysis/analyze_four_conditions.py` | `3e42c15a5123a0b3c6b618693d5edc4f8ed76b4ee5707bb5d1535ad06b7beddd` |
| `calibration/analysis/plot_four_conditions.py` | `759fb6f5b8dcdc7a1eae2d2c77e6b9f9ed0e1afe19f641505ecf87c28b824cfb` |
| `calibration/tests/test_four_condition_analysis.py` | `54fd9916b6143bfe0f3f6eae21e2fc0287d6a54e06f49feb4b23138a74add9ab` |
| `calibration/configs/boundary_conditions/run6_two_gripper_cycle.json` | `70a13f9ee685f8668b5980fa0e9d4fdffa4d43eebfa23f7381d1e85ea253c60a` |
| `calibration/configs/boundary_conditions/run9_two_gripper_cycle.json` | `a7eb26cb34f82e68cf8c617e0104b2fbe9e26f21b1941dcd410b13ff457f418c` |
| `run/two_gripper_comparison/analysis_four_conditions/four_condition_metrics.json` | `6758ed7e60b499f5e97dc634ace38c0188e49974492c003524fcf7dac12bb260` |
| `run/two_gripper_comparison/analysis_four_conditions/run6_full_field.npz` | `23f40de05b6779d1ac113a64d230a4269c73134f48696cb501aea9fff5357a72` |
| `run/two_gripper_comparison/analysis_four_conditions/run9_full_field.npz` | `88d5d5ff7cecd2eac245d07fe5c027c7d7b4bc8ec8495bb604a7f9fee5db9330` |
| `docs/reports/boundary_conditions_and_springback_synthesis.md` | `bc72584ecda6071cff8bceb6d391892f32cbf4f6f110b990cc9285453ec96fcf` |

The final-only checkpoint identities are:

- `run/two_gripper_comparison/run6_final_only/never_clamped_final_checkpoint.json`: `c42a9becf57f3ae90ab9a65a4a5e8c6e8ed424b9ea205a36c3e325481fed8beb`
- `run/two_gripper_comparison/run9_final_only/never_clamped_final_checkpoint.json`: `0647743319fd1a9498b32cb81bcbd26596ea9fd503afef10e7e0fed2006cb960`

Connected analyzers retain the identities from the prior final review:

- `analyze_never_clamped.py`: `cc45d72dea09dcb7e3cca9e605c0a75e42510e1e697db3a86a5fe85e776d5c34`
- `analyze_two_gripper_release.py`: `c24890c9e4e39cc25d4cc1a08a15812b98ceafadca31cd65a1b69e1d8129432b`

**Evidence limitation:** These hashes were computed by the parent, not independently by this reviewer. I inspected the records and audit implementation. `verify_final.py` and `final_identity_audit.log` report **174 passing checks and 196 bound identities**, including unchanged retained arrays and recomputed cadence metrics. The synthesis was separately bound after its status/results additions rather than required to match its preexecution text.

## 3. Scientific and implementation checks

### Comparability and execution — satisfactory

The archive exclusion is justified by the contract: archived final-only commands used **3600 seconds**, versus the matched **600-second** native setting. Their saved convergence records show accepted solves; exclusion is not evidence of historical failure. The Run 9 path alias remains disclosed.

Inspection of the reviewed delta, current cadence gates, production execution records, history/convergence CSVs, and analyzer establishes:

- Fresh initialization, with `resume: null`; no initialization from another condition’s deformed checkpoint.
- All **29/33 ordered updates** before optimization, rather than multiplied strain.
- History marks only the last repeat solved; each convergence CSV contains exactly one accepted full-load row, with `lambda_trial=1` and no retry.
- No intermediate accepted checkpoints.
- Exact reference DOFs—including reference directors—topology, edges, UV, reference curvature, final target metrics, and hit-history matching.
- Byte-identical initial VTPs supplement, rather than replace, full-state/source evidence.
- Zero physical masks and six numerical rigid modes.
- Identical sequence and solver options except the declared cadence and `sequence_adaptive=false` switches.

`calibration/analysis/analyze_four_conditions.py` fails closed on mismatched completion, proof, constraints, targets/history, options, checkpoint linkage, or cadence. The unchanged connected analyzers preserve the two retained contrasts.

The preexecution review establishes that retained per-cycle comparators did not exercise adaptive subdivisions. Disabling adaptive continuation therefore does not conceal an additional exercised load-step difference.

### Acceptance and resources — satisfactory

The production proofs retain the project’s CPU double-precision contract, **5e−14** gradient threshold, genuine `gradient_target` termination, independent endpoint verification, and `positive_pivots` local-curvature evidence.

| Final-only diagnostic | Run 6 | Run 9 |
|---|---:|---:|
| Gradient norm | 2.70515645e−15 | 4.06428064e−16 |
| Normalized energy, E=1 | 5.396721183487773e−12 | 3.555730154312341e−12 |
| Energy minus per-cycle | +3.43910033e−23 | −2.26182199e−26 |

`production_accounting.txt` records jobs **4214036/4214037**, both completed with exit **0:0**, in **4:45/3:27**, using eight CPUs. These are within the reviewed 30-minute limits. Neither production trajectory resumed. The prior exercised terminal-export fixture is not being represented as optimizer-internal restart capability.

Positive pivots establish local numerical evidence—not global uniqueness or a shape-error bound.

### Alignment, statistics, and retained evidence — satisfactory

The shared alignment correctly uses material correspondence, reference-area weights, and determinant-corrected Kabsch rotation: no reflection or rescaling.

For aligned vertex distances \(d_i\), RMS is
\[
\sqrt{\sum_i w_i d_i^2}.
\]

All **15,251 vertices and 30,000 triangles** are retained. Quantiles are vertex quantiles; CDFs accumulate reference-area weights. Neither constitutes independent spatial replication.

| Contrast: RMS / maximum, mm | Run 6 | Run 9 |
|---|---:|---:|
| Cadence | 4.34584695e−5 / 1.12201205e−4 | 5.28371274e−7 / 1.62967599e−6 |
| Forming restraint | 1.42913106e−7 / 4.48085700e−7 | 6.59321899e−7 / 1.98368157e−6 |
| Release constraints | 2.10318672e−5 / 5.43744706e−5 | 2.21528981e−5 / 7.39372845e−5 |

These agree with the synthesis. The parent audit checks every retained NPZ array for exact equality and recomputes cadence vectors, magnitudes, RMS, and maximum.

Previously reviewed release-parent equality, unchanged targets during release, fixed-coordinate satisfaction, gauge conditioning, and unmasked reaction-equivalent diagnostics remain applicable under the retained identities. Energies and generalized gradients remain **E=1 normalized**, not measured forces.

No new final-only scan score is produced. Earlier retrospective scan scores retain their frozen registration/VTP-scoring contract and prior limitations; they do not establish predictive improvement or physical mechanism.

## 4. Tests and figures

I inspected the new test source and saved `analysis_tests.log`: **2/2 tests passed**, covering the saved-fixture contrast and rejection of corrupted history/cadence.

Native, integration, retained-regression, and terminal-recovery checks were reused through the preexecution review and matching identity evidence; they were not rerun.

I independently inspected, one image per tool call:

- All three PNG figures in `calibration/analysis/plots/four_condition_comparison/`.
- All three corresponding `pdf_check_*/page-0001.png` renders.
- Plotting source, connected stage/save helpers, provenance, and PDF-render manifests.

No clipping, label collision, missing condition, or unit inconsistency was observed. The figures correctly disclose original solver poses and **×5 vertical exaggeration**. Their apparent pose differences are not intrinsic differences.

Shared map scales intentionally suppress the smallest contrasts. CDF horizontal ranges differ between cases; cross-case visual comparison must consult the axes.

## 5. Findings and approval limits

### Launch-/completion-blocking findings

**None unresolved for the reviewed completed slice.** No additional simulation or redundant rerun is required to support these conclusions.

### Nonblocking caveats

1. **Numerical resolution and model scope.**  
   Preserve the qualifications in `docs/reports/boundary_conditions_and_springback_synthesis.md:186`: nonzero residuals do not establish exact equality, global uniqueness, a certified error bound, or resolved physical effects. The results do not demonstrate general interchangeability of final-only and per-cycle optimization.

2. **Figure interpretation and metadata.**  
   Preserve the scale/pose warnings in `docs/reports/boundary_conditions_and_springback_synthesis.md:188` when reusing figures. Producer metadata still says `pending`/`visually_verified: false`; the manual inspection recorded here is separate evidence, not an automated flag update. No color-vision-deficiency simulation was performed.

3. **Verification provenance.**  
   Hashing, NPZ arithmetic, tests, production, and PDF rendering were parent-executed. This reviewer performed independent static and visual inspection, not independent numerical execution.

4. **Identity-limited approval.**  
   Consequential source, recipe, input, or trajectory changes require reassessment. This review does not certify runtime enforcement or tracker/auditor completion.

## Supported conclusion

For these two frozen Attempt 209 prescribed-growth cases, **all three contrasts produce extremely close computed final intrinsic shapes**. The largest discrepancy is Run 6 cadence: approximately **0.04346 µm RMS and 0.11220 µm maximum**.

The evidence supports case-specific endpoint insensitivity to the tested cadence and restraints—not universal equivalence, physical plastic-flow invariance, or identification of Putong’s unavailable historical discrepancy.

No files were modified, commands executed, experiments launched, packages installed, configurations changed, or children delegated by this reviewer. No account/model override was requested; runtime routing metadata was not independently inspectable.