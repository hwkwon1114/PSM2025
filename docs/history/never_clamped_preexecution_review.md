# Never-clamped controls: pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved launch blockers found for the reviewed implementation and proposed two-control protocol.** No additional simulation or routine rerun of the preserved paired releases is required by this review.

This is technical clearance of the reviewed slice—not execution authorization, a resource extension, sealed-data release, or approval of scientific conclusions.

## 1. Scope and authority

Reviewed:

- Authoritative objective: `.pi/goals/active_goal_2026101623411509_mvbwsnko-2nl477.md`.
- Current investigation, particularly the approved never-clamped extension: `docs/reports/boundary_conditions_and_springback_synthesis.md:112–130`.
- Control implementation, runner/batch changes, checkpoint implementation, connected native acceptance/gauge code, tests and saved v2 fixtures.
- Preserved baseline identities, implementation snapshots and Run 6/9 execution records.

Historical single-pad sections and obsolete missing-review-tool statements were **not** treated as current authority.

The reviewed production scope is exactly:

- Run 6: **29 accepted repeats**, never clamped.
- Run 9: **33 accepted repeats**, never clamped.
- Each: **8 CPUs, 32 GB, six-hour Slurm limit**, existing cooperative **600-second solve budget**, unchanged numerical acceptance.
- Fresh initial reference/history; same-control accepted-repeat resumption only.
- Preserve existing clamped forming and paired releases; no recalibration or changed physical recipe.

No routing override was requested and no quota block was observed. Runtime model/account routing was not independently inspectable through the available review tools.

## 2. Reviewed identities

Paths below are relative to `/gpfs/projects/p33082/PSM2025-zigzag`.

The complete supplied identity enumeration is bound by:

- `run/two_gripper_comparison/checks/never_clamped_preexecution_identity.json`  
  SHA-256: `81a46d4316b4412c6bf58f7ab6b447603e063230d464a48c2563c1fd2e735832`
- Current verification evidence:  
  `run/two_gripper_comparison/checks/never_clamped_preparation/review_identity_recheck.json`
- Preserved baseline record:  
  `run/two_gripper_comparison/checks/never_clamped_preparation/verified_baseline.json`  
  SHA-256: `54cad2203418b739044dd49ee3380e1863afb12505a5c9d2719054983be22972`

Principal current SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| `bin/shell` | `d93b053a9bc6dbd5ecb3e964494bb1f0c33dc5d985e021fc1b25da81ef6edf39` |
| `bin/native_hybrid_fixtures` | `b54f4425ba416142cb0c658b30c5420c588425538c2b608d21d3dab4e8b12504` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `e3a40e9c6f29c375fe7d4b024cba2fe6022cc66533453ceb30750acf69c2617c` |
| `scripts/run_two_gripper_comparison.py` | `9ff3f3c41621d27d3a35fab7827c6c6d299868eaa1aa384865a49c65e321939c` |
| `scripts/run_two_gripper_comparison.sbatch` | `99afe2de5947676543140f6b96b8954a6f1d353b8dcf77bece2158f8c1b57dfe` |
| `src/libshell/ZigZagSequenceCheckpoint.hpp` | `479e07608d379e1e136722c3a76f695dc233ff2759fbf6d6fc73c8b68e25febc` |
| `src/libshell/ZigZagSequenceBoundaryConditions.hpp` | `b8a46800bb25efeececc4b16932eff4f70673f0fc50da2682e23d21aff018f33` |
| `calibration/tests/test_two_gripper_setup.py` | `0b2066fef723412593693606e362578d940a47e3c200fed538dca0b4a9b34e1d` |
| `test/testshell/Test_ZigZagSequenceCheckpoint.cpp` | `0ae2bac8b4235db263dd447e228ced945e357c07a40fb2860c291cf5d5db6785` |
| `calibration/configs/boundary_conditions/run6_two_gripper_cycle.json` | `70a13f9ee685f8668b5980fa0e9d4fdffa4d43eebfa23f7381d1e85ea253c60a` |
| `calibration/configs/boundary_conditions/run9_two_gripper_cycle.json` | `a7eb26cb34f82e68cf8c617e0104b2fbe9e26f21b1941dcd410b13ff457f418c` |
| `CMakeLists.txt` | `476d035f63c002ed25859010e43b084f9d3d328716907f31f3b216c61aff1f58` |
| Current investigation document | `f30d573114f3989b0f9a8e3da865a401952ad7c14e1dd77264f4edfe4ec493e2` |

Connected numerical implementation:

| Artifact | SHA-256 |
|---|---|
| `src/libshell/NativeHybridShellSolver.hpp` | `6d59215f94fd89453a2638fecb248f139c28ff645cc02b3ec032494f3cd462fb` |
| `src/libshell/ShellEquilibriumSolver.hpp` | `ae023d092104e21d3524456a452428919608b4cc0e52fa4669d816485d5a2d8e` |
| `src/libshell/ReferenceHybrid/ShellProblem.hpp` | `d26babeaf2e12dc42d197cf0302221b96b8f3dcbb2ce125ff073183986cadff6` |
| `src/libshell/ReferenceHybrid/EndpointCheck.hpp` | `b8121115354e34eaa92375ad24636cbd87054850285555d953819d9d4c256c2f` |
| `src/libshell/ReferenceHybrid/Core.hpp` | `558a8e8810dd505acb23e742bc207d0ef5ce0fd70e0891fb21e91d10152d7b3d` |

The recheck also enumerates the remaining connected native-source hashes and reports all ten as tracked and unchanged.

**Administrative exception:** Of 51 packet entries, the only reported mismatch is the investigation document’s review-restoration update. Its actual reviewed hash is listed above, replacing packet value `20e92b7d1a54f8898161683447e52169fbff2254cb60af0a4b9976e2b97d4843`. This does not change the scientific protocol or implementation.

**Verification limitation:** Hash computation was performed and attested by the parent. I inspected the resulting records and fixture-recorded identities; my tools did not permit independent hash execution.

## 3. Checks and findings

### Scientific comparison and baseline preservation — satisfactory

The design separates:

1. **Release-constraint effect:** preserved Putong-versus-free endpoints from the same clamped parent.
2. **Forming-restraint effect:** new never-clamped endpoint versus preserved clamped-then-free endpoint.

The second contrast is appropriately restricted to the prescribed-growth model. Neither improved scan fit nor agreement between these endpoints identifies physical plastic-flow behavior.

I inspected both preserved production execution records and the control delta:

`run/two_gripper_comparison/checks/never_clamped_preparation/review_control_delta.diff`

The delta adds the opt-in control/checkpoint path and final export; it does not change the growth law, mesh, material constants or native solver algorithm. The runner retains the production settings except the intended BC/release switches.

Preserved original production binary:

`d8c2964734bc8afa9b84a8ab88f3172fc901521f267a20e53bb28f072eb45149`

Preserved execution-record hashes:

- Run 6: `11ff2e158fc2723d474a9514647fad5fdabb49b0e8a313784c6737e962fc1175`
- Run 9: `0d8656303d00c769de14577b62ae665503d1ad3e853b5786e558c91bd9caf0a9`

Frozen Attempt 209 sequence hashes:

- Run 6: `ae885273b4da727a1815c4104d41328a09f96749e478cbf89dbe2ff58df92bb8`
- Run 9: `e3fe1bb10eaf4b45858b5902567dd7c6428da814642b1ea1951ed1d1ed4e86b5`

Job4181609’s saved verification covers all **62 accepted checkpoints**, final parents and paired endpoints. The current parent recheck reports all four preserved implementation snapshots and twelve baseline input/endpoint records unchanged. Archived `final_only` runs are correctly excluded as matched controls.

### Constraints and numerical rigid modes — satisfactory

`src/simulations/Sim_Bilayer_Growth.cpp:1047–1062,2333–2356` rejects incompatible BC, paired-release and cadence combinations. Never-clamped initialization checks zero fixed vertex coordinates; checkpoint machinery rejects edge/normal clamps.

Every checkpoint capture checks the current vertex mask against the immutable identity. The final control path, `src/simulations/Sim_Bilayer_Growth.cpp:2986–3015`, requires zero fixed vertex coordinates and **six remaining rigid modes**.

The native Newton gauge is numerical, not a persistent physical mask:

- `src/libshell/ShellEquilibriumSolver.hpp:139–298` constructs rigid translations/rotations from current vertices, with zero director components.
- `src/libshell/ReferenceHybrid/ShellProblem.hpp` selects exactly the surviving rigid-mode rank through pivoted QR when constructing reduced Newton coordinates.
- No Putong mask is installed in the never-clamped branch.

### Targets, history and acceptance — satisfactory

Initialization retains the same reference sheet/directors and zero initial hit history. With slope feedback disabled, material-coordinate hit ordering and hardening determine the target updates independently of the current formed geometry.

The inspected growth update uses the congruence
\[
A_{\mathrm{new}}=T^\mathsf{T}A_{\mathrm{old}}T,
\]
with positive stretch and positive-definite metric checks. Hardening remains the frozen hit-count exponential law. Intermediate continuation interpolates targets; acceptance at full load uses the **exact final target arrays**, not a rounded interpolation.

`src/simulations/Sim_Bilayer_Growth.cpp:2720–2822` publishes an accepted-repeat checkpoint only after accepted full-load equilibrium and committed target/history state.

The unchanged native acceptance requires:

- genuine `gradient_target` termination;
- independently evaluated finite energy and gradient;
- gradient norm at most **5e-14**;
- unshifted reduced-Hessian `positive_pivots` evidence.

A capped or failed solve is not promoted to acceptance because its retained candidate looks satisfactory. Positive pivots remain local numerical evidence, not proof of global optimality.

### Accepted-repeat resumption and final export — satisfactory

`src/libshell/ZigZagSequenceCheckpoint.hpp` preserves full current DOFs, including directors, both target metrics, hit counts, accepted cursor/proof and immutable reference/topology/UV/curvature/masks. Restore validates identity, cursor, acceptance, dimensions, finite values, metric positivity and integer history before applying state.

The runner’s digest includes binary, runner, sequence and options. Its resume identity check precedes output-directory creation. C++ restore independently rejects a clamped identity.

`src/simulations/Sim_Bilayer_Growth.cpp:2447` skips completed repeats. Terminal-only resume reaches the final diagnostic/export path without replaying growth or invoking the optimizer.

Checkpoint publication is no-overwrite and uses checksum plus file/directory synchronization. This is accepted-repeat recovery—not restoration of an interrupted optimizer’s internal state.

### Saved v2 evidence — satisfactory within fixture scope

Inspected evidence under:

- `run/two_gripper_comparison/checks/never_clamped_preparation/`
- `run/two_gripper_comparison/fixtures/never_clamped_v2/`

Job4182922 evidence shows:

- `unit_tests_v2.log`: **33/33** tests passed—27 native-production and six checkpoint tests.
- `integration_tests_v2.log`: **4/4** Python tests passed.
- All five fixture execution records identify the reviewed binary/runner and job4182922.
- Uninterrupted/resumed controls have exactly equal DOFs, targets and history under the saved assertions.
- Control and paired-regression fixtures have equal reference geometry, topology, UV, curvature, final targets and history.
- Terminal resume has no accepted-cycle replay or native optimizer log entry; the saved test verifies identical final checkpoint and byte-identical final VTP.
- Clamped resume produces `Resume build/recipe identity mismatch`, without creating its output directory.
- Inspected control result reports zero fixed DOFs, six rigid modes and final gradient norm `6.0836733970531435e-19`.

These are coarse correctness fixtures, not production convergence or scientific-effect evidence.

## 4. Launch-blocking findings

**None unresolved.**

Approval is limited to the identities and protocol above. Consequential implementation, dependency, recipe or acceptance changes require reassessment.

## 5. Nonblocking caveats and required follow-through

1. **Timers are cooperative.**  
   `src/libshell/NativeHybridShellSolver.hpp` permits verification to start before 630 seconds and rejects acceptance at/after 650 seconds for the 600-second setting. Individual evaluations are not preempted. The six-hour Slurm allocation is the outer limit; completion within it is not demonstrated.

2. **Resume permission remains conditional.**  
   `scripts/run_two_gripper_comparison.py:77–86` checks identity, not predecessor resource accounting or review authorization. Retain predecessor artifacts and record resource use before same-control resumption. No restart from flat or tolerance relaxation is cleared.

3. **Production equality still requires measurement.**  
   Before interpreting either forming-restraint contrast, explicitly compare production reference/director state, topology, UV, curvature, final targets and history against the preserved clamped trajectory. Fixture equality is not a substitute.

4. **Use the checkpoint as the complete endpoint record.**  
   `src/simulations/Sim_Bilayer_Growth.cpp:3004–3005` exports a terminal snapshot with zero per-cycle increment/hit arguments. Its accumulated targets/history are valid, but zero incremental diagnostic fields do not describe the last forming repeat. Terminal-only resume also does not regenerate the optimizer’s diagnostic state export.

5. **Do not overstate numerical or physical identification.**  
   Exact resumed equality was exercised only on the coarse fixture. Positive pivots do not establish uniqueness or a shape-error bound. Energies/gradients use normalized **E=1**. Two deterministic cases and their mesh vertices do not provide population replication.

6. **Postexecution analysis remains separately reviewable.**  
   Require proper material-correspondence rigid alignment, full-field distributions, convergence/local-stability evidence and inspected figures. Scan comparisons must retain the verified registration/scoring contract; no sealed-data release is implied.

## 6. Unperformed checks and approval limits

I performed read-only source, diff, record and log inspection. I did **not** execute tests, rebuild, launch simulations, independently recompute hashes, inspect new production results, or validate final plots/scan scores.

No code or existing records were modified by this reviewer. The parent’s identity-only recheck is explicitly distinguished from independent reviewer execution.

Agreement would support insensitivity to forming restraint only for these cases and this model. Disagreement with matched targets and accepted equilibria may support path-selected branches; unequal targets, incomplete convergence or unresolved numerical sensitivity require an inconclusive interpretation.