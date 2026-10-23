# Final-only controls: independent pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers found for the reviewed final-only slice.**

This is technical clearance, not execution authorization, additional resource permission, approval of final conclusions, or sealed-test release. It does not authorize rerunning the three completed conditions.

## 1. Scope and authority

Root: `/gpfs/projects/p33082/PSM2025-zigzag`. Paths below are relative to this root.

Authority inspected:

- `.pi/goals/active_goal_2026101623411509_mvbwsnko-2nl477.md`: the current four-condition revision.
- `docs/reports/boundary_conditions_and_springback_synthesis.md:155–165`: **Approved four-condition expansion (m00667)**.

Prior evidence was reused within its original limits:

- `docs/reports/never_clamped_preexecution_review.md`
- `docs/reports/two_gripper_release_final_review.md`

Those reviews did not approve this cadence extension. Historical single-pad claims and superseded handoffs were not treated as authority.

Reviewed implementation: the final-only opt-in changes in the simulation, Python/batch runners and new integration tests, plus connected growth, checkpoint, native acceptance and rigid-mode handling.

The reviewed production contract is **one fresh final-only trajectory each for Run 6 and Run 9**, accumulating respectively **29 and 33 ordered repeats**. Each uses **8 CPUs, 32 GB, 30-minute Slurm wall time**, unchanged **2000 warmup attempts, 600-second native setting and 5e−14 acceptance tolerance**. Adaptive load continuation is deliberately disabled.

No routing override was requested. Runtime model/account routing was not independently inspectable through the available tools.

## 2. Reviewed identities

The complete 67-entry enumeration is:

`run/two_gripper_comparison/checks/final_only_preparation/preexecution_identity.json`

SHA-256: `79f3b601c5894549bbf61d7c23b45f774c842317f785fb0745afb9593b16ba6a`

The reviewed delta is:

`run/two_gripper_comparison/checks/final_only_preparation/control_delta.diff`

SHA-256: `a6f550edb8cf8c9503f78117f1b20a776f304ce8e26fa9d9aae8af07335049a3`

Principal identities:

| Artifact | SHA-256 |
|---|---|
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `e86851a42805e1b8de08ae7d5faa2ed7bf42665a6006fc8123c9890cb10f5693` |
| `scripts/run_two_gripper_comparison.py` | `f9b948c12b3e7b529b6277389d0a2efb5bb81ae378fa41b063e4006fa291aeee` |
| `scripts/run_two_gripper_comparison.sbatch` | `93b76db6ae2bb6b732329d07f7e192425f32a5203db9e31ecd9f9b10abe84d07` |
| `calibration/tests/test_final_only_controls.py` | `af09475425117e5c1937ecbefd1ee1cee26908d3b016529c95298af5cc59a669` |
| `calibration/configs/boundary_conditions/run6_two_gripper_cycle.json` | `70a13f9ee685f8668b5980fa0e9d4fdffa4d43eebfa23f7381d1e85ea253c60a` |
| `calibration/configs/boundary_conditions/run9_two_gripper_cycle.json` | `a7eb26cb34f82e68cf8c617e0104b2fbe9e26f21b1941dcd410b13ff457f418c` |
| `src/libshell/ZigZagSequenceCheckpoint.hpp` | `479e07608d379e1e136722c3a76f695dc233ff2759fbf6d6fc73c8b68e25febc` |
| `src/libshell/GrowthHelper.hpp` | `323dd80a7fd9cdb1bf9f6723209c1fcc552a4e7b08b8bc999b729ba53166dd04` |
| `src/libshell/ZigZagSequenceGrowth.hpp` | `75887bd1885b8d41f843451da778df414c93bbcade75f3afea715ec6093bb6ec` |

The packet also binds the connected native implementation, fixtures, logs, investigation document and preserved implementation snapshots.

Parent-executed identity evidence, independently read here:

- `run/two_gripper_comparison/checks/final_only_preparation/review_identity_attestation.json`: all **67 entries match**; recomputed three-file delta is byte-identical; both fresh production directories are absent.
- The same attestation binds and confirms equality of the fixture initial-VTP pair and final/terminal endpoint-VTP pair.
- `run/two_gripper_comparison/checks/final_only_preparation/growth_identity_attestation.json`: both growth helpers are tracked and unchanged from HEAD; no staged files reported.

**Limitation:** Hash computation was performed by the parent, not independently executed by this reviewer.

## 3. Scientific and implementation checks

### Comparison design and retained evidence

The design correctly separates:

1. Per-cycle versus final-only: **equilibrium cadence**.
2. Per-cycle never-clamped versus clamped-then-free: **forming restraint**.
3. Putong-pinned versus free release from the same clamped parent: **release constraints**.

The latter two comparisons remain preserved.

`run/two_gripper_comparison/checks/final_only_preparation/archive_audit.json` records the 225 retained-identity checks and preserved snapshots. Its archived final-only convergence records show accepted full-load solves, not failures. Nevertheless, their **3600-second setting differs from the matched 600-second recipe**, so excluding them from exact reuse is justified. Run 9’s archived command-path alias is explicitly acknowledged rather than inferred from directory naming.

### Ordered accumulation and initialization

`src/simulations/Sim_Bilayer_Growth.cpp:1192–1205,2333–2363,2452–2668` preserves fresh reference initialization and zero initial hit history, then applies every original material hit in order.

The unchanged hardening law is
\[
q(n)=f+(1-f)e^{-\beta n}.
\]

The multiplicative target update is
\[
A_{\mathrm{new}}=T^\mathsf{T}A_{\mathrm{old}}T,\qquad
T=D_m^{-1}GD_m.
\]

`src/libshell/GrowthHelper.hpp:740–826` checks positive incremental stretches, material-triangle conditioning and positive-definite updated metrics. No multiplied-strain replacement is introduced.

With slope feedback disabled, hit selection and target/history updates do not depend on the current formed shape. `warm_start=true` does **not** import another condition’s endpoint: on a fresh final-only run there is no earlier equilibrium to warm-start from.

### No intermediate equilibrium or persistent pins

`src/simulations/Sim_Bilayer_Growth.cpp:2340–2348,2489–2491,2663–2668` rejects adaptive continuation and stop-after for this mode and skips optimization before the final repeat.

At the final repeat, the continuation step is exactly one; the exact accumulated target arrays are installed before calling the native solver. Failed acceptance throws rather than taking a smaller load step. Successful capture follows restoration of the complete accumulated history.

The endpoint and runner additionally require:

- One equilibrium on a fresh run; zero on terminal export recovery.
- Only the final accepted-cycle checkpoint.
- One accepted convergence row at the final cycle with `lambda_trial=1`.
- Zero physical vertex/edge masks and six remaining rigid modes.

The inspected numerical gauge removes surviving rigid freedoms without installing persistent physical pins.

### Numerical fairness and fail-closed acceptance

The runner preserves mesh, material, growth recipe and native settings apart from the explicitly approved cadence/continuation switches.

The retained per-cycle convergence CSVs contain 29/33 accepted full-step records without subdivision or retries. Thus disabling adaptive continuation does not remove an actually exercised substep path from those retained comparators.

The unchanged native implementation requires genuine `gradient_target` termination, independent finite energy/gradient evaluation, gradient norm ≤ **5e−14**, and unshifted reduced-Hessian `positive_pivots` evidence. Failed or capped optimization is not promoted to acceptance merely because its retained candidate looks satisfactory.

This is the project’s CPU double-precision contract (`Real` is `double`), not a newly imposed precision requirement. Positive pivots are local numerical evidence, not global uniqueness or a shape-error certificate.

### Terminal recovery and diagnostics

`src/simulations/Sim_Bilayer_Growth.cpp:2352–2360` accepts only a terminal checkpoint in final-only mode. The runner rejects cross-cadence identity mismatch before output creation.

`src/libshell/ZigZagSequenceCheckpoint.hpp` validates checksum, immutable identity, acceptance, dimensions, finite state, positive metrics and integer history before restoration. It retains full DOFs, including directors, both target fields and hit counts.

Terminal recovery skips growth and optimization, then exports the restored endpoint. This is exercised export recovery—not optimizer-internal resumability.

## 4. Evidence inspected

Under `run/two_gripper_comparison/checks/final_only_preparation/`:

- `build.log`: simulation rebuilt and linked successfully.
- `unit_tests.log`: **33/33 passed**, comprising native-production and checkpoint tests.
- `integration_tests_corrected.log`: **2/2 final-only tests passed**.
- `retained_tests.log`: **4/4 retained setup tests passed**.
- `integration_tests.log`: preserved initial assertion failure.

The initial assertion expected a pending row in convergence CSV. Source inspection confirms skipped repeats are written to **history CSV**, and bypass convergence output. The correction is appropriate and does not conceal a solver failure.

Under `run/two_gripper_comparison/fixtures/final_only_v1/`, inspected execution records, solver logs and cadence outputs support:

- Reviewed binary/runner identities and job4211885 provenance.
- Exact reference/target/history equality under the saved tests.
- Fresh history `solved=[0,1]` and one full-load accepted solve.
- Final gradient norm **2.4533837872933932e−17**, zero fixed coordinates and six rigid modes.
- Terminal recovery with no native optimizer invocation, identical checkpoint and byte-identical endpoint VTP.
- Rejection of cross-cadence recovery without creating its output directory.

Job4212696’s Python syntax, batch syntax and whitespace checks are **parent-attested**; no separate saved shell-check log was available. Test logs were independently read. These coarse fixtures establish correctness, not production convergence or runtime.

## 5. Findings and conditions

### Launch-blocking findings

**None unresolved for the reviewed identities and contract.**

### Nonblocking caveats and required follow-through

1. **Explicit wall-time override remains necessary.**  
   `scripts/run_two_gripper_comparison.sbatch:9,15–19` retains a six-hour default for older modes. The reviewed final-only invocation includes `sbatch --time=00:30:00`. The Python process ceiling is 1800 seconds; the runner is not an authorization gate.

2. **Native timing is cooperative.**  
   `src/libshell/NativeHybridShellSolver.hpp:339–358` allows verification to begin before 630 seconds and rejects acceptance at/after 650 seconds for the 600-second setting. Individual evaluations are not preempted. Production feasibility is plausible but not demonstrated by the coarse fixture. Timeout or nonconvergence must remain a retained outcome, not trigger a fresh restart or changed recipe.

3. **Recovery accounting remains external.**  
   `scripts/run_two_gripper_comparison.py:89–94` checks identity, not predecessor resource accounting. Recovery is limited to the same trajectory’s accepted terminal checkpoint, with predecessor evidence retained. No incomplete-optimizer recovery is cleared.

4. **Use acceptance evidence, not history labels alone.**  
   `src/simulations/Sim_Bilayer_Growth.cpp:2663–2668` writes `solved=1` before attempting optimization. On failure, that field does not establish acceptance. Require native proof, accepted checkpoint and completed endpoint checks. Terminal export’s zero incremental fields likewise do not describe the final forming repeat.

5. **Production comparability remains to be measured.**  
   Before interpreting cadence differences, verify full reference/director state, topology, UV, curvature, final target metrics and hardening history against retained controls. Use proper material-correspondence rigid alignment and full-precision checkpoint geometry. Scan scoring must retain the frozen registration/VTP contract. The developing four-condition analyzer and final figures were outside this launch review.

Consequential changes to reviewed implementation, inputs, dependencies or acceptance require reassessment. No additional fixture campaign or rerun of completed conditions is requested.

## 6. Claim boundaries and review limitations

With matched targets and accepted endpoints, agreement supports cadence insensitivity **only for these two prescribed-growth cases**. Disagreement may reflect numerical path selection or distinct local branches; it does not by itself establish physical plastic-flow effects. Unequal targets, failed acceptance or insufficient numerical resolution leave the cadence contrast inconclusive.

Normalized **E=1** energies/gradients are not measured forces. Tiny shape or scan-score differences must not be called exact equality, meaningful predictive improvement, or global uniqueness. Mesh vertices are not independent experimental replicates.

I performed read-only source, delta, authority, record and log inspection. I did not execute shell commands, tests, builds, simulations, hash calculations, scan scoring or plotting; make changes; install software; or delegate. Parent-executed evidence is distinguished above. Approval covers preparation of this slice, not completion of the four-condition investigation.