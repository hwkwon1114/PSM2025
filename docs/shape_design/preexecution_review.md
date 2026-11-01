# Independent pre-execution review

**BLOCKED**

The frozen parameters, generated sequences and solver arguments match the recorded plan. However, the curl claim, analysis input precision and 1 mm execution envelope have blocking issues.

## BLOCKING ISSUES

1. **The 7% curl claim is not established for the generated toolpaths.**
   - Card: `docs/shape_design/attempt209_double_curvature.md:39,78` claims a worst-case 7% residual-curl limit.
   - Evidence: `shape_design/tests/test_family.py:133-148` assumes one hit per active F/C pass (`n += 1`), without using actual strip intersections, overlaps or segment directions.
   - Solver-free counterexample using production geometry: at `(u,v)=(-98.666667,-49.333333)` mm, `flip,k=0` receives eight hits, **all back-face C hits**, from strips 45 and 46. Its linearized layer-difference proxy is therefore 100% one-sided, not ≤7%.
   - `shape_design/family.py:128-130` makes the footprint depend on the physical face. Changing round order cannot cancel growth where only one face’s footprint intersects.
   - Consequence: the test verifies an idealized ordering surrogate, not a family-wide curl bound or equilibrium-shape bound. The card also leaves “curl” aggregation undefined; the stated 12–27% alternatives are not reproduced by this test.

2. **Float32 mesh ingestion changes the prescribed quadratic-fit ROI and violates the local-curvature helper’s precision contract.**
   - Card: `docs/shape_design/attempt209_double_curvature.md:67` specifies inclusive `|u|≤100, |v|≤52` mm.
   - Evidence: `src/libshell/WriteVTK.hpp:51-56` creates default Float32 `vtkPoints`; `shape_design/run_case.py:60-74` saves these coordinates for analysis.
   - `shape_design/analysis.py:27` uses only `1e-9` mm boundary tolerance. Exported nominal `u=100` becomes `100.00000149011612` mm.
   - Reproduced ROI counts: **5049 instead of 5353 vertices at 2 mm**, and **20497 instead of 21105 at 1 mm**. Both nominal boundary rows/columns are excluded.
   - Consequence: nonquadratic shapes are fitted over different domains than specified, with different domain shrinkage between meshes; `K_fit`, classifications and mesh comparisons can change.
   - Additionally, `dome_bo/dome_curvature.py:1` requires “native full-precision vertices,” but `shape_design/analysis.py:57` receives the quantized VTP-derived coordinates. Casting them to double does not restore precision.

3. **The 1 mm scheduler limit can preempt the promised controller timeout and failure recording.**
   - Card: `docs/shape_design/attempt209_double_curvature.md:61` gives the 1 mm check a 7200 s limit.
   - Evidence: `shape_design/dry_run.py:23` and `shape_design/family.py:185-186` assign 7200 s to controller/native limits, while `shape_design/run_family.sbatch:9` also sets total job wall time to exactly `02:00:00`.
   - Python startup, hashing and preparation occur before `calibration/ewcal/runner.py:109` starts the controller timer; validation and output recording occur afterward.
   - Consequence: near-limit mesh checks can be scheduler-killed before the controller finishes its acceptance/failure path. `shape_design/run_case.py:84-86` writes `result.json` only in `finally`, which ordinary scheduler termination need not execute.

## Verified contract-to-code trace

4. **Frozen model and geometry — verified.**
   - Required: fixed engineering-strain parameters, layer orthotropy and shared hardening; no refitting.
   - `shape_design/family.py:30-50,141-163` preserves the source values, uniform profile and disabled clamps, with the documented back-face layer swap.
   - All 20 generated sequence byte strings match their recorded files and hashes. Complete command-option comparisons found only the documented geometry changes, plus native seconds for 1 mm.
   - `src/simulations/Sim_Bilayer_Growth.cpp:4954-4959` uses `edge=2*lx*res`; `src/libshell/Geometry.hpp:479-498` gives **111×63 vertices / 13,640 triangles** at 2 mm and **221×125 / 54,560** at 1 mm.
   - Width maps to `u/lx`, feed to `v/ly`. `calibration/ewcal/runner.py:275-279` correctly checks `[220,124]` mm extents.

5. **Toolpaths and masks — verified.**
   - Required: pitch `130*tan(1°)`, F `(104 mm,90 strips)`, C `(200 mm,48 strips,90°)`, symmetric center masks and zero-growth inactive strips.
   - Located at `shape_design/family.py:19-23,54-75,104-137,151-156`; construction matches `src/libshell/ZigZagGrowth.hpp:208-279`.
   - Footprints pass the documented ≥8 mm clearance fixture. Center/edge strip counts are F **44/46**, C **24/24**.
   - Round indices start at zero; activity is `j<4+k` or `j<4-k`. Pass count is `4*(4+|k|)`, hence 16–32.

6. **Turning over — algebraically correct under the stated model assumption.**
   - `shape_design/family.py:84-91,141-144` implements the required reflection and layer swap.
   - With `M=diag(-1,1)`, `M R(θ)=R(-θ) M`; direct indexed segment checks give zero mirror error for F and C.
   - Solver direction is the segment angle plus placement rotation (`src/libshell/ZigZagGrowth.hpp:835`); stretches are `g(1+ortho)` along and `g(1-ortho)` across (`src/simulations/Sim_Bilayer_Growth.cpp:2623-2650`).
   - Therefore **swap, but do not negate, orthotropy**. No sign/direction error found. Physical transfer to turned-over processing remains an acknowledged extrapolation.

7. **Hardening and hit-count acceptance — correct mathematics; numerical equivalence checked without executing C++.**
   - Required per triangle: apply `q(n)=floor+(1-floor)exp(-beta*n)` using previous shared hit count, then increment only for nonzero growth.
   - `src/libshell/ZigZagSequenceGrowth.hpp:653-669` implements this, with nonzero threshold `1e-20`; `src/simulations/Sim_Bilayer_Growth.cpp:2585-2588,2658-2660` applies the timing correctly.
   - Masked zero-growth events do not increment `total_hit_count`; exported totals are the shared history (`src/simulations/Sim_Bilayer_Growth.cpp:1827-1828,1907-1908`).
   - `shape_design/family.py:166-177` matches centroid-to-finite-segment distance, inverse rotation and inclusive half-width membership.
   - The actual JSON route uses the **placement-aware** collector at `src/libshell/ZigZagGrowth.hpp:744-842`, with explicit center zero—not the older bbox-only overload. These centers coincide for this panel.
   - Independent SI-formula emulation versus Python using Float32-exported reference coordinates found **zero count differences across all 18 cases on both grids**.
   - No exact half-width ties found. Minimum boundary gaps were approximately **1.637e-5 mm at 2 mm** and **3.621e-6 mm at 1 mm**. This is not bitwise C++ proof; retain the post-solve equality check.

8. **Acceptance and analysis formulas — verified apart from issue 2.**
   - `calibration/ewcal/runner.py:174-215` requires complete cycles, one final-only solve, independent verification, gradient ≤`5e-14`, `gradient_target` and positive pivots.
   - `src/simulations/Sim_Bilayer_Growth.cpp:2534-2537` accumulates growth in order and solves at the final cycle.
   - Height is an `N`-vertex, mean-removed normal component in mm; proper unweighted Kabsch alignment is reused (`calibration/ewcal/geometry.py:57-70`; `shape_design/analysis.py:19-22`).
   - `shape_design/analysis.py:30-51` implements the six-term quadratic fit, Hessian eigenvalues in mm⁻¹, `K_fit=4ac-b²` in mm⁻² and the 0.2 magnitude-ratio classification.
   - `shape_design/analysis.py:54-69` converts to SI, uses 15 mm material neighborhoods and reference-area weighting; invalid faces remain in the denominator.
   - `shape_design/analysis.py:72-85` performs sample-centered PCA on rows of height fields. Signs are oriented toward positive correlation with `k`; that sign convention is not evidence of a physical trend.
   - No scan/test-truth access or parameter fitting was found in the execution/analysis paths. These descriptors address model shapes; PCA alone does not establish double curvature or validation.

## Non-blocking caveats and specification gaps

9. **Manual progression gate:** `shape_design/run_family.sbatch:14-24` accepts arbitrary stages/cases; `shape_design/run_case.py:35-58` never checks six pilot results. The card’s pilot→sweep condition must therefore be enforced externally. Code provides no automatic campaign gate.

10. **Timeout interpretation:** controller time includes process startup/output, whereas native time concerns optimization and can include verification grace (`src/libshell/NativeHybridShellSolver.hpp:378-409`). Equal limits are not interchangeable. No external retry loop exists; frozen internal continuation controls remain unchanged.

11. **Known-answer coverage:** 33 tests passed. Geometry, masks, frozen values, analytic classifications, curvature units and one-parameter PCA exercise production Python functions. However, `shape_design/tests/test_family.py:71-84` sorts endpoint coordinates independently: its mirror test also accepts an **unmirrored** F/C pattern. Direct indexed checks verified current correctness. Neither JSON parsing nor hit equality was exercised through the compiled solver.

12. **Reporting/mesh-PCA gap:** the card specifies nine-field, joint eighteen-field PCA and fine-mesh ordering persistence, but `shape_design/analysis.py:72-85` is only a generic PCA helper. Accepted-case assembly, plots/correlations and the common representation/basis for comparing different mesh sizes are not implemented or specified here.

13. **“Constant work” wording:** `docs/shape_design/attempt209_double_curvature.md:10` incorrectly says each strip receives eight rounds. Individual regions receive `4±k`; total active strip-passes are **1120,1104,1088** for `k=-4,0,+4`. “Nearly constant total work” is supported, not constant per-strip work.

14. **Identity enforcement:** `shape_design/run_case.py:28-31,46-57` records current identities but does not compare them to the approved plan. Ordinary existing-directory overwrite refusal is present at lines 43–45, but is not an atomic concurrency lock.

## Evidence and limitations

- **Recomputed and matched:** plan SHA-256 `b7ac3e79cc39fbe48121617944cb1b8c3d8b4c3d80dea3f0871e33e5367ced08`; all six `identities` entries; test and launcher hashes; all 20 sequence hashes.
- Binary hash recomputed: `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841`.
- **Inspected, not hash-attested by this plan:** analysis, dry-run implementation, geometry/curvature helpers and C++ sources. Binary-to-current-source build correspondence was not independently established.
- Only permitted solver-free CPU checks ran through `srun-here`: **6014843** (hashes, commands, 33 tests), **6015282** (hit geometry), **6015567** (ROI, mirror-test and curl counterexamples).
- No files edited, no shell solver executed, and no simulation results inferred.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Independent review completed with three blocking findings, code/card evidence, recomputed identities and residual risks."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "srun-here --cpus 1 --mem 4G --time 00:10:00 python -B -",
      "result": "passed",
      "summary": "Three read-only stdin checks: identities/commands/tests; hit geometry; ROI and counterexamples. CPU-only environment and bytecode/cache suppression used."
    },
    {
      "command": "git diff --cached --stat",
      "result": "passed",
      "summary": "No staged changes."
    }
  ],
  "validationOutput": [
    "All recorded hashes and 20 generated sequences matched.",
    "33 tests passed in 21.52 seconds.",
    "No emulated hit-count mismatches or exact half-width ties on either grid.",
    "Confirmed one-sided-hit curl counterexample and 304/608 omitted quadratic-ROI vertices."
  ],
  "residualRisks": [
    "Blocking curl-claim, precision/ROI and scheduler-envelope issues remain unresolved.",
    "Campaign progression and approved-plan enforcement are manual.",
    "Compiled-source correspondence and actual solver output equality remain unverified.",
    "Fine-mesh PCA comparison convention remains unspecified."
  ],
  "noStagedFiles": true,
  "diffSummary": "Read-only review; pre-existing worktree changes untouched.",
  "reviewFindings": [
    "BLOCKING: shape_design/tests/test_family.py:133-148 — 7% surrogate omits actual hit geometry.",
    "BLOCKING: shape_design/analysis.py:27 — Float32 ingestion excludes prescribed ROI boundaries.",
    "BLOCKING: shape_design/run_family.sbatch:9 — 7200-second wall time leaves no mesh-check controller/reporting allowance."
  ],
  "manualNotes": "Card read fully before code. Findings distinguish static verification, solver-free numerical checks and unverified compiled/runtime behavior. Complete artifact returned for persistence to the authoritative output path."
}
```

Verdict: BLOCKING ISSUES