# Independent pre-execution re-review

**BLOCKING ISSUES — do not start shell simulations.**

Artifact: `/gpfs/projects/p33082/PSM2025-zigzag/docs/shape_design/preexecution_rereview.md`

The exact requested plan and its recorded identities match the current files. The precision/ROI repair is verified. The revised timeout protocol provides headroom when its explicit three-hour submission override is used. However, the replacement face-level curl diagnostic still misrepresents one-sided treatment.

## 1. Prior blockers

### 1.1 Curl claim: withdrawn, but replacement diagnostic remains BLOCKING

The card explicitly withdraws the false 7% bound (`docs/shape_design/attempt209_double_curvature.md:7`). Its rectangle and tile statistics correctly reproduce the documented **linearized treatment comparison**, not an equilibrium-shape bound.

However, the card reports:

> “Faces with ≥90% one-sided difference | 0% (`k = 0`)…”  
> `docs/shape_design/attempt209_double_curvature.md:29`

The implementation does not measure that population:

```python
# shape_design/family.py:234-238
hit = np.abs(d0).sum(axis=(1, 2)) > 0
# a face is one-sided if its net difference is >= 90% of its own unsigned (oneface) difference
out['face_one_sided_fraction'] = float(np.mean(
    np.linalg.norm(d1[hit], axis=(1, 2)) >=
    0.9 * np.linalg.norm(d0[hit], axis=(1, 2))))
```

Here `d0` comes from the **separate `oneface` geometry and dose**, not the unsigned contributions of the actual `flip` hits. Mirrored footprints differ.

**Verified counterexample, using production geometry:**

| Quantity | `flip`, `k=0`, 2 mm |
|---|---:|
| Triangles receiving `flip` hits | 10,671 |
| Triangles receiving `oneface` hits | 9,744 |
| `flip`-hit triangles excluded because `oneface` has no hits | 927 |
| Triangles receiving hits exclusively on one physical face | 1,852 |
| Those exclusive triangles inside the treated rectangle | 1,852 |
| `flip`-hit triangles inside that rectangle | 10,365 |
| Reported `face_one_sided_fraction` | 0 |

Thus approximately **17.9% of actually treated ROI triangles receive exclusively one-face hits**, despite the reported zero. This is a direct footprint fact, not an alternative equilibrium prediction.

The prior review’s example remains:

- `(u,v)=(-98.666667,-49.333333)` mm;
- eight back-face hits, zero front-face hits;
- zero `oneface` hits;
- therefore excluded from this diagnostic.

Front-only triangles can also fail the 90% test because the denominator uses a different accumulated dose.

**Consequence:** the zero does not establish absence of locally one-sided treatment. The reported conditional comparison reproduces numerically, but its “own unsigned” explanation and one-sided interpretation are incorrect. **Severity: BLOCKING.**

**Specification gap:** the card does not explicitly define the face statistic’s eligible population or ROI. The code uses whole-panel `oneface` support. That ambiguity is separate from the demonstrated incorrect interpretation.

The saved diagnostic test only checks reference ratios and finite tile values; it does not test this face statistic (`shape_design/tests/test_family.py:163-167`).

### 1.2 Full-precision state, correspondence and ROI: resolved

Required: use native state coordinates, preserve vertex/triangle correspondence, and include the prescribed ROI boundaries.

Verified:

- The generated command enables `-native_hybrid_export_state true`.
- C++ exports schema version 1, units `"metres"`, row-wise `positions`, `reference_positions`, and `faces`, only for an accepted terminal endpoint (`src/simulations/Sim_Bilayer_Growth.cpp:1362-1415`).
- The canonical filename matches `shape_design/run_case.py:26`; boundary conditions are disabled.
- Analysis arrays come from that JSON, multiplied by 1000—not from Float32 VTP coordinates (`shape_design/run_case.py:91-94`).
- Both VTP vertex arrays, both face arrays, and the Python grid are checked before acceptance (`shape_design/run_case.py:95-115`).
- Saved `shape.npz` contains the full-precision reference and aligned deformed coordinates (`shape_design/run_case.py:116-118`).
- ROI tolerance is now `1e-3` mm (`shape_design/analysis.py:15-28`).

Independent checks reproduced:

| Mesh | Vertices | Triangles | Quadratic ROI vertices |
|---|---:|---:|---:|
| 2 mm | 6,993 | 13,640 | **5,353** |
| 1 mm | 27,625 | 54,560 | **21,105** |

The Python grid differs from an independent transcription of the C++ SI-coordinate formula by at most **1.42e-14 mm**; face ordering matches exactly. Float32-scale perturbations no longer remove boundary rows.

An existing calibration output was read strictly as a serialization/correspondence fixture. Its state schema, triangle order and proof matched; maximum state/VTP coordinate differences were **7.39e-6 mm initially** and **7.45e-6 mm finally**. This does not substitute for the required checks on future family outputs.

### 1.3 1 mm timeout envelope: resolved conditionally in the submission protocol

The card now explicitly requires `sbatch -t 03:00:00` for the 7200-second mesh checks (`docs/shape_design/attempt209_double_curvature.md:9,84`).

- The launcher still defaults to **02:00:00**; its three-hour override is an **operator requirement**, not an enforced check (`shape_design/run_family.sbatch:2-11`).
- Controller timing starts before `Popen`, covers the solver process, and terminates its process group on timeout (`calibration/ewcal/runner.py:109-146`).
- Cleanup sends SIGTERM, waits up to three seconds, then sends SIGKILL.
- `run_case.py` installs a SIGTERM handler and attempts failure recording through `finally` (`shape_design/run_case.py:29-34,55-56,123-129`).
- Three hours provides one hour beyond the 7200-second controller allowance for startup, validation and recording.

**Limitation:** no 1 mm job was submitted or observed. Omitting the override recreates the previous scheduling defect. Abrupt termination cannot guarantee a result file, as the repaired card now acknowledges. Native optimization timing and controller elapsed timing remain different.

## 2. Other contract-to-code verification

| Required behavior | Located implementation and finding |
|---|---|
| Frozen parameters, uniform profile, shared Voce history; no refitting | `shape_design/family.py:30-50,141-163`. Values and source identities match. |
| F/C dimensions, pitch, symmetric masks, `4±k` rounds | `shape_design/family.py:19-28,54-75,104-137`. Pass count is `4*(4+abs(k))`; active strip-pass totals agree with 1120/1104/1088 at `k=-4/0/+4`. |
| Back-face reflection and layer swap | `shape_design/family.py:84-91,141-144`. Indexed mirror tests now reject an unmirrored pass (`shape_design/tests/test_family.py:81-91`). Orthotropy swaps layers; it is not negated. |
| C++ grid coordinates and face order | `src/libshell/Geometry.hpp:479-535`; edge length is `2*lx*res` at `src/simulations/Sim_Bilayer_Growth.cpp:4954-4959`. Python indexing and both triangle orientations agree. |
| Ordered hit membership and hardening | `shape_design/family.py:191-220`; placement-aware C++ collector at `src/libshell/ZigZagGrowth.hpp:744-842`. Each face/strip intersection is separate; `q(n)=floor+(1-floor)exp(-beta*n)` uses previous shared count. C++ increments after applying the hit, only for nonzero growth (`src/libshell/ZigZagSequenceGrowth.hpp:653-669`; `src/simulations/Sim_Bilayer_Growth.cpp:2583-2660`). |
| Linearized treatment tensor | Per hit, add `q*g*((1+ortho) ddᵀ+(1-ortho) eeᵀ)` to each layer’s `F×2×2` dimensionless tensor. Production Python implements this. C++ instead updates metrics multiplicatively; the diagnostic correctly discloses its linearization. |
| Solver acceptance | `calibration/ewcal/runner.py:174-215,265-285`: complete cycles, one final-only solve record, finite numerical evidence, independent verification, gradient ≤`5e-14`, `gradient_target`, positive pivots and initial extents. Family-specific correspondence/hit checks occur before status becomes accepted. |
| Height and rigid alignment | `rigid_fit(source,target)` returns a **row-vector** transform. `pos @ rot + t` is correct (`calibration/ewcal/geometry.py:57-70`; `shape_design/run_case.py:116-118`). Height is the mean-removed normal component in mm, publicly **y** (`shape_design/analysis.py:20-23`). |
| Quadratic descriptors | `shape_design/analysis.py:31-52`: six-term unweighted ROI fit; Hessian `[[2a,b],[b,2c]]`; eigenvalues in mm⁻¹, `K_fit=4ac-b²` in mm⁻²; magnitude ratio ≥0.2 required for dome/saddle labels. |
| Local curvature | `shape_design/analysis.py:55-70` converts coordinates to metres and calls the 15 mm material-neighborhood helper. Gaussian curvature is in m⁻²; sign fractions use reference triangle areas, retaining invalid faces in the denominator. |
| PCA | `shape_design/analysis.py:73-86`: rows are height fields, column means are removed, scores are `U*S`, explained ratios are `S²/sum(S²)`. PC signs are oriented toward positive correlation with `k`; this convention is not independent evidence of a trend. |

**Information leakage:** none found in the execution/analysis paths. Although the inherited configuration contains target filenames, these paths call command construction, launch and mechanical acceptance—not target loading, parameter fitting or scan-based selection. Analytic fixtures are tests only.

## 3. Caveats and specification gaps

- **MINOR table error:** `docs/shape_design/attempt209_double_curvature.md:28` assigns the approximately 8.4% tile maximum to `k=0`. Recomputed values are **8.3825% at `k=+1`**, **8.6475% at `k=0`**, and **26.8909% at `k=+4`**.
- **Manual progression:** the card explicitly assigns pilot→sweep enforcement to the lead. Neither launcher nor `run_case.py` checks six accepted pilot results. No automatic authorization for sweep or retries follows from a case’s acceptance.
- **Identity enforcement is relative to the current plan:** `shape_design/run_case.py:73-85` checks file identities, case membership, timeout and generated sequence hash. It records, but does not independently pin, the approved plan digest. Replacing/regenerating the plan is not automatically rejected.
- **Reporting implementation remains incomplete:** the generic PCA helper exists, but accepted-case assembly, nine-/eighteen-case reports, figures and fine-grid projection are not implemented in this scope. The repaired card now specifies the common coarse PCA basis. Exact nested-grid correspondence was verified.
- **Scientific limits remain material:** thickness, crosswise treatment and back-face processing extrapolate beyond calibration. The treatment diagnostic is neither curvature nor an equilibrium bound. Quadratic curvature is a small-slope descriptor; PCA is descriptive. No shape outcome or convergence was established.
- **Runtime coverage:** saved tests do not exercise real family solver output, C++ JSON parsing, SIGTERM cleanup, or the complete `run_case.main` acceptance path. Post-solve hit equality remains essential.
- **Build provenance:** inspected C++ source hashes agree with the existing build-evidence record where listed. This corroborates, but does not independently prove, source-to-binary correspondence; no build or solver execution occurred.

## 4. Exact identities

Independently recomputed:

| Artifact | SHA-256 |
|---|---|
| `run/shape_design/plan/plan.json` | `0ebcd8875fbfdc3a52c743bec23ef0e69488093d0790e43aeb7858c34ab923f8` |
| `run/shape_design/plan/curl_diagnostic.json` | `27d85e6c8e20adf7e6cde6e3765b4235fb5df246dd15832442bb2a7a6a8e9c66` |
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |
| `shape_design/family.py` | `f845bf865d53ff3ec61a10c5ef09d14290757b57ed2cfb46cdb3e36aeb26f472` |
| `shape_design/run_case.py` | `8b0b5fec28507b8d3a2752fa9fb0f79f57e613abcfbe74089d9d4f54e2293bd2` |
| `shape_design/analysis.py` | `f5b485914b1e3e28945aaea92407d913e7c292c616682d799b6c25406f0a2635` |
| `shape_design/dry_run.py` | `3ad22fc98c34de1fe9e17a4ee164c722a5e4eff6b618a2f6c43b9f026d011f96` |
| `shape_design/run_family.sbatch` | `44dade8de7c7e9c791b585c8e81ac614e5a7ceb4c17df4c87a45d1da0018863e` |
| `shape_design/tests/test_family.py` | `fe83692984014d4b8e7d224502388cefd9be23312517ce2edd0408994cd7bc74` |
| Frozen source sequence | `2bdefb2df44bdb28a3a4694d6c04d3951d97619dc79e953a12094b027fa02421` |
| Frozen source configuration | `519e5053eaee2d3e7218e96c208fbe1f899d697cbaf711310f8577a2e85fc591` |

Source paths are those specified in the card under `run/convention_corrected/run6/`.

All **13 plan identity entries**, **20 generated sequence byte strings/hashes**, and recorded case timeouts/pass counts matched. The exact plan digest above identifies the complete case-to-sequence-hash mapping, including both mesh checks.

Complete generated-command comparisons found no removed or unexpected options. Relative to the frozen command, changes are only:

- `-lx 0.11`, `-ly 0.062`, `-h_total 0.00061`;
- `-res 0.00909090909090909` at 2 mm;
- `-res 0.004545454545454545` and native limit 7200 seconds at 1 mm.

The execution parser may serialize timeout values as `3600.0`/`7200.0`; these are numerically unchanged.

BLIS policy is correctly applied **before Python** by `shape_design/run_family.sbatch:20-22` and reinforced before solver launch by `calibration/ewcal/runner.py:102-105`: `BLIS_NUM_THREADS=1`, all five `BLIS_*_NT` overrides unset.

## 5. Verification evidence

- **Slurm 6027664:** read-only CPU identity/command/grid audit; **36 tests passed in 25.87 s**, with `RuntimeWarning` treated as an error; all 11 saved curl diagnostic entries reproduced.
- **Slurm 6028875:** confirmed the one-sided-face counterexample/population and historical state/VTP correspondence.
- **Slurm 6028340:** tool timeout interrupted output collection after queueing. Its Python step completed with exit 0; the supervisor subsequently cancelled the lingering allocation. Findings above use captured results from the other checks.
- Saved tests exercise production geometry, masks, mirror construction, hardening tensors, ROI, analytic dome/saddle/cylinder descriptors, rigid alignment, local curvature and PCA.
- No files edited, solver launched, simulation started, or GPU work performed.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Independent read-only re-review completed: exact identities verified, prior repairs assessed, one blocking diagnostic misinterpretation demonstrated with production geometry, and residual risks documented."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "CPU-only srun-here --cpus 1 --mem 4G --time 00:10:00 python -B -; job 6027664",
      "result": "passed",
      "summary": "Verified plan, 13 identities, 20 sequences, generated option comparisons, both grids and ROI counts; 36 tests passed; reproduced saved diagnostics."
    },
    {
      "command": "CPU-only srun-here --cpus 1 --mem 4G --time 00:05:00 python -B -; job 6028340",
      "result": "failed",
      "summary": "Tool timed out during queued execution/output collection. Python step later completed 0:0; supervisor cleaned up lingering allocation. Uncaptured results were not relied upon."
    },
    {
      "command": "CPU-only srun-here --cpus 1 --mem 4G --time 00:05:00 python -B -; job 6028875",
      "result": "passed",
      "summary": "Confirmed excluded one-sided triangles, historical native-state schema/VTP correspondence, and inspected C++ source identities."
    },
    {
      "command": "CPU-only python -B -; single-point production treatment check",
      "result": "passed",
      "summary": "Prior counterexample retains eight back hits, zero front hits and zero oneface reference hits."
    },
    {
      "command": "git diff --cached --stat",
      "result": "passed",
      "summary": "No staged changes."
    }
  ],
  "validationOutput": [
    "Requested plan SHA-256 matched exactly.",
    "36 saved tests passed with RuntimeWarning treated as an error.",
    "ROI counts are 5353 and 21105; C++-formula coordinate agreement is within 1.42e-14 mm.",
    "At flip k=0, 1852 ROI triangles receive exclusively one-face hits despite a reported face_one_sided_fraction of zero."
  ],
  "residualRisks": [
    "Blocking face-level diagnostic interpretation remains unresolved.",
    "Three-hour mesh-check submission and pilot-to-sweep gating require external enforcement.",
    "No real family solver output or scheduler-termination behavior was exercised.",
    "Reporting assembly and fine-mesh PCA projection are not implemented in the reviewed scope.",
    "Physical extrapolation and descriptive-analysis limitations remain."
  ],
  "noStagedFiles": true,
  "diffSummary": "Read-only review; existing worktree changes untouched.",
  "reviewFindings": [
    "BLOCKING: shape_design/family.py:234-238 excludes flip-only treatment and uses a different-dose reference while reporting a one-sided-treatment fraction.",
    "MINOR: docs/shape_design/attempt209_double_curvature.md:28 attributes the minimum tile maximum to the wrong k.",
    "Verified: native-state precision, correspondence logic, grid ordering, ROI repair, rigid-fit convention and exact reviewed identities."
  ],
  "manualNotes": "Card and prior review were read before implementation. Static verification, captured CPU checks and unverified runtime behavior are distinguished. Complete artifact returned for persistence to /gpfs/projects/p33082/PSM2025-zigzag/docs/shape_design/preexecution_rereview.md; no files were edited directly."
}
```

Verdict: BLOCKING ISSUES