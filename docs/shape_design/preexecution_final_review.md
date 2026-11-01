# Independent final pre-execution review

**APPROVED_WITH_CAVEATS — no blocking issue found in this targeted re-review.**

Artifact: `/gpfs/projects/p33082/PSM2025-zigzag/docs/shape_design/preexecution_final_review.md`

Approval applies to the exact frozen plan below and the staged protocol. It does **not** establish an equilibrium-shape outcome, authorize bypassing the pilot gate, or certify the unfinished reporting workflow.

## 1. Blocking diagnostic issue: resolved

### Removed statistic and corrected disclosure

- `shape_design/family.py:223-248` now returns only rectangle/tile treatment comparisons. The invalid `face_one_sided_fraction` calculation is absent.
- `shape_design/dry_run.py:34-40` saves and prints the remaining diagnostics without that statistic.
- All 11 saved diagnostic entries omit it.
- `docs/shape_design/attempt209_double_curvature.md:30` explicitly withdraws the earlier zero-one-sided-face claim and explains the different-footprint denominator error.

Independent production-path checks reproduced:

| Quantity, `flip`, `k=0`, 2 mm | Verified value |
|---|---:|
| Exclusively front- or back-hit triangles inside the rectangle | 1,852 |
| Flip-hit triangles inside the rectangle | 10,365 |
| All triangles inside the rectangle | 10,400 |
| Exclusive-hit reference area | 3,704 mm² |
| Flip-hit reference area | 20,730 mm² |
| Whole-rectangle reference area | 20,800 mm² |

Thus the card’s **17.9%** is correctly **1,852/10,365 = 17.8678% of flip-hit ROI area**, not of the entire rectangle. The whole-rectangle fraction is **17.8077%**. Triangle counts and reference-area fractions coincide here because each triangle has area 2 mm².

The original counterexample remains: the centroid  
`(-98.66666666666667, -49.333333333333336)` mm receives **eight back hits, zero front hits, and zero oneface-reference hits**. Across the panel, 927 flip-hit triangles have no oneface-reference hits. Removing the misleading statistic—not asserting that this geometry disappeared—correctly resolves the blocker.

### Regression coverage

`shape_design/tests/test_family.py:163-183` now:

- asserts the removed diagnostic key is absent;
- locates the counterexample on the production grid;
- uses production pass construction and treatment functions;
- asserts `(back_hits, front_hits) == (8, 0)` and zero oneface hits.

Separating passes resets hardening, but this test uses only hit counts; that does not compromise the asserted coverage property. The full-population/area counts above were independently checked, not asserted by this saved single-point test.

### Corrected minimum

The card correctly assigns the minimum tile maximum to **`k=+1`**, not `k=0`:

- `k=+1`: **8.3824779%**
- `k=0`: **8.6475343%**
- `k=+4`: **26.8908878%**

All 11 saved diagnostic entries reproduced through production treatment tensors and independently calculated reference-area aggregation.

## 2. Remaining diagnostic limitations and specification gaps

The implemented comparison is

\[
R_A(k)=
\frac{\left\|\operatorname{mean}_{f\in A}
(T_f^{flip}-B_f^{flip})\right\|_F}
{\left\|\operatorname{mean}_{f\in A}
(T_f^{oneface}-B_f^{oneface})\right\|_F}.
\]

`shape_design/family.py:226-244` implements the **Frobenius norm after tensor averaging**, using the same spatial centroid-selected region for both variants. Unlike the removed statistic, it does not discard individual flip-only triangles based on oneface support.

However:

- Spatial cancellation occurs **before** taking the norm. A small rectangle/tile ratio cannot bound face-level imbalance.
- The denominator remains treatment from a separate geometry/dose. This is a between-variant aggregate comparison, not the fraction of locally uncancelled treatment.
- It is an additive small-strain treatment proxy, not the solver’s multiplicative metric update, curvature, equilibrium response, or convergence evidence.

**Non-blocking specification gap:** the card does not explicitly describe zero-reference tile exclusion. The implementation says:

```python
# shape_design/family.py:240-242
ref = norm(d0[m].mean(axis=0))
if ref > 0:
    out['tiles'].append(norm(d1[m].mean(axis=0)) / ref)
```

Consequently, the median/maximum use **46 tiles at `k=-4`**, **42 at `k=+4`**, and **50 otherwise**. Both reviewed mesh sizes have these counts. Every omitted tile was independently verified to have **zero flip hits**, so no flip-only treatment is hidden by this omission in the reviewed cases.

The card’s “1 mm values are similar” is qualitative, not numerical stability evidence. Only `k=±4` were prepared at 1 mm. Tile maxima change from **8.8805% to 12.4658%** at `k=-4`, and **26.8909% to 18.6980%** at `k=+4`; rectangle ratios are much closer. No convergence claim is justified.

## 3. Contract-to-code checks

| Required behavior | Located implementation and verification |
|---|---|
| Frozen parameters, uniform profile, shared Voce history; no refitting | `shape_design/family.py:30-50,141-163`. Source values and identities match. |
| Pitch `130 tan(1°)` mm; F/C geometry; symmetric masks | `shape_design/family.py:19-28,54-137`. Round index starts at zero; activity is `j<4+k` centrally and `j<4-k` at edges. Pass count is `4*(4+abs(k))`; active strip-pass totals reproduce 1120/1104/1088. |
| Back-face reflection and layer swap | `shape_design/family.py:84-91,141-144`. Rotation and tilt change sign; growth and orthotropy swap layers. Indexed mirror fixtures exercise production functions. |
| Per-triangle hit membership and ordered hardening | `shape_design/family.py:191-215`: input is `F×2` centroid coordinates in mm; count is length `F`; layer tensors are `F×2×2`, dimensionless. Each active strip intersection uses `q(n)=floor+(1-floor)exp(-beta*n)` **before** incrementing the shared count. |
| Per-hit tensor | Adds `q*g*((1+ortho)ddᵀ+(1-ortho)eeᵀ)` to each layer. Single/repeated-hit and same-footprint front/back fixtures test the stated formulas. C++ timing and directions agree at `src/libshell/ZigZagSequenceGrowth.hpp:653-669` and `src/simulations/Sim_Bilayer_Growth.cpp:2582-2660`; this is static verification, not compiled execution. |
| Grid dimensions and ordering | `shape_design/family.py:166-182` matches the inspected C++ construction. Reproduced 6,993 vertices/13,640 triangles at 2 mm and 27,625/54,560 at 1 mm. |
| Numerical acceptance | `calibration/ewcal/runner.py:174-215,265-285`: complete cycles, one final-only solve, finite evidence, independent verification, gradient ≤`5e-14`, `gradient_target`, positive pivots, and initial extents. Family correspondence/hit checks precede accepted status. |
| Full-precision analysis input | `shape_design/run_case.py:91-118`: native JSON metres → mm; state/VTP differences ≤`1e-4` mm, exact face correspondence, reference/grid differences ≤`1e-9` mm, and per-face hit equality. No Float32 VTP coordinates become the analysis arrays. |
| Height | `shape_design/run_case.py:116-118`; `shape_design/analysis.py:20-23`: proper unweighted full-grid Kabsch alignment; normal coordinate becomes an `N`-entry, mean-removed height field in mm, publicly **y**. |
| Quadratic descriptors | `shape_design/analysis.py:26-52`: inclusive ROI with `1e-3` mm tolerance; six-term least squares; Hessian `[[2a,b],[b,2c]]`; curvatures in mm⁻¹; `K_fit=4ac-b²` in mm⁻²; absolute-curvature ratio ≥0.2 for dome/saddle labels. ROI counts reproduce **5,353/21,105**. |
| Local sign fractions | `shape_design/analysis.py:55-70`: converts positions to metres and uses 15 mm material neighborhoods. Gaussian curvature is m⁻². Denominator is **all reference triangle area in the centroid-selected ROI**, including invalid faces—not valid-only or hit-only area. |
| PCA | `shape_design/analysis.py:73-86`: height fields are rows; subtracts the across-sample mean, uses scores `U*S` and explained ratios `S²/sum(S²)`. Signs are oriented toward positive correlation with `k`; that convention is not independent trend evidence. |
| Fine-grid comparison | Exact nesting of the two production grids was checked. The prescribed restriction, mean removal and projection onto the coarse flip PCA basis remain **unimplemented** in the reviewed reporting scope. |

**Information leakage:** none found in the reviewed execution/analysis paths. The inherited configuration contains scan/target filenames, but these paths do not load those targets, fit parameters, or select cases using measured truth. Analytic surfaces are test fixtures only.

## 4. Operational caveats remain

1. **Manual campaign gate.**  
   `docs/shape_design/attempt209_double_curvature.md:14,81-82` assigns enforcement to the lead. Neither `shape_design/run_family.sbatch:16-26` nor `shape_design/run_case.py:55-85` checks six accepted pilot results. Sweep authorization remains conditional; failure means stop without campaign retries or tolerance relaxation.

2. **Required 1 mm Slurm override.**  
   `shape_design/run_family.sbatch:11` still defaults to two hours. The card’s explicit **`-t 03:00:00`** submission override is required for 7200-second mesh checks. It is documented, not enforced. Controller and native-solver timing differ. SIGTERM cleanup/failure recording is attempted; abrupt termination cannot guarantee a record.

3. **Identity enforcement trusts the current plan.**  
   `shape_design/run_case.py:70-85` checks identities, case membership, timeout and sequence hash against the plan it reads. It records, but does not independently pin, the approved plan digest. Replacing the plan is not automatically rejected.

4. **Partial analysis implementation.**  
   Per-shape descriptors and generic PCA exist. Accepted-case assembly, nine-/eighteen-case reporting, figures, correlations and fine-grid projection are not implemented here. Approval does not certify those absent paths.

5. **Unverified runtime/physical behavior.**  
   No family solver output, compiled JSON parsing, complete `run_case.main` execution or scheduler termination was exercised. Post-solve correspondence and hit checks remain essential. Thickness, crosswise processing and back-face processing remain acknowledged calibration extrapolations.

BLIS single-threading and removal of the five `BLIS_*_NT` overrides occur before Python in the launcher and are reinforced before solver launch.

## 5. Exact identities and actual scope

Independently recomputed:

| Artifact | SHA-256 |
|---|---|
| Reviewed card | `e3404ac5bebb3bc025aed0edd21cbe05819e8b3b313f426ed86138d4e0945bb1` |
| `run/shape_design/plan/plan.json` | `0bb8a542cd8f1656a193aaa7dd1ed5e9d458644a37857694f7d59f014ce1e311` |
| `run/shape_design/plan/curl_diagnostic.json` | `578c9c3c32c1550dc06ce0f5cfd642fcb58b391b94e01036a670927a2a12fde3` |
| `shape_design/family.py` | `d6b63c429f8a0c088ac4ffba845c2e98f1c77191fdb1533671dd81a7eccfc266` |
| `shape_design/dry_run.py` | `789dbb43d684f88e8a0ce959ddb866d29a855abac7d7e811997740c5529e1d9a` |
| `shape_design/tests/test_family.py` | `54cf80f5beddc45a5d0118409740ca5229db563b29792fb352808b5c121b4abc` |
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |

**All 13 plan identity entries matched**, including the remaining runner, analysis, launcher, geometry, curvature/common helpers and frozen source sequence/configuration. Redundant test/launcher entries also matched.

All **20 generated sequence byte strings and hashes**, case membership, pass counts, rounds and timeouts matched. Complete command-option comparisons found no removed or unexpected options: only the documented geometry changes and 7200-second native limit for 1 mm differ from the frozen command.

`run_case.py`, `analysis.py`, the launcher, binary and frozen inputs match their prior published hashes. Changed family/preparation/test sources were read fully. The saved test delta reconstructs the prior test hash exactly when the new omission assertion and back-only test are removed. Inspected C++ source fingerprints match the existing build-evidence record; that does not independently prove source-to-binary correspondence.

This was a targeted review, not an audit of unrelated worktree changes.

## 6. Verification evidence

- **Slurm 6032809:** identity, 20-case command/sequence and grid checks passed. A reviewer-only aggregation-fixture indexing typo then stopped the check before diagnostics/tests.
- **Slurm 6033205:** corrected read-only fixture reproduced all 11 diagnostics, counterexample populations/areas and the corrected minimum; **37 tests passed in 25.29 s**, with `RuntimeWarning` treated as an error. Final plan and identity checks still matched.
- A login-node `python` lookup was unavailable; subsequent small hash checks used the explicit installed interpreter.
- CPU-only environment, one-core compute allocations, bytecode/cache suppression; no solver, simulation, GPU work, file edits or plan regeneration.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Targeted independent review completed: prior diagnostic blocker resolved, exact frozen identities and sequences verified, 37 tests passed, and remaining operational/scientific limitations documented."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "CPU-only srun-here --cpus 1 --mem 4G --time 00:08:00; Slurm 6032809",
      "result": "failed",
      "summary": "Identity, sequence, command and grid checks passed; reviewer-only aggregation fixture stopped on an indexing typo."
    },
    {
      "command": "CPU-only srun-here --cpus 1 --mem 4G --time 00:08:00; Slurm 6033205",
      "result": "passed",
      "summary": "All 11 diagnostics reproduced; coverage/area counterexample confirmed; 37 tests passed in 25.29 s; final identities matched."
    },
    {
      "command": "Login-node python lookup",
      "result": "failed",
      "summary": "python was unavailable on PATH; small read-only checks subsequently used the explicit installed interpreter."
    },
    {
      "command": "Read-only source searches, explicit-interpreter SHA checks, and git diff --cached --stat",
      "result": "passed",
      "summary": "Confirmed statistic removal, source fingerprints and no staged changes."
    }
  ],
  "validationOutput": [
    "Requested plan SHA-256 matched exactly; all 13 identities and 20 sequence bytes/hashes matched.",
    "1852/10365 flip-hit ROI triangles = 17.8678%; whole-ROI fraction = 17.8077%.",
    "Counterexample has eight back hits, zero front hits and zero oneface-reference hits.",
    "Minimum tile maximum is 8.3824779% at k=+1.",
    "37 production-path solver-free tests passed."
  ],
  "residualRisks": [
    "Pilot progression, approved-plan preservation and the three-hour 1 mm submission override require external enforcement.",
    "Aggregate treatment ratios permit spatial cancellation and are not equilibrium or local-imbalance bounds.",
    "Zero-reference tile exclusion is not explicit in the card; omitted reviewed tiles have no flip hits.",
    "Reporting assembly and fine-grid PCA projection remain unimplemented.",
    "Compiled/runtime behavior, source-to-binary correspondence and physical extrapolations remain unproven."
  ],
  "noStagedFiles": true,
  "diffSummary": "Read-only review; existing worktree changes untouched.",
  "reviewFindings": [
    "No blocking issue found in the targeted scope.",
    "Resolved: invalid face-level diagnostic removed and counterexample correctly disclosed.",
    "Resolved: minimum tile maximum correctly attributed to k=+1.",
    "Non-blocking specification gaps and existing operational caveats remain."
  ],
  "manualNotes": "APPROVED_WITH_CAVEATS for the exact reviewed plan and staged protocol. Card and prior reviews were read before code. Complete artifact returned for persistence to /gpfs/projects/p33082/PSM2025-zigzag/docs/shape_design/preexecution_final_review.md; no files were edited directly."
}
```

Verdict: PASS WITH MINOR