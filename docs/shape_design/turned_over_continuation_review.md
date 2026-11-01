# Turned-over-only continuation review

**APPROVED_WITH_CAVEATS — no blocking issue found for the revised scope.**

Artifact: `/gpfs/projects/p33082/PSM2025-zigzag/docs/shape_design/turned_over_continuation_review.md`

The original six-pilot gate **failed**. This approval implements the explicitly disclosed researcher revision; it does not retroactively approve the original sweep or erase `oneface k=-4`.

## 1. Independently verified pilot evidence

Production `runner.check_output` accepted all five saved successful pilots. Additional checks reproduced native-state/VTP/grid correspondence, per-face hits, saved full-precision aligned arrays, quadratic descriptors and local-curvature summaries.

| Flip pilot | Recorded elapsed time, s | Final gradient norm | `K_fit`, mm⁻² | Curvature ratio | Label |
|---|---:|---:|---:|---:|---|
| `k=-4` | 244.642 | 4.057070e−16 | −2.520991e−8 | 0.206925 | saddle-like |
| `k=0` | 628.129 | 1.702824e−17 | 2.931963e−7 | 0.896455 | dome-like |
| `k=+4` | 990.370 | 1.186145e−14 | 3.423572e−7 | 0.830395 | dome-like |

For each accepted pilot:

- Completed `final_only` manifest: one equilibrium solve, correct cycle count, return code zero, `gradient_target`, `positive_pivots`, finite evidence and gradient ≤ **5e−14**.
- Independent verification counts: **one evaluation and one Hessian**.
- Native arrays: **6,993 vertices × 3 coordinates**, **13,640 triangles × 3 indices**.
- Exact initial/final/native/Python face correspondence and **zero hit-count mismatches**.
- Maximum state/VTP coordinate discrepancy across accepted cases: **3.725212e−6 mm**, below **1e−4 mm**.
- Reference/Python-grid discrepancy: **1.421086e−14 mm**, below **1e−9 mm**.
- Recomputed native-to-NPZ aligned coordinates matched exactly.

The two accepted oneface pilots also reproduced their reported cylinder-like labels and ratios, **0.049259** and **0.079111**.

### Original failure remains a failure

`run/shape_design/pilot/oneface_k-4/result.json` records **2365.157 s**, failed status and solver exit **−6**. There is no accepted `shape.npz` or completed `sequence_result.json`; production acceptance rejects it.

`run/shape_design/pilot/oneface_k-4/solver.log:38` states:

> `equilibrium corrector failed at cycle 32, lambda=0.921875, gradient_norm=0.000000, termination=indefinite_curvature.`

The final native proof gives gradient **2.1968891199205567e−12**, not zero, and:

> `persistent indefinite curvature detected (consecutive_shifts=3)`

The displayed `0.000000` is rounded. Earlier accepted internal continuation steps are not an accepted final shape or campaign retries.

**Provenance qualification:** `termination.json` conservatively says `numerical:false` / `unexplained_process_exit`. `calibration/ewcal/runner.py:46-67` recognizes only a particular historical time-cap abort. The numerical mechanism here is supported by the detailed native proof and solver log—not by the exit signal alone. This does not change the failed gate.

## 2. Revised authorization and proposed arguments

`docs/shape_design/attempt209_double_curvature.md:4,6,88-93` explicitly distinguishes the failed original gate from the subsequent researcher decision. No numerical acceptance threshold, solver setting or frozen parameter was relaxed.

The supervisor supplied these exact proposed commands. They were inspected, **not executed**:

```bash
sbatch --wait --parsable --array=0-5 shape_design/run_family.sbatch \
  sweep_flip 2.0 3600 flip:-3 flip:-2 flip:-1 flip:1 flip:2 flip:3
```

Only after **all six actual result records pass every acceptance check**:

```bash
sbatch --wait --parsable -t 03:00:00 --array=0-1 shape_design/run_family.sbatch \
  mesh_flip 1.0 7200 flip:-4 flip:4
```

Verified:

- Array indices map to exactly the six missing coarse flip cases and the two prescribed fine cases.
- Every case, generated sequence and timeout belongs to the unchanged frozen plan.
- `run/shape_design/sweep_flip` and `run/shape_design/mesh_flip` were absent at inspection.
- Launcher resources are normal partition, one 8-CPU/16-GB task; BLIS single-threading and removal of the five `BLIS_*_NT` overrides precede Python.
- The fine command supplies the mandatory three-hour override.

**Mandatory operational caveats:** campaign gating remains manual. `shape_design/run_family.sbatch:16-26` dispatches arguments without examining previous results; `shape_design/run_case.py:73-85` checks membership in the original 20-case plan, not the revised authorization subset. Scheduler success alone is insufficient. Any failed, timed-out or missing new record stops progression after submitted work finishes. No retries, pilot reruns or additional oneface cases are authorized.

The runner records the current plan digest but does not independently pin this approved digest. Preserve and recheck the identities below immediately before submission.

## 3. Contract-to-code verification

| Required behavior, shapes and timing | Located implementation and review result |
|---|---|
| Frozen growth/orthotropy and shared Voce history; no refitting or target-data access | `shape_design/family.py:30-50,141-163`. Source identities and values match. |
| Pitch `130 tan(1°)` mm; 2-mm band; central/edge rounds `4+k`/`4-k`; zero-based round activity; four passes per round | `shape_design/family.py:19-28,54-137`. Production fixtures exercise geometry, masks, counts and indexed reflection. |
| Back pass reflects `u`, negates rotation/tilt and swaps layer parameters | `shape_design/family.py:84-91,141-144`. Indexed mirror fixtures pass. |
| Ordered per-hit hardening: apply `q(n)=exp(-βn)` before incrementing shared count; counts have length F, layer tensors shape F×2×2 | `shape_design/family.py:191-215`. Single-hit, repeated-hit `1+exp(-β)` and same-footprint front/back `1-exp(-β)` fixtures pass. |
| Treatment proxy: norm **after** averaging top-minus-bottom tensors, divided by same-region oneface norm | `shape_design/family.py:223-248`. Unchanged source and diagnostic hashes; previous independent diagnostic reproduction is inherited, not rerun here. Cancellation and distinct-dose denominator limitations remain. |
| Acceptance only after complete solver evidence, correspondence and exact per-face hits | `calibration/ewcal/runner.py:174-285`; `shape_design/run_case.py:89-122`. Rechecked against all saved pilots. |
| Height: full-grid proper unweighted Kabsch alignment of native N×3 arrays; metres→mm; normal coordinate then whole-grid mean removal | `calibration/ewcal/geometry.py:57-63`; `shape_design/run_case.py:91-118`; `shape_design/analysis.py:20-23`. Reproduced saved arrays. Public height is **y**. |
| Six-term quadratic least squares over `|u|≤100`, `|v|≤52` mm with 1e−3-mm boundary tolerance; Hessian `[[2a,b],[b,2c]]`; `K_fit=4ac-b²`; magnitude ratio ≥0.2 | `shape_design/analysis.py:15-52`. Curvatures are mm⁻¹, Gaussian descriptor mm⁻², residual RMS mm. Saved coefficients, labels and summaries reproduced. |
| Local curvature: 15-mm material neighborhoods; positions converted to metres; sign fractions weighted by reference triangle area in centroid-selected ROI | `shape_design/analysis.py:55-70`; `dome_bo/dome_curvature.py:9-51`. Gaussian curvature is m⁻²; invalid faces remain in the denominator. Saved summaries reproduced. |
| PCA only from all nine accepted coarse flip fields: 9×6,993 matrix, across-sample centering, scores `U*S`, variance ratios `S²/sum(S²)` | `shape_design/analysis.py:73-86`. Generic implementation and known-answer fixture pass; accepted-case assembly and complete reporting remain absent. |
| Fine comparison: align full fine mesh, restrict to coarse vertices, remove restricted-field mean, project using the coarse flip PCA mean/components | Required by `docs/shape_design/attempt209_double_curvature.md:89-93`; projection/reporting remains unimplemented and is **not certified** by this approval. |

All **20** saved sequence byte strings/hashes, pass counts, round counts, timeouts and documented command-option differences reproduced.

## 4. Analysis limitations, leakage and minor finding

### Interpretation

The card’s quadratic numbers are correct, but `flip k=-4` is only **0.006925 above** the 0.2 ratio boundary. Its reproduced local ROI fractions are **40.64% negative K and 59.36% positive K**. Thus “saddle-like” is a global quadratic descriptor, not a claim of uniformly saddle-shaped local geometry.

For `k=0/+4`, positive local-area fractions are **100% / 85.29%**. All three have zero invalid ROI fraction.

The nine-point family would remain deterministic, exploratory and outcome-adapted at the variant-selection stage. The revised card discloses that adaptation. It cannot establish continuity, mesh convergence, physical validity or an unbiased completed joint-variant comparison.

**Information leakage:** none found in the reviewed paths. The inherited configuration contains target references, but these execution paths do not load scans, optimize against truth or refit parameters. Analytic surfaces are test fixtures only.

### Non-blocking specification gaps

The card does not fully specify the local-fraction invalid-area denominator, PCA sign convention or correlation statistic. Current code retains invalid area in the denominator and orients PC signs toward positive correlation with `k`. That sign convention is not independent trend evidence. Reporting assembly and fine projection remain unfinished.

### MINOR — visualization coordinate origin

Card line 51 correctly names public axes; project `AGENTS.md` additionally requires **`z=v+65`**.

`shape_design/plot_pilot_shapes.py:46-58` plots `p[:,1]` and `ref[:,1]` while labeling them `Feed z (mm)`. The existing image therefore shows centered **−62…62 mm**, rather than public **3…127 mm**. This is a presentation-origin discrepancy only: no height, curvature, acceptance or PCA number changes.

The script is not imported or called by the solver launcher, runner or package initializer. Its figure/provenance inputs match the three accepted pilot NPZ files. It does not alter solver authorization.

## 5. Exact identities

All 13 plan identity entries independently matched current files and every pilot record.

| Artifact | SHA-256 |
|---|---|
| Revised card | `b26d339c72de32fad50c5e60bcb605276bf4b7b6db68a98be39f3c50667f093f` |
| `run/shape_design/plan/plan.json` | `0bb8a542cd8f1656a193aaa7dd1ed5e9d458644a37857694f7d59f014ce1e311` |
| Previous final review | `168dceeae2c845851f4f08909f0390b78ebef273d5d694d9f63675c3877a37ea` |
| `shape_design/family.py` | `d6b63c429f8a0c088ac4ffba845c2e98f1c77191fdb1533671dd81a7eccfc266` |
| `shape_design/run_case.py` | `8b0b5fec28507b8d3a2752fa9fb0f79f57e613abcfbe74089d9d4f54e2293bd2` |
| `shape_design/analysis.py` | `f5b485914b1e3e28945aaea92407d913e7c292c616682d799b6c25406f0a2635` |
| `shape_design/dry_run.py` | `789dbb43d684f88e8a0ce959ddb866d29a855abac7d7e811997740c5529e1d9a` |
| `shape_design/run_family.sbatch` | `44dade8de7c7e9c791b585c8e81ac614e5a7ceb4c17df4c87a45d1da0018863e` |
| `shape_design/tests/test_family.py` | `54cf80f5beddc45a5d0118409740ca5229db563b29792fb352808b5c121b4abc` |
| `dome_bo/dome_curvature.py` | `85b0e6ac1942b3c01be28e321b839694300f7ec5419d7a9852ea70c7838ed180` |
| `calibration/ewcal/runner.py` | `3e4f2067f7db490c98b57de7ee1b86e18fc2ca51e57d26b4c1ebf859de405a43` |
| `calibration/ewcal/geometry.py` | `eebb68941f953806f64a086f3b75696756be435237e5c67027b67e3e236006ea` |
| `calibration/ewcal/common.py` | `abd5b2e4206b2695e692c5dbfaa0ca8aed0a42f2a1037fcc426464b5243b81c7` |
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |
| `run/convention_corrected/run6/evaluations/attempt_0001/case_00/sequence.json` | `2bdefb2df44bdb28a3a4694d6c04d3951d97619dc79e953a12094b027fa02421` |
| `run/convention_corrected/run6/proposed_config.json` | `519e5053eaee2d3e7218e96c208fbe1f899d697cbaf711310f8577a2e85fc591` |
| `run/shape_design/plan/curl_diagnostic.json` | `578c9c3c32c1550dc06ce0f5cfd642fcb58b391b94e01036a670927a2a12fde3` |
| `shape_design/plot_pilot.py` | `29283c88d6a0146d887778e986e24b483f6c08d389268158a772d88f3e82f179` |
| `shape_design/plot_pilot_shapes.py` | `170f7c96e883b880b07832dcdb1166480095775534b66e794461d5347659219f` |
| `run/shape_design/pilot_analysis/results.json` | `ba89ee5f7f5b5b89b5f5ba5d7a76e6e87d663019edf2befe2d216e23e366b7b3` |
| `run/shape_design/pilot_visualization/turned_over_pilot_shapes.provenance.json` | `8546d102c4a7c4edf73b4d24c073553e3c3d1e46d84ccbeedf0e4a9d46f1c9ee` |

### Complete pilot evidence fingerprints

These pin every immediate file in each pilot directory, including commands, sequences, logs, manifests, proof/state files and saved shapes where present.

Definition: SHA-256 of UTF-8 `json.dumps({filename: SHA256(file_bytes)}, sort_keys=True, separators=(',', ':'))`.

| Directory under `run/shape_design/pilot/` | Files | Evidence fingerprint |
|---|---:|---|
| `flip_k-4` | 20 | `e67b7458005834ab15627844b4ef0477b3ffabd9e92edc51c601e7152b086f5a` |
| `flip_k+0` | 25 | `67ccac0ecb6a566bf1ca77076f2176be6dc4203ec6036724520f00d81cecfccb` |
| `flip_k+4` | 27 | `e4467fffd04f91f3ba7b754f2b2ad54d7499da18694906576b8179a7d1365c41` |
| `oneface_k-4` | 47 | `bf4f9518c56dbb40e63861cff9db1b5dbbb22d717e27f0e5a384ab989e58c0ea` |
| `oneface_k+0` | 37 | `4ad857cdf2e2aa4097b04cdda609ea47aacdca40103fa15460a1be6f59acb6e1` |
| `oneface_k+4` | 18 | `9a3ee55cb45730c85a6311f9a1a41f5e8137ec7fddf2e9df56b98ca285ea3610` |

Matching hashes establish artifact consistency, not independent source-to-binary build attestation or physical validity.

## 6. Verification scope

- Read card and previous review before code.
- **Slurm 6065586:** identities, all 20 sequences/command definitions and proposed mappings passed. Reviewer fixture then stopped because it used integer `3600` rather than the production CLI’s float `3600.0` for bytewise command comparison.
- **Slurm 6065718:** corrected fixture reproduced all six pilot provenance chains, five accepted outputs and descriptors, and the rejected failed case. **37 tests passed in 29.51 s**, with `RuntimeWarning` treated as an error.
- Known-answer tests exercise production geometry, hardening, alignment, quadratic/local curvature and PCA paths. They do not implement or test the absent fine-grid projection.
- CPU-only, one-core inspection; bytecode and pytest cache disabled. No solver, simulations, GPU work, file edits or plan regeneration. Unrelated worktree changes were not audited.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Completed independent revised-scope review; reproduced pilot acceptance and failure evidence, checked frozen identities and exact proposed arrays, and documented residual risks."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "Read-only source, artifact, hash and staging inspection",
      "result": "passed",
      "summary": "Card and plan pinned; continuation directories absent; no staged files."
    },
    {
      "command": "CPU-only srun-here inspection, Slurm 6065586",
      "result": "failed",
      "summary": "Identity, 20-case sequence/command and array checks passed; reviewer fixture stopped on integer-versus-float timeout serialization."
    },
    {
      "command": "CPU-only srun-here inspection and pytest, Slurm 6065718",
      "result": "passed",
      "summary": "All pilot evidence checked; five accepted outputs reproduced, original failure rejected; 37 tests passed in 29.51 seconds."
    }
  ],
  "validationOutput": [
    "All 13 frozen plan identities matched current files and all six pilot records.",
    "All 20 generated sequence identities and documented command differences matched.",
    "Five accepted pilots had exact 13,640-face hit correspondence and exact reproduced aligned NPZ coordinates.",
    "Three flip pilots satisfy the unchanged 5e-14 gradient gate.",
    "Original oneface k=-4 remains failed with indefinite-curvature evidence.",
    "Proposed arrays contain only the authorized six coarse and two conditional fine flip cases."
  ],
  "residualRisks": [
    "Manual enforcement remains necessary for case scope, accepted-record progression, no retries and approved-plan preservation.",
    "Fine projection and complete nine-case reporting remain unimplemented and uncertified.",
    "Scheduler termination behavior and source-to-binary build correspondence were not independently exercised.",
    "Quadratic labels do not establish uniform local curvature, continuity, convergence or physical validation.",
    "Visualization-only public z origin is offset by 65 mm from project convention."
  ],
  "noStagedFiles": true,
  "diffSummary": "Read-only review; no files changed.",
  "reviewFindings": [
    "No blocking issue for the explicitly revised turned-over-only continuation.",
    "MINOR: shape_design/plot_pilot_shapes.py:46-58 labels centered v as public z without the required +65 mm offset.",
    "Original pilot gate failure and outcome-adapted scope revision are explicitly disclosed."
  ],
  "manualNotes": "APPROVED_WITH_CAVEATS for the pinned revised card, frozen plan and exact conditional submission scope. Complete artifact returned for persistence to /gpfs/projects/p33082/PSM2025-zigzag/docs/shape_design/turned_over_continuation_review.md; no direct write performed."
}
```

Verdict: PASS WITH MINOR