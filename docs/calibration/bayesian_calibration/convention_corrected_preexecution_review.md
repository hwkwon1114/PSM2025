# Convention-corrected pre-execution review

**Verdict: CHANGES_REQUESTED — BLOCKED.**

The proposed geometry and preserved-sample transformation are coherent. The runner does **not** preserve the archived solver recipe or executable. Repair those two blockers before launching any of the three evaluations.

## Scope and authority

Authoritative protocol: `docs/calibration/bayesian_calibration/convention_corrected_investigation.md`.

I also read `docs/calibration/bayesian_calibration/gripper_coordinate_audit.md`. Its earlier requirement for physical orientation confirmation **before simulation is superseded**. Photo/face uncertainty limits interpretation, not this numerical diagnostic.

Reviewed:

- `calibration/bo/run_convention_corrected.py` — untracked, 43-line version using raw config JSON at line 12.
- `calibration/analysis/preserve_samples_putong_flip.py`.
- Current Run6/9/10 configs, archived requests, selected sequence/command records, cache identities, frozen parameter summary.
- Python parameter codec, command construction, acceptance, material-frame scoring and timeout handling.
- C++ sequence parser, strip generator, placement-aware hit collection, ordered accumulation and metric update.
- Preserved-target sidecars/report and supplied binary identity evidence.

**No edits, solver launches, executable fixtures or builds were performed.** Available reviewer tools supported file inspection, not command execution.

## Launch-blocking findings

### B1 — Frozen solver settings are not carried into the command

**Scientific/operational blocker:** `calibration/bo/run_convention_corrected.py:12`, invoked at line 39.

The runner reads `calibration/configs/{run}.json` with `json.loads`. It neither resolves `solver_config` nor copies the archived effective arguments. `Evaluator` does not perform that merge.

Consequences:

- `-native_hybrid_export_state true` is absent. Current C++ defaults this to false at `src/simulations/Sim_Bilayer_Growth.cpp:1363`, while material-reference evaluation requires that state. A successful solve can therefore consume its allocation and then fail scoring.
- Archived explicit settings are missing, including `-tol 3e-12`, `-hessian_threads 8`, both gradient tolerances, warmup count and adaptive-continuation selection.
- Current source defaults to `native_hybrid`; this is **not necessarily a backend-fallback bug**. It is nevertheless an incomplete, different execution recipe.

Simply replacing `json.loads` with `load_config` is insufficient: `shellsim/configs/default_adaptive_cutback.json` introduces other controls and specifies a native timeout of **1800 s**, whereas all three archived requests specify **3600 s**.

**Smallest repair:** construct each proposal from its archived `request.json["config"]`:

- `run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/request.json`
- `run/run9_transfer_expanded/evaluations/attempt_0209/request.json`
- `run/run10_transfer/evaluations/attempt_9902/request.json`

Change only the authorized angle, material shift, target/orientation and output/report paths. Retain the archived effective solver arguments without merging today's shared defaults.

**Clearance condition:** preparation-only structural comparison shows no other consequential differences; generated commands retain the archived settings, native state export and 3600-second limits.

### B2 — Selected executable differs from the archived executable

**Scientific blocker:** `calibration/bo/run_convention_corrected.py:12`, through each current config's `solver.executable="bin/shell"`.

Supplied read-only evidence, inspected at `/tmp/convention_binary_identity.txt`:

| Executable | SHA-256 |
|---|---|
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |
| `run/native_shape_refinement/build_9060950/bin/shell` | `c639c9ca7502a1698bdca297a3e3150cd042ab96f7c264d8c07c5e968d75b91a` |

The latter matches all three archived cache identities and `run/native_shape_refinement/build_9060950/binary.sha256`. No evidence establishes equivalence of the current executable. Different hashes alone do not prove incompatibility, but they prevent treating this as an unchanged frozen solver.

**Smallest repair:** use the existing archived executable and verify its recorded hash before launch. Do not rebuild or substitute the current binary merely to run this diagnostic.

**Clearance condition:** revised proposals select that executable; source/build evidence links the reviewed signed-angle implementation to the selected build. `source.sha256` exists alongside the archived binary. My current-source inspection is not independently verified archived-machine-code evidence.

## Checks and results

### Geometry, ordering and signed angles

`src/libshell/ZigZagGrowth.hpp:200–293` constructs
\[
p=L_v\tan\alpha,\qquad W=(N-2)p,\qquad u_0=-W/2.
\]
Negating alpha negates every generated horizontal coordinate, preserving point and strip order. Thus the first segment starts at the rightmost centreline coordinate and travels in increasing \(v\).

With \(s_v=L_v/2-65\), its initial \(v=-65\), hence public \(z=v+65=0\).

| Case | Alpha | Centre \(s_v\), mm | Initial public x, approximately mm | Public centreline z range, mm |
|---|---:|---:|---:|---:|
| Run6 | −1° | 0 | +74.882 | [0,130] |
| Run9 | −1° | 28.75 | +49.092 | [0,187.5] |
| Run10 | −1.5° | 22.5 | +50.408 | [0,175] |

These are analytic calculations, not executed fixtures.

- Placement uses actual endpoint minima/maxima, not an assumption that the variable named `x_L` is numerically leftmost.
- `src/libshell/ZigZagGrowth.hpp:744–846` applies material-frame shifts directly and preserves strip-major hit order.
- `src/libshell/ZigZagSequenceGrowth.hpp:542–567` does not reject negative alpha.
- The parser still requires the literal `left_bottom_up`; do **not** invent a `right_bottom_up` enum value. Signed alpha supplies the desired geometry.
- Segment angles come from `atan2` and normalization modulo π. This is appropriate for the orthotropic principal-axis tensor.
- `src/simulations/Sim_Bilayer_Growth.cpp:2412–2659` passes signed alpha through and accumulates successive hits without sorting by horizontal position.

**Nonblocking clarification:** 85 mm is the clearance to the nominal centreline endpoint. The 2 mm-wide segment footprint extends approximately 1 mm farther, giving **84 mm footprint clearance**. This follows the explicitly authorized start coordinate; do not silently alter placement.

### Accumulation and numerical validity

For the reviewed path, hardening is
\[
q(n)=e^{-\beta n}=0.8^n.
\]
The two layer increments use \(qg(1\pm o)\). Frozen growth values are positive and orthotropy lies within \([-1,1]\), so incremental principal stretches remain positive.

`src/libshell/GrowthHelper.hpp:740–826` uses
\[
G=R_\theta\operatorname{diag}(1+g_1,1+g_2)R_\theta^\top,\quad
T=D_m^{-1}GD_m,\quad A_{\rm new}=T^\top A_{\rm old}T.
\]
This preserves positive definiteness for nonsingular \(T\); the implementation additionally checks triangle conditioning and updated eigenvalues. Horizontal reflection transforms the growth tensor consistently. No positive-alpha assumption was found in the inspected active accumulation path.

This conclusion is restricted to the frozen uniform-profile, disabled-clamp, zero-slope-feedback recipe. It is not certification of every optional model feature or of nonlinear convergence.

### Frozen parameter codec

All three archived requests and the current Run6 summary agree:

```text
gtop          0.00028835116227798787
gbot          1.4587994001079146e-05
ortho_top    -0.3989157893255133
ortho_bottom  0.960306323937336
```

`calibration/ewcal/common.py` uses logarithmic growth coordinates and linear orthotropy coordinates. These values are within the identical bounds. Encoding with the destination config and decoding through `Evaluator` is mathematically correct; an executed floating-point roundtrip was not performed.

Prefer the fixed archived request as parameter authority, or assert that `bayes_summary.json["best"]` still identifies Attempt209 and matches it. This avoids a mutable “best” becoming an accidental parameter change.

### Targets and material-frame score

`calibration/analysis/preserve_samples_putong_flip.py:14–40` applies \(R=\operatorname{diag}(-1,-1,1)\) to retained points, basis columns and local shift. The recorded formula remains consistent:
\[
((s-o)B-t)R=(s-o)(BR)-tR.
\]

The script does not rebin, recrop or reselect samples. Existing comparison coordinates are rotated; scalar distances are preserved. Its report records:

| Case | Retained samples | New target SHA-256, from sidecar |
|---|---:|---|
| Run6 | 14504 | `00d64e18d28eab6c7254d674af8424d169cac3672abaa2bcf072cab59673a348` |
| Run9 | 14468 | `aa1fc209fa1b1eee7929a050957c3551ec959d09ed0fec92b02769a01451890e` |
| Run10 | 14504 | `97a0e13ba494487578107c39fc0eff8e44706eb8f0ba783ab0b78583adcbd578` |

The report's original-target hashes match the corresponding archived cache identities. Sidecar basis/shift sign changes were checked manually. NPZ contents, fresh hashes and roundtrip assertions were **not independently executed** here.

`calibration/ewcal/material_frame.py` aligns simulated vertices to their own material reference, not to the scan. It centres the scan without fitting its rotation. The symmetric score is
\[
\sqrt{\tfrac12\left(\operatorname{mean}d_{\rm scan\to mesh}^2+
\sum_i w_i d_{\rm vertex\to scan}^2\right)},
\]
with reference-area weights.

Both clouds must remain in internal `(u,v,h)` coordinates for this scorer. Public `(u,h,v+65)` is an output/display convention; applying it to only one cloud would invalidate scoring.

### Isolation and timeout handling

- Separate output roots and `use_cache=False` are correct.
- The existing `evaluations` guard plus exclusive attempt-directory creation prevents ordinary reruns under the stated single-writer arrangement.
- `launch()` enforces elapsed timeout and terminates the process group on timeout or interruption.
- `--prepare-only` may replace proposal JSON; use it before launch, not to rewrite an executed proposal.
- Existing preserved targets already exist: do not rerun their preparation script.
- No restart capability is claimed or required: this protocol stops failed evaluations rather than retrying from scratch.

## Practical launch plan after repair and reassessment

1. Prepare all three proposals from archived requests; pin the archived executable and frozen parameters.
2. Inspect generated sequences and commands without solving. Confirm the permitted differences, target identity checks, parameter roundtrip, unchanged scoring implementation and build/source linkage.
3. Preserve this review's hashes in durable launch evidence; the supplied `/tmp` file is not durable provenance.
4. After blockers clear, submit one CPU Slurm job per case: **8 CPUs, 16 GB, 2 hours, no GPU**, with the existing Python/VTK environment and thread limits matching the allocation. Invoke the repaired runner once for each case; do not solve on the login node.
5. Retain both the controller and archived native **3600 s** limits. Numerical failures stop that case; no tolerance relaxation or fresh retries.
6. Report accepted numerical evidence, both directional RMS components, symmetric RMS and spatial residuals. Preserve failures as failures, not large physical discrepancies.

## Scientific boundaries and residual risks

These are three frozen, model-conditional forward predictions under a researcher-selected convention—not new independent held-out validation or physical parameter identification.

The target's 180° rotation and the path's horizontal reflection are **different transformations**. This is not proof that the corrected simulation is simply the old prediction relabelled.

Improved agreement would support compatibility with the selected convention, not establish worked face, physical edge/start identity, or a clamp mechanism. Worse agreement would not justify selecting another flip or refitting. Solver failure supplies no geometric conclusion. Scan provenance, top-surface-as-midsurface approximation, discretization, and final-only equilibrium remain interpretation limits.

Approval is withheld for the reviewed runner. Any later approval applies only to the repaired slice and does not itself authorize execution, additional evaluations or altered stopping rules.