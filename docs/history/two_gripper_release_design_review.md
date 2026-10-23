# Two-gripper paired-release design review

## Verdict: `CHANGES_REQUESTED` — design only

The approved comparison is scientifically coherent and implementable without a workflow-engine refactor. The blockers below concern complete restart state, paired-release isolation and evidence integrity. This is **not production-launch approval**.

### Scope and authority

Read the complete approved objective:

- `.pi/goals/active_goal_2026101623411509_mvbwsnko-2nl477.md`
- Active protocol in `docs/reports/boundary_conditions_and_springback_synthesis.md`

Reviewed source against supplied HEAD identity `e827565a2a4f56c05398789679621abaa24fb28e`, including the requested primary files, underlying serializers, gradient merging and endpoint checks. Read both supplied Attempt 209 sequences and commands. Historical report sections 1–6 were not used as scientific evidence or authorization.

The comparison remains: 29/33 clamped repeats with equilibrium after each repeat; two releases from identical complete final clamped state; unchanged targets/history; researcher-specified geometry, not historical reproduction.

No execution, builds, tests, edits or literature search were performed. Source remained unchanged in observed working-tree deltas. The report changed during review and two new configuration files appeared; those new configurations are **outside this review**. Supplied commit and recorded SHA-256 identities were not independently recomputed.

## Ranked findings and clearance conditions

### 1. Blocking: unmodified `Mesh` serialization is unsafe for complete restart

**Evidence**

- `src/libshell/Mesh.hpp:178–212` reads/writes rest and current configurations using the **same basename**.
- `src/libshell/DCSConfigurations.hpp:371–378,581–594` delegates geometry serialization to `DCSConfigurationData`.
- `src/libshell/ConfigurationData.hpp:302–332` writes both configurations to identical `_vertices.dat` and `_edgedirs.dat` names.

Consequently, current geometry/directors overwrite rest geometry/directors. Loading reads that current geometry into both configurations.

**Important distinction:** bilayer target growth is serialized: `_aform_bot.dat`, `_aform_top.dat` and `_bform.dat` are explicitly written/read. The defect is not omission of those forms. However, sequence hit counts, cursor, material-coordinate provenance and acceptance identity are absent. Underlying readers also lack checked stream completion and robust dimension validation (`src/libshell/common.hpp:278–288,336–354`).

**Clearance:** use a sequence-local validated checkpoint, or explicitly separated rest/current serialization with a validated sequence manifest. Do not describe existing `-cycle_write_state` output as complete restart evidence.

The diagnostic export also needs a precise correction: `selected_dofs` contains `report.selected`, which is the complete optimizer vector, including director components (`src/simulations/Sim_Bilayer_Growth.cpp:1344`; `src/libshell/NativeHybridShellSolver.hpp:344–345`). It lacks a restart contract, history and masks—not necessarily all director information.

### 2. Blocking: paired release must restore the same accepted state, not chain releases

Current release code at `src/simulations/Sim_Bilayer_Growth.cpp:2937–2950` restores `initialBC`, which initialization requires to contain no vertex constraints (`:1088–1094`). It therefore implements the free branch only.

**Smallest practical seam**

Add a narrowly scoped checkpoint helper, for example under `src/libshell/ZigZagSequenceCheckpoint.hpp`, with these proposed interfaces:

- `SequenceCheckpoint captureAcceptedCycle(const BilayerMesh&, const SequenceProgress&, const SequenceIdentity&);`
- `void restoreAcceptedCycle(BilayerMesh&, SequenceProgress&, const SequenceCheckpoint&, const SequenceIdentity&);`
- `void writeAcceptedCycle(const std::string&, const SequenceCheckpoint&);`
- `SequenceCheckpoint readAcceptedCycle(const std::string&, const SequenceIdentity&);`

Use owning vectors/matrices, not `Eigen::Map`/`Ref` aliases or whole-mesh assignment.

**Checkpoint contents**

| Preserve or validate exactly | Recompute safely after validation |
|---|---|
| Full current `[x\|y\|z\|directors]` vector | Current triangle/normal caches |
| Rest vertices/directors; top/bottom target metrics; reference curvature | Topology adjacency from fixed faces, checking ordered edge correspondence |
| Ordered faces/edges and material coordinates | Toolpath hit lists from identical configuration/material mapping |
| Vertex/edge masks and any clamped-normal data | Constant material operators |
| `totalPassCount`, accepted global cycle and toolpath/repeat cursor | Fresh optimizer state for the next objective |
| Effective material/solver/configuration identities and accepted endpoint proof | Cycle-zero reference forms from identical initialization, or save explicitly |
| Progress/evidence bookkeeping needed to avoid replay and overwrite | Transient within-repeat diagnostics |

For this task, reject nonempty director/normal clamps rather than silently treating vertex-mask removal as complete physical release.

Save only **after the full repeat reaches accepted `lambda=1` and commits `pathEndPassCount`** (`src/simulations/Sim_Bilayer_Growth.cpp:2760–2771`). A failed/interrupted repeat resumes from its preceding accepted cycle, without replaying completed growth. Restore the cursor before entering growth or cold-start logic.

Before **each** release:

1. Restore the immutable final clamped checkpoint.
2. Verify complete state/target/history identity.
3. Replace the vertex mask with either `buildMinimalReleaseGaugeMask(materialCoordinates).vertex_mask` or an all-false mask.
4. Construct a fresh native solve, capturing fixed values at the restored release coordinates.
5. Perform no growth update.

Checkpoint publication should be checked, atomic and no-clobber; reject corruption, dimensions/order mismatch, wrong recipe/build identity, invalid cursor and unaccepted checkpoints before mutating live state. Resume into distinct evidence paths rather than truncating existing CSVs.

`ReferenceHybrid/Core.hpp:235–286` provides useful publication/validation patterns, but its checkpoint is a **fixed-objective optimizer** checkpoint. It does not save material history or the entire adaptive `HybridSession` controller. Accepted-cycle restart avoids extending that machinery.

### 3. Blocking: endpoint outputs need distinct identities and unchanged strict acceptance

Minimal edits in `src/simulations/Sim_Bilayer_Growth.cpp`:

- **`:2949`:** remove the release-specific `max(...,1e-11)` and pass the same acceptance tolerance used for forming. Currently it does **not** relax the supplied `5e-14` native target, because `:1293–1296` takes the minimum with the requested native tolerance. It can nevertheless discard a stricter configured acceptance tolerance.
- **`:1305–1355`:** pass an explicit endpoint identity/output stem to the native wrapper. Keep separate clamped, Putong-release and free-release diagnostics and proofs.
- **`:1331–1334`:** the current diagnostic exporter refuses an existing filename; a second terminal export would **throw**, not silently overwrite. Generic release VTP names would also collide if the branch were simply duplicated.
- **`:2974–3039`:** use distinct branch filenames and records, with parent-checkpoint identity, mask, gauge vertices/fixed values, target/history hashes, acceptance and remaining-rigid-mode count.
- **`:2981`:** replace unconditional gauge metadata with unambiguous fields: six persistent Putong coordinate constraints versus remaining numerical rigid-mode treatment. A boolean alone conflates different mechanisms.
- **`:3191–3211`:** BC-enabled runs currently emit no `sequence_result.json`. Add a small paired-result manifest; do not let a single terminal variable stand for both releases.

Preserve failed proof records and mark incomplete comparisons explicitly. Keep `seed_escape` and other unapproved post-solve modifications disabled.

**Clearance:** fixture evidence must show distinct outputs, identical branch inputs, unchanged target/history fields, no acceptance relaxation and no successful-pair manifest after a failed branch.

### 4. Blocking before launch: resolve command provenance and timeout semantics

The saved Run 9 command at:

`run/run9_transfer_expanded/evaluations/attempt_0209/case_00/command.json:39`

points to **`attempt_9901/case_00/sequence.json`**, not the supplied Attempt 209 sequence. The supplied Attempt 209 sequence itself contains the expected frozen parameters.

Do not replay that argv blindly. Generate a new command referencing the reviewed new configuration and record this provenance discrepancy. Historical reproduction is unnecessary.

Both saved commands otherwise specify identical geometry/material thickness and native controls, including `h_total=0.0007700000000000001`, `5e-14` targets and eight Hessian threads. Current source fixes `E=1`, `nu=0.33` (`src/simulations/Sim_Bilayer_Growth.cpp:203–212`); report energies/reactions in that normalization, not automatically as physical joules/newtons.

The proposed resource ceilings are reasonable **limits, not demonstrated runtime estimates**. In particular:

- Native optimization time checks are cooperative.
- Endpoint verification may start within `options.seconds+30`; acceptance is rejected after `+50` (`src/libshell/NativeHybridShellSolver.hpp:349–367`).
- An individual expensive evaluation is not preempted by those checks.

Therefore, `-native_hybrid_seconds 600` is not a hard 600-second complete-solve limit. Record the actual optimization/verification convention and job-level hard stop before launch.

Existing continuation defaults are initial/max step `1`, minimum `1/64`, growth `2`, retries `8` (`src/simulations/Sim_Bilayer_Growth.cpp:1359–1369`). Retries reset after accepted continuation substeps. Clarify that “stop on unaccepted endpoints” permits only these recorded within-repeat remedies; an unaccepted complete repeat/release cannot advance the trajectory. Repeated process restarts must not become unlimited numerical retries.

## Six-coordinate gauge: compatible locally, not guaranteed globally

`ShellProblem` captures current fixed values and rejects changes exactly (`src/libshell/ReferenceHybrid/ShellProblem.hpp:60–90`). Its `freeDofs` adds only the rigid modes surviving the active mask (`:135–160`). Thus, a full-rank Putong mask receives **no redundant extra rigid gauge**.

However, material-space noncollinearity in `src/libshell/ZigZagSequenceBoundaryConditions.hpp:164–233` does not establish current-state gauge conditioning. Its “remove only rigid translation/rotation” comment should be qualified.

Let \(b=x_B-x_A\), \(c=x_C-x_A\). After fixing A’s translation, the infinitesimal rotation constraints have matrix

\[
\begin{pmatrix}
-b_z&0&b_x\\
b_y&-b_x&0\\
c_y&-c_x&0
\end{pmatrix},
\qquad
\det=b_x(b_xc_y-b_yc_x).
\]

Therefore current **projected xy geometry**, not just material triangle area, controls local rank. Inspect scaled singular values and condition number at release start and endpoint. Compare with the projector’s normalized-Gram threshold `1e-12` and QR threshold `1e-10`; near degeneracy can cause threshold-sensitive classification.

Full rank gives a local coordinate slice through rigid-motion orbits. It neither forces all z values to zero nor guarantees finite/global reachability. For example, preserving B’s fixed y/z offsets from A requires

\[
|x_B-x_A|\geq\sqrt{(\Delta y_{\rm fixed})^2+(\Delta z_{\rm fixed})^2}.
\]

An intrinsically changed configuration can violate that requirement even though the original release state had full rank.

**Reactions are especially informative:** at exact constrained stationarity, rotational/translational invariance and full rank of the constrained rigid-motion matrix imply zero six-coordinate reactions. Significant reactions therefore warrant investigation of residuals, conditioning, symmetry or rank loss—not an automatic claim of physical constraint distortion.

The current analytical gradient zeros constrained entries (`src/libshell/MergePerFaceQuantities.hpp:57–85,129`). For reactions, temporarily remove the vertex mask under an exception-safe restore guard, update current caches, evaluate `engOps.compute` at unchanged DOFs, then restore the mask/cache. Assert no edge/normal clamps. Check energy invariance and validate selected derivatives by finite differences on a small fixture. Do not read reactions from the masked solver gradient.

## Smallest fixtures required before production review

Extend `test/testshell/Test_NativeHybridProduction.cpp` or add one focused sequence test file:

1. **Geometry/masks:** confirmed pad bounds, both nonempty, xyz only; Putong has exactly six selected coordinates; free branch has none.
2. **Nontrivial round-trip:** curved current geometry, nonzero directors, unequal top/bottom target metrics, nonzero history; exact persisted-field equality and unchanged rest geometry.
3. **Actual accepted-cycle resumption:** interrupt a tiny two-repeat case after repeat one; resume repeat two once; compare targets, counts, masks, final state and evidence against uninterrupted execution.
4. **Branch isolation:** both branches start from the same full vector and immutable fields; first branch cannot contaminate the second; distinct outputs; failure remains failure.
5. **Gauge/reactions:** curved nonzero fixed z values, full-rank and near-singular cases, expected surviving rigid counts, exact fixed-coordinate preservation and independently checked unmasked derivatives.
6. **Invalid restart:** truncated/corrupt payload, wrong topology/order/configuration, invalid cursor and unaccepted checkpoint all fail closed.

Existing tests cover native controller budgets, fixed-objective behavior and unconstrained curved-shell adapter consistency (`test/testshell/Test_NativeHybridProduction.cpp:277–345`), but do not establish this sequence-resume or paired-release contract.

Keep each fixture invocation within the recorded 120-second limit on compute nodes. No production convergence result is implied by these proposed checks.

## Scientific boundaries and residual risks

The target is a **model-conditional release comparison**, not physical identification of Attempt 209 parameters or validation of historical claims. Agreement supports equivalence only for these tested states. Disagreement can reflect finite gauge incompatibility, incomplete convergence or different stable equilibria.

Use identical material-vertex correspondence for rigid alignment; report full-field distributions, energies, fixed residuals, rank/conditioning and unconstrained-gradient diagnostics. Positive reduced-Hessian pivots provide local numerical evidence, not global uniqueness. Scan scoring remains conditional on separately verified registration/provenance; no scan data or scoring pipeline was reviewed.