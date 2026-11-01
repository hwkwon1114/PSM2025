# Convention-Corrected Forward Predictions — Current-Solver Pre-Execution Review (Argon)

**Verdict: `APPROVED_WITH_CAVEATS`**  
**Launch-blocking findings:** None for the reviewed three-case (`run6`, `run9`, `run10`) frozen-parameter forward evaluation slice using the current adaptive-cutback solver `bin/shell`.

Approval covers **only** the three authorized single-shot forward evaluations (`run6`, `run9`, `run10`) at the frozen Attempt209 calibration parameters under `docs/calibration/bayesian_calibration/convention_corrected_investigation.md`. It does not authorize retries, parameter refitting, tolerance relaxation, additional candidate evaluations, login-node solver execution, or access to sealed test sets.

---

## 1. Scope, Authority, and Relationship to Prior Reviews

- **Authoritative protocol:** `/gpfs/projects/p33082/PSM2025-zigzag/docs/calibration/bayesian_calibration/convention_corrected_investigation.md` (including the researcher-directed solver upgrade section superseding the archived-executable plan).
- **Reviewed runner:** `calibration/bo/run_convention_corrected.py` (SHA-256 `8821a2c6d9e07b07d25e334d7edd0d3e22651f47eab786401f22508506fd6490`).
- **Supersession of prior review artifacts:**
  - `docs/calibration/bayesian_calibration/convention_corrected_preexecution_review.md` (`CHANGES_REQUESTED`) reviewed an earlier 43-line raw-config version of the runner.
  - `/home/pxl1051/.pi/agent/sessions/--gpfs-projects-p33082-PSM2025-zigzag--/subagent-artifacts/f3d4335a-ae3d-406f-ac57-17bcaeb1e90c_astra-code-reviewer_output.md` and `docs/calibration/bayesian_calibration/convention_config_check.json` evaluated the **archived** executable (`run/native_shape_refinement/build_9060950/bin/shell`, SHA-256 `c639c9ca7502a1698bdca297a3e3150cd042ab96f7c264d8c07c5e968d75b91a`) and are **stale** for the current-solver upgrade.
  - Prior square-dome campaign reviews (`docs/dome_bo/square_bo_pre_execution_review.md`, `docs/dome_bo/square_verification_pre_execution_review.md`, `docs/dome_bo/square_verification_review_evidence.txt`) pinned the same binary SHA-256 (`522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` at `run/dome_square/frozen/shell`), but approval of an unrelated campaign is not treated here as proof of current source linkage; this review independently audited the `build_cmake/` build graph, timestamps, object hashes, C++ sources, and refreshed command/config evidence.
- **Read-only compliance:** No files were modified, no builds were executed, and no solver runs were launched during this review.

---

## 2. Resolution of the Two Pre-Review Evidence Gaps

Before quota failure, Astra noted that the current-solver slice was statically coherent pending two evidence items: (1) current binary (`bin/shell`) build/source linkage, and (2) refreshed preparation and effective command verification. Both have now been inspected and verified.

### 2.1 Current Executable (`bin/shell`) Build-Tree and Source Linkage

Inspected `docs/calibration/bayesian_calibration/convention_current_build_evidence.txt`, `build_cmake/build.ninja:235–275`, `build_cmake/.ninja_log`, and `build_cmake/compile_commands.json`:

| Artifact / Source | SHA-256 | Size (B) | Modification Time (CDT, `-0500`) |
|---|---|---:|---|
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` | 2,195,728 | `2026-10-21 09:19:12.069747603` |
| `run/dome_square/frozen/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` | 2,195,728 | `2026-10-21 20:30:06.282639063` |
| `build_cmake/CMakeFiles/shell.dir/src/simulations/Sim_Bilayer_Growth.cpp.o` | `f6c799dcedcf38dbca2d7f098f041594281321ea471ed2df54a36bed59221d31` | 2,999,640 | `2026-10-21 09:19:10.204786265` |
| `build_cmake/CMakeFiles/shell.dir/src/simulations/main.cpp.o` | `1362b10b577e453f9df477f2d18ec6d1d3ab99c1f93b6f5ea4763c3bb9b6c27a` | 257,008 | `2026-10-16 20:51:38.024101359` |
| `lib/liblibshell.a` | `77c9d445c9145daea841190bc082b9ec1773734006d906353163478346abbdd4` | 2,617,862 | `2026-10-16 20:51:34.412105659` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `eb842b6451136caaa4f4749eb02dd50c10626fb6e05e0c4978dbb61cb2fca475` | 233,469 | `2026-10-21 09:15:51.109559286` |
| `src/libshell/NativeHybridShellSolver.hpp` | `319ebdbb93b29abcc92ee765a0a504edf488473b5caa2d3685709a4ddb261ac7` | 23,235 | `2026-10-21 08:56:02.379146090` |
| `src/libshell/ZigZagSequenceGrowth.hpp` | `75887bd1885b8d41f843451da778df414c93bbcade75f3afea715ec6093bb6ec` | 25,539 | `2026-10-15 21:03:14.928178053` |
| `src/libshell/ZigZagGrowth.hpp` | `30177a331cddf8500560a66c8ce72cdeb6c1f54819d402ee6a63ed74b2f6da35` | 38,669 | `2026-10-15 21:02:44.870205939` |
| `src/simulations/main.cpp` | `3c0ee5ef0fef71f7e89d087953923ebfb3cbaf3ab9be6bc01de204d2c818b1ca` | 1,951 | `2026-09-29 20:23:17.836842846` |
| `src/libshell/GrowthHelper.hpp` | `323dd80a7fd9cdb1bf9f6723209c1fcc552a4e7b08b8bc999b729ba53166dd04` | 44,488 | `2026-09-29 20:23:17.708559484` |

- **Linkage verification:** `build_cmake/build.ninja:235–275` compiles `Sim_Bilayer_Growth.cpp.o` and `main.cpp.o` with `-DNONCONVEX_WITH_CHOLMOD=1 -DSHELL_WITH_NATIVE_HYBRID=1` and links `/home/pxl1051/project_shortcut/PSM2025-zigzag/bin/shell` against `lib/liblibshell.a`, `lib/libhlbfgs.a`, SuiteSparse CHOLMOD, `libblis.so`, `libgomp.so`, and VTK 9.3. The final two lines of `build_cmake/.ninja_log` record compiling `Sim_Bilayer_Growth.cpp.o` (`1792592178631651000` ns) and linking `bin/shell` (`1792592350333170858` ns), matching the exact timestamps of `Sim_Bilayer_Growth.cpp.o` (`09:19:10.204 -0500`) and `bin/shell` (`09:19:12.069 -0500`), both strictly newer than all inspected C++ sources and headers.
- **Evidence boundary on Ninja dry-run:** In `docs/calibration/bayesian_calibration/convention_current_build_evidence.txt`, `/gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/ninja -C build_cmake -d explain -n shell` triggers the forced phony glob check (`VerifyGlobs.cmake_force is dirty` -> `[1/2] Re-running CMake...`) rather than printing `no work to do`. Accordingly, source-to-binary correspondence is supported by the `.ninja_log` link history, `build.ninja` dependency graph, `compile_commands.json`, and consistent source/object/binary timestamps and hashes—not by overclaiming that the dry-run alone proved bit-for-bit reproducible compilation.

### 2.2 Refreshed Preparation Log, Proposed Configs, and Effective Commands

- **Diagnosis of `docs/calibration/bayesian_calibration/convention_new_preparation.log`:** The earlier `srun-here` job (`5178556`) ran in the `DHD` conda environment where `run_convention_corrected.py <run> --prepare-only` succeeded for all three runs and wrote the updated `run/convention_corrected/{run6,run9,run10}/proposed_config.json` and `frozen_parameters.json` (passing both `BINARY_SHA` and `TARGET_SHA` assertions), before a trailing `ninja` command in the same shell snippet exited with code 127 (`bash: ninja: command not found`) because `ninja` is installed in `smcpp_vtk38`, not `DHD`.
- **Fresh command/config verification (`docs/calibration/bayesian_calibration/convention_current_commands.json`):** Generated by `calibration/analysis/check_convention_commands.py` (Slurm job `5179586`, exit 0). It confirms:
  1. `config(run)` in `calibration/bo/run_convention_corrected.py` is identical to the persisted `run/convention_corrected/{run}/proposed_config.json` for `run6`, `run9`, and `run10`.
  2. `Parameters(cfg).decode(Parameters(cfg).encode(params))` roundtrips all four Attempt209 parameters within `rtol = 1e-14`.
  3. The structural differences between each archived `request.json['config']` and the current `proposed_config.json` are restricted to the authorized convention changes, output directories, and the researcher-directed adaptive-cutback solver upgrade.

| Artifact | SHA-256 |
|---|---|
| `calibration/bo/run_convention_corrected.py` | `8821a2c6d9e07b07d25e334d7edd0d3e22651f47eab786401f22508506fd6490` |
| `shellsim/configs/default_adaptive_cutback.json` | `94252b22fd97d165de5fa0c1b203dbf0ef5c9d77639ef5474f9fa925fc187ce3` |
| `run/convention_corrected/run6/proposed_config.json` | `519e5053eaee2d3e7218e96c208fbe1f899d697cbaf711310f8577a2e85fc591` |
| `run/convention_corrected/run9/proposed_config.json` | `2041610df3629d668aaea75a4e2405fc1bef8a695d71060090ead57f15c4ce57` |
| `run/convention_corrected/run10/proposed_config.json` | `ac2bd195ba70233246e3f1452604cd2d5868c095e1421540a8b57e5fd9869cfc` |
| `run/convention_corrected/{run6,run9,run10}/frozen_parameters.json` | `d34a6bdd0ee759b0c051884b5795ad7dd63709128c13e5f276fceb2fb92bae94` |
| `run/putong_convention_check/preserved_samples/run6.npz` | `00d64e18d28eab6c7254d674af8424d169cac3672abaa2bcf072cab59673a348` |
| `run/putong_convention_check/preserved_samples/run9.npz` | `aa1fc209fa1b1eee7929a050957c3551ec959d09ed0fec92b02769a01451890e` |
| `run/putong_convention_check/preserved_samples/run10.npz` | `97a0e13ba494487578107c39fc0eff8e44706eb8f0ba783ab0b78583adcbd578` |

---

## 3. Physical, Constitutive, and Algorithmic Verification

### 3.1 Coordinate Mapping, Placement Shifts, and Bottom Clearance

- **Public vs. internal coordinates:** The simulator operates in material coordinates \((u, v) \in [-100, 100] \times [-150, 150]\text{ mm}\) with out-of-plane deflection \(h\). Under the researcher-specified mapping \((x, y, z) = (u, h, v + 65)\text{ mm}\), the \(200 \times 300\text{ mm}\) sheet spans \(x \in [-100, 100]\text{ mm}\) and \(z \in [-85, 215]\text{ mm}\), with \([-85, 0]\text{ mm}\) as the 85 mm bottom clearance zone.
- **Centerline shifts:** Setting `shift_uv_mm = [0.0, lv_mm / 2 - 65.0]` places the bottom of every zigzag pattern at \(v_{\text{start}} = s_v - L_v / 2 = -65\text{ mm}\), i.e., public \(z_{\text{start}} = 0\text{ mm}\):

| Case | \(L_v\) (mm) | \(N_{\text{strips}}\) | Repeats | \(\alpha\) (deg) | `shift_uv_mm` (mm) | Initial \((x_0, z_0)\) (mm) | Public centerline \(z\) range (mm) | Centerline / 2 mm-footprint clearance (mm) |
|---|---:|---:|---:|---:|---|---:|---|---|
| `run6` | 130.0 | 68 | 29 | \(-1.0\) | `[0.0, 0.0]` | \((+74.882, 0.0)\) | \([0.0, 130.0]\) | \(85.0\) bottom, \(85.0\) top / \(84.0\) footprint |
| `run9` | 187.5 | 32 | 33 | \(-1.0\) | `[0.0, 28.75]` | \((+49.092, 0.0)\) | \([0.0, 187.5]\) | \(85.0\) bottom, \(27.5\) top / \(84.0\) footprint |
| `run10` | 175.0 | 24 | 50 | \(-1.5\) | `[0.0, 22.5]` | \((+50.408, 0.0)\) | \([0.0, 175.0]\) | \(85.0\) bottom, \(40.0\) top / \(84.0\) footprint |

- **Bounds check:** `checkPlacedFootprintInsideMaterialBounds` (`src/libshell/ZigZagGrowth.hpp:658–716`) inflates the actual endpoint bounding box by `half_w = 1.0 mm` and checks against \([-100, 100] \times [-150, 150]\text{ mm}\). All three placed footprints lie well inside the material domain.

### 3.2 Negative Signed Angle, Traversal Order, and Metric Accumulation

- **Reflected strip generation (`src/libshell/ZigZagGrowth.hpp:200–293`):**
  - `buildZigZagStrips` computes \(p = L_v \tan\alpha\), \(W = (N_{\text{total}} - 2)p\), \(x_L = -0.5 W\), \(x_R = +0.5 W\), and builds `pts[0] = (x_L, -0.5 Lv)`, `pts[1] = (x_L, +0.5 Lv)`, `pts[1+j] = (x_L + j p, \pm 0.5 Lv)`.
  - For \(\alpha < 0\), \(p < 0\) and \(x_L = +0.5 |W| > 0\). Therefore strip \(k = 0\) starts at the **rightmost** centerline coordinate \((+0.5|W|, -0.5 L_v)\) (public \(z = 0\)) and travels upward in \(+v\) (public \(+z\)) to \((+0.5|W|, +0.5 L_v)\), and subsequent inclined strips step in \(-u\) (right-to-left) down to \(x_R = -0.5|W|\).
  - `reverse_traversal` is `false` and `start_mode` remains `"left_bottom_up"` (which `src/libshell/ZigZagSequenceGrowth.hpp:395–402` requires as the literal schema token).
- **Hit collection and ordered Voce hardening (`src/libshell/ZigZagGrowth.hpp:744–846`, `src/simulations/Sim_Bilayer_Growth.cpp:2412–2659`):**
  - `collectMaterialCoordinateHits` iterates `k = 0 .. strips.size() - 1` and collects face hits in strip-major order without sorting by coordinate sign.
  - `Sim_Bilayer_Growth.cpp:2585–2659` loops `repeat_index = 1 .. toolpath.repeat` and `for (const auto& hit : materialHits)`, evaluating Voce decay \(q(n) = e^{-\beta n} = 0.8^n\) (\(\beta = 0.2231435513142097\)) from `totalPassCount(face)` and incrementing `totalPassCount(face)` after each hit.
  - Because `-sequence_slope_feedback` is `"0"`, `slopeFactor = 1.0` identically (`Sim_Bilayer_Growth.cpp:2601–2624`).
- **Orthotropic growth tensor under horizontal reflection (`src/libshell/GrowthHelper.hpp:740–826`):**
  - Each strip angle is \(\theta = \operatorname{normalize\_angle\_pi}(\operatorname{atan2}(d_y, d_x))\). Negating \(\alpha\) maps \((d_x, d_y) \mapsto (-d_x, d_y)\), so \(\theta \mapsto \pi - \theta \pmod\pi\).
  - `updateAbarWithMaterialGrowthIncrement` forms \(G = R(\theta)\operatorname{diag}(1 + g_1, 1 + g_2)R(\theta)^\top\) and updates \(\bar{a}_{\text{new}} = T^\top \bar{a}_{\text{old}} T\) with \(T = D_m^{-1} G D_m\). Under \(u \mapsto -u\), \(G\) conjugates by \(\operatorname{diag}(-1, 1)\), which is physically exact for a horizontally mirrored toolpath. With the frozen Attempt209 parameters (`gtop = 2.8835116227798787e-4`, `gbot = 1.4587994001079146e-5`, `ortho_top = -0.3989157893255133`, `ortho_bottom = 0.960306323937336`), \(1 + g_{1,2} > 1 > 0\) on both layers and every updated target metric remains symmetric positive definite.

### 3.3 Unchanged Scientific Recipe vs. Intentional Solver Upgrade

- **Unchanged physical model:** Sheet dimensions (`200 x 300 x 0.77` mm; `-lx 0.1 -ly 0.15 -res 0.01 -h_total 0.0007700000000000001`), free boundary conditions (`boundary_conditions.enabled = false`), uniform strip profile, Voce hardening (`beta = 0.2231435513142097`), and `solve_mode = "final_only"` are identical to the archived Attempt209 requests.
- **Intentional solver upgrade (`bin/shell` + `shellsim/configs/default_adaptive_cutback.json`):**
  - `calibration/bo/run_convention_corrected.py:19–26` merges `default_adaptive_cutback.json["options"]` into `cfg["solver"]["args"]` and explicitly restores `-native_hybrid_seconds = 3600` (overriding the shared file's `1800`) and `-sequence_minimize_every = 1`.
  - In `src/simulations/Sim_Bilayer_Growth.cpp:2536–2548,2705–2708`, `solveThisCycle` is `sequence_solve_mode == "every_cycle" || executed_cycle == requested_cycles`. Because `sequence_solve_mode` is `"final_only"`, `if (!solveThisCycle) continue;` skips equilibrium solves on all intermediate cycles `1 .. repeat - 1` regardless of `-sequence_minimize_every = 1`, and performs the equilibrium solve only on the final cycle (`29`, `33`, `50`).
  - On that final cycle (`Sim_Bilayer_Growth.cpp:2543–2548,2757–2868`), `pathStartAformsTop/Bot` are set to the flat initial forms (`initialAformsTop/Bot`) and `pathEndAformsTop/Bot` hold the exact accumulated final target metrics. The outer continuation attempts full load \(\lambda = 1.0\) first (`-sequence_initial_step 1.0`), and only if Newton stalls (`-native_hybrid_on_newton_stall cutback`, window `5`, ratio `0.85`, grace `10`, max stalls `2`) or encounters indefinite curvature (`-native_hybrid_max_newton_indefinite 3`) or reaches `max_hessians = 40` does it restore `acceptedState` and halve the load increment down to `-sequence_min_step 0.015625` (`1/64`) for up to `8` retries. At \(\lambda \ge 1.0\), the exact `pathEndAformsTop/Bot` (not floating-point interpolated forms) are qualified (`Sim_Bilayer_Growth.cpp:2772–2777,2866–2868`).
  - Independent endpoint verification in `src/libshell/NativeHybridShellSolver.hpp:367–409` reserves one Hessian under the 40-Hessian cap, re-evaluates energy, gradient, and Hessian at the selected state, and requires `status == "gradient_target"`, `gradientNorm <= 5e-14`, and `curvature == "positive_pivots"`. `calibration/ewcal/runner.py:139–227` (`check_output`) enforces these exact conditions plus `verification_evaluations >= 1`, `verification_hessians >= 1`, and `bilayer_zigzag_sequence_native_hybrid_state.json` export.

### 3.4 Preserved Scan Targets and Material-Frame Scoring

- **Preserved-sample rotation:** `run/putong_convention_check/preserved_samples/{run6,run9,run10}.npz` and their `.json` sidecars apply \(R = \operatorname{diag}(-1, -1, 1)\) to the retained point clouds (`14504`, `14468`, and `14504` points), negating the first two columns of `basis_columns` and the first two components of `shift_local_mm` while preserving edge alignment, sample selection, and preprocessing metadata (`spacing_mm=2`, `edge_inset_mm=2`, `z_outlier_floor_mm=20`, `max_cell_z_mad_mm=0.5`, `min_points=1000`).
- **Sidecar validation in `Evaluator.identity_inputs` (`calibration/ewcal/runner.py:255–291`):** Verifies `sidecar['target_sha256'] == digest(p)`, `sidecar['preprocess'] == config['preprocess']`, and `sidecar['orientation'] == case['scan_orientation']`. All three `proposed_config.json` files match their sidecars' `orientation` and `preprocess` blocks.
- **Material-frame scoring (`calibration/ewcal/material_frame.py:10–57`):** `score_material_frame` aligns the simulated deformed mesh `p` to its own exported material reference `st_ref_mm` (`reference_positions * 1000` in internal \((u, v, 0)\) mm) via a proper Kabsch rotation (\(\det R = +1\)) and reference-area-weighted centroid subtraction, and centers the target scan (`target - target.mean(0)`) without fitting scan rotation. It then computes directional and area-weighted symmetric RMS (`rms_symmetric_mm`) in internal `(u, v, h)` coordinates. Because both simulation and preserved targets remain in internal `(u, v, h)` coordinates during `Evaluator.evaluate()`, the scorer is coordinate-consistent; public `(x, y, z) = (u, h, v + 65)` is for reporting/display only and must not be applied asymmetrically to one cloud.

### 3.5 Execution Isolation, Timeouts, and Planned Slurm Hardware

- **Single-evaluation isolation:** `calibration/bo/run_convention_corrected.py:47–49` checks `if (out/'evaluations').exists(): raise FileExistsError(...)` and invokes `Evaluator(cfg).evaluate(..., use_cache=False, attempt_id=1)` once for the single case in `cfg['cases']`, writing `run/convention_corrected/{run}/forward_result.json` and `run/convention_corrected/{run}/evaluations/attempt_0001/`. Currently, `run/convention_corrected/{run6,run9,run10}/` contain only `frozen_parameters.json` and `proposed_config.json`; no `evaluations/` directory exists.
- **Planned Slurm allocation:** One independent CPU batch job per case (`run6`, `run9`, `run10`) with **8 CPUs, 16 GB RAM, 2-hour (`02:00:00`) wall-clock limit, no GPUs**, and **no login-node solve** matches `-hessian_threads 8`, the ~3000-vertex mesh size (`-lx 0.1 -ly 0.15 -res 0.01`), and the 3600 s controller timeout (`solver.timeout_seconds = 3600`).

---

## 4. Non-Blocking Caveats and Operational Conditions for Launch

1. **Thread environment variables in the Slurm launcher (important operational safeguard):**  
   `build_cmake/build.ninja:264` links `bin/shell` against both OpenMP (`libgomp.so`), TBB (`libtbb.so`), and BLIS (`/gpfs/home/pxl1051/miniforge/envs/DHD/lib/libblis.so`). While `solveNativeHybridShell` caps TBB parallelism to 1 (`NativeHybridShellSolver.hpp:355`) and passes `hessianThreads = 8` to OpenMP assembly, `calibration/ewcal/runner.py` does not set `BLIS_NUM_THREADS=1` inside `launch()`. Each 8-CPU Slurm job script should explicitly export:
   ```bash
   export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
   export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
   ```
   to prevent nested BLIS/OpenMP thread oversubscription during CHOLMOD factorizations.
2. **Standard Python invocation (no `-O` / `PYTHONOPTIMIZE`):**  
   `calibration/bo/run_convention_corrected.py:25–26,33,40,42` enforces binary SHA-256, target SHA-256, `solve_mode == 'final_only'`, and parameter agreement via `assert`. Invoke `/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python` without `-O` or `PYTHONOPTIMIZE`.
3. **Substep vs. whole-process timeout and failure-class reporting:**  
   In `Sim_Bilayer_Growth.cpp:1303`, `-native_hybrid_seconds 3600` bounds each `runNativeHybrid` call (per continuation substep), whereas `launch()` in `calibration/ewcal/runner.py:81–118` enforces `solver.timeout_seconds = 3600` across the entire `bin/shell` process. Furthermore, `termination_evidence()` in `runner.py:37–59` only labels `SIGABRT` as `numerical=True, timeout=True` when the final `[native_hybrid]` log line has `status=time_cap`; if adaptive continuation exhausts its cutbacks and aborts on `newton_stagnation`, `indefinite_curvature`, or `max_hessians`, `Evaluator` records `valid=False, failure_class='infrastructure'` alongside `solver.log` and `termination.json`. Under the stopping rules of `convention_corrected_investigation.md`, **any** non-accepted solve stops that case without retry and must be inspected in `solver.log` and reported accurately as a solver/continuation failure (not a physical shape error).
4. **Scientific interpretation boundaries:**  
   - These three solves are **model-conditional forward predictions** at frozen Attempt209 parameters under a researcher-directed coordinate/toolpath convention, **not** independent held-out validation or recalibration.
   - The scan target transformation (\(\operatorname{diag}(-1,-1,1)\), a \(180^\circ\) in-plane rotation) and the toolpath transformation (\(\alpha \mapsto -\alpha\) with \(z_0 = 0\) bottom clearance, a horizontal reflection plus longitudinal shift) are **not** equivalent relabellings; genuine re-solving is required.
   - Physical worked-face ambiguity, top-scan-as-midsurface approximation, mesh discretization (`res = 0.01` m), and final-only equilibrium path selection remain interpretation limits whether agreement improves or worsens.