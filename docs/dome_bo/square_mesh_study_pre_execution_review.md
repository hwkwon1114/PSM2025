# Square-sheet mesh/tolerance study: pre-execution review

**Verdict: APPROVED_WITH_CAVEATS.**

**Launch-blocking findings: none for the reviewed 17-solve collection slice.** The driver implements the frozen design. Analysis and reporting limitations below must remain explicit; this approval does not establish mesh or tolerance independence.

## Scope, authority and identities

Authority reviewed: `docs/reports/dome_toolpath_investigation.md:384–446`, particularly the amendment, verification outcome, and frozen study protocol at lines 436–446. This authorizes a convergence study—not BO restart, new recipes, automatic retries, or finer-than-1-mm execution.

Hashes were **freshly computed by the supervisor**, not independently recomputed by this reviewer: this toolset provides no command execution. Full identities are recorded in:

`run/dome_square/mesh_study/checks/identity_manifest.sha256`

| Reviewed file | SHA256 |
|---|---|
| `calibration/dome/mesh_study.py` | `fd378604cccbad40abbb023ae6444eccae6902d7c4d845563fe8973b36c10107` |
| `calibration/dome/run_mesh_study.sbatch` | `214301beced3befc71c838c893acd941dfed194bed56d4bc68f168bd938990fb` |
| `calibration/dome/mesh_convergence.py` | `7a6acfb66e0bf22543dcffa4913448dd6cffa1ab4ceb618631e6ab8534139682` |
| `run/dome_square/frozen/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |

The supplied driver and sbatch prefixes **match**.

Reused identities, abbreviated; full hashes are in the manifest:

| File | SHA256 prefix |
|---|---|
| `calibration/dome/square_attempt.py` | `cb3708762e3b` |
| `calibration/dome/whole_sheet_diagnostics.py` | `07214aaa70d1` |
| `scripts/run_dome_continuation.py` | `201830532596` |
| `calibration/configs/solver/default_adaptive_cutback.json` | `94252b22fd97` |
| `calibration/dome/square_hatch.py` | `6557ef685f1c` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942` |
| `calibration/analysis/dome_continuous_objective.py` | `dda4e7717a86` |
| `docs/reports/dome_toolpath_investigation.md` | `687ea21fb0de` |

The manifest also identifies all six campaign requests and the reused verification request/status/state. Campaign and verification attempt-030 requests have identical SHA256:
`0dc74cb8e6a9d539c10e778dc9abd9630762c273d80fab1b5099594b1276c2cd`.

## Findings

### 1. Driver, isolation and numerical settings: correct

`calibration/dome/mesh_study.py:16–36` enumerates:

- indices 0–5: six 1-mm cells;
- indices 6–10: five fresh 2-mm cells, excluding attempt 30;
- indices 11–16: six fresh 4-mm cells.

Thus there are **17 fresh attempts and one reused cell**. Directory names distinguish candidate and resolution. `mkdir(exist_ok=False)` reserves each directory before copying its request; the worker independently refuses an existing `solve` directory. Duplicate launches cannot overwrite or concurrently solve the same cell. Requeueing an already-created cell would fail rather than resume it.

The monkeypatch **does take effect**: `square_attempt.run_attempt()` looks up its module-global `solver_options` when constructing the command (`calibration/dome/square_attempt.py:115`). Replacing `sa.solver_options` changes that lookup. The saved `base` function avoids recursion; each array task is a separate process, so the override does not leak into another task.

The resulting changes are exactly those declared:

- mesh `.1 .1 { .02, .01, .005 }`;
- both gradient options set to `1e-14`;
- verifier called with `gradient_ceiling=float(tolerance)`;
- native watchdog `14400` only for `.005`;
- process ceilings `5400/16200` seconds;
- worker scoring alarm raised to `3600` seconds.

The verifier checks the endpoint **and every accepted continuation substep**, requiring accepted gradient-target termination, positive pivots, the requested gradient ceiling, consistent artifacts/history, and exact final load one (`scripts/run_dome_continuation.py:19–60`). The deterministic Hessian/warmup settings remain unchanged.

`calibration/dome/run_mesh_study.sbatch:6–18` supplies eight CPUs, 32 GB, 5.5 hours, array concurrency four, thread controls and the exact frozen-binary hash check. The logs directory exists. No search or outer retry loop is present.

### 2. Failure classification: suitable, with collection caveats

`calibration/dome/square_attempt.py:125–168` preserves the reviewed distinctions:

- declared solver timeout or genuine numerical nonacceptance: `failed`, unscored;
- artifact/configuration/process/analysis exceptions: `error`, unscored;
- only verified, successfully scored endpoints: `accepted`.

Numerical failure intentionally exits zero; **Slurm completion is not scientific acceptance**. Startup exceptions, OOM, cancellation or allocation expiry may leave missing/`started` status. These must be reported as incomplete/error outcomes, not zero gates or evidence of mesh nonconvergence.

Unlike the BO controller, this fixed array does not globally cancel pending tasks after one task errors. Its concurrency bound is per submitted array, not a project-wide lock. Do not infer otherwise.

### 3. Attempt-030 reuse: legitimate

The retained command in `run/dome_square/verification/attempt_030/status.json` matches the new 2-mm settings: same binary, recipe, geometry, tolerance, threads, adaptive options and **1800-second native watchdog**.

Its process took `786.8797121495008` seconds, below both the historical 3600-second and proposed 5400-second process ceilings. The five retained continuation proofs show indefinite-curvature cutbacks followed by accepted substeps—not time-cap cutbacks. Each reported warmup-plus-Newton duration is roughly 140–160 seconds, far below 1800 seconds. Final gradient is `6.019115148211936e-16`, with positive pivots.

Consequently, the differing outer process allowance does not invalidate this completed result. The reused state identity is:

`9c859d193ac56d1d843793847a6f5d04511660fdad28fa0959ddca351e37d53b`

The 1-mm watchdog change is a separate, declared difference. Inspect new proof histories for time-cap-triggered cutbacks: those can affect continuation paths even when the eventual endpoint is accepted.

### 4. C1–C3: implemented, with tie/reporting limitations

`calibration/dome/mesh_convergence.py:17,59–76` correctly implements:

- **C1:** ROI RMS ≤ `1e-4 m` = **0.10 mm**;
- **C2:** absolute connected-fraction change ≤ `0.05` = **5 percentage points**, plus identical `>=0.80` classification;
- **C3:** Spearman correlation ≥ `0.9` and the same selected top candidate.

The gate uses the unchanged 15-mm estimator, square ROI, curvature floor and anisotropy criterion. All six candidates must have both accepted cells before the pair can pass. Missing/failed cells cannot silently produce a positive result.

**Nonblocking caveat—ties:** at lines 72–74, Spearman uses average ranks, while `np.argmax` selects the first maximum in frozen `CAND` order. This is deterministic, but does not establish an unchanged *set* of tied winners or a uniquely best candidate. Document that convention; do not choose a different tie rule after seeing outcomes.

If either ranking is constant—including all-zero gates—rho is NaN and C3 fails conservatively. Report this as **ranking unidentifiable / C3 not established**, not proof that identical flat outcomes changed ranking. Line 80 currently writes nonstandard JSON `NaN`; emitting `null` with an explanatory status would improve interoperability without changing the gate.

### 5. Mesh generation and shape comparison: valid for this design

The rectangle route uses `edgeLength = 2*Lx*res`
(`src/simulations/Sim_Bilayer_Growth.cpp:4951–4959`).

`src/libshell/Geometry.hpp:476–535` constructs an open regular grid with consistent diagonals. For `Lx=Ly=.1`, the requested resolutions yield:

| Resolution | Axis spacing | Vertices | Faces |
|---|---:|---:|---:|
| `.02` | 4 mm | 51² | 5000 |
| `.01` | 2 mm | 101² | 20000 |
| `.005` | 1 mm | 201² | 80000 |

These meshes are nested. “Edge” here means grid spacing; diagonal triangle edges are √2 larger.

`calibration/dome/whole_sheet_diagnostics.py:119–140` interpolates using the **actual coarse triangles**, not a replacement triangulation. Fine-face material centroids lie inside the common sheet, including boundary-adjacent faces; outside points explicitly raise an error. Barycentric coordinates and the proper-rotation, area-weighted Kabsch formula are correct.

The RMS is **fine-face-centroid quadrature**, not an exact continuous integral of squared displacement. ROI and whole-sheet comparisons receive separate optimal rigid alignments. Both choices are consistent with the declared implementation and should remain disclosed.

The current C++ geometry source was inspected as supporting evidence; I did not independently bind its complete source tree to the frozen executable or execute a mesh-generation fixture.

### 6. Curvature cost and wall safeguards: bounded collection, unqualified scale

`calibration/analysis/dome_curvature.py:25` materializes neighborhoods for all faces. At 1-mm spacing, an interior neighborhood contains approximately 707 vertices at 15 mm and 1257 at 20 mm: roughly **57 million / 101 million neighbor references** before boundary reductions. Python lists/integers can therefore consume several GB. With fixed physical radius, total neighborhood work scales approximately as `h^-4`.

No demonstrated 32-GB overrun was found, but 80,000-face runtime and peak memory are unqualified.

Important distinctions:

- `whole_sheet_diagnostics.analyse()` caches 15-mm curvature.
- `square_attempt.analyze_state()` does not use that cache.
- Aggregate analysis computes 10/15/20-mm curvature again and, for uncached states, duplicates the 15-mm computation.
- **The 3600-second worker alarm does not bound `mesh_convergence.py`.**

Run aggregate analysis on a compute allocation with appropriate memory/wall limits and controlled BLAS threading—not on the login node. Reusing already-computed curvature could reduce cost, but is not a solve-launch gate.

**Wall-margin caveat:** `16200 + 3600 = 19800` seconds, exactly the 5.5-hour allocation. This leaves no worst-case allowance for imports, verification, JSON loading or final status writes. Slurm still bounds resource use, but near-limit tasks may terminate incompletely. Reserving overhead within the agreed wall limit would improve reliability. SIGALRM is not a strict hard deadline for every native-library call.

### 7. Scientific relevance and reporting completeness

The screening ledger independently supports the declared selection:

| Attempt | Screening J |
|---|---:|
| 30 | 0.1297392072 |
| 31 | 0.1082023815 |
| 21 | 0.0484514481 |
| 32 | 0.0417276257 |
| 27 | 0.0299913659 |

Attempt 19 is the highest-connected-fraction zero-J case, at approximately 58.09%. Thus the implemented selection needs **no fine-mesh values**. However, the study was designed after the attempt-030 discrepancy was known; this is not a blinded holdout experiment.

The study answers whether these six fixed recipes are stable across the specified meshes **at fixed `1e-14` tolerance**. It does not independently establish tolerance invariance, negligible optimizer error, asymptotic convergence, global-equilibrium uniqueness, calibration transfer validity or physical dome identification.

Rival explanations for discrepancies remain treatment discretization, equilibrium-branch changes, estimator sensitivity and watchdog-triggered continuation changes. The study measures their combined practical effect; it does not isolate each mechanism.

**Reporting omissions, not collection blockers:** `calibration/dome/mesh_convergence.py:61–64` retains ROI p99 and whole-sheet RMS, but omits the protocol’s whole-sheet p99. It also omits hit-count totals. Both can be recovered from retained artifacts without new solves. Add them before the final scientific report. Geometry statistics are useful but do not replace visual inspection or certify absence of self-intersections.

Cache files lack state/estimator identity binding (`calibration/dome/whole_sheet_diagnostics.py:45–51`). Fresh isolated cells reduce this risk, and primary gates are recomputed independently; preserve state immutability and invalidate caches if inputs or estimator code change.

## Evidence and limitations

Performed: static inspection of the protocol, reviewed scripts/configuration, reused verifier/estimator/objective, selection ledger, request/status/proof histories, C++ mesh route, relevant test sources and supplied hash manifest.

The retained bounded check:

`run/dome_square/mesh_study/checks/analysis_pair_check.py`  
→ `run/dome_square/mesh_study/checks/mc_test2.json`

reproduces gate `0.43976562500003286`, ROI RMS `0.5726711689654506 mm`, and whole-sheet RMS `0.8596357082523712 mm`. Its coarse input is historical **5e-14 screening data**, so this validates analysis compatibility—not a study cell. With one candidate it does not validate six-candidate C3 behavior.

Not performed: commands, test execution, solves, independent hashing, 80,000-face profiling, scheduler checks or timeout/resumption exercises. No restart/resumability claim is made.

Approval applies only to the identified slice. Preserve these identities throughout collection; the wrapper checks the binary, not every imported source/configuration identity. This review grants no execution permission, budget extension, automatic retry, BO restart or sealed-test release.