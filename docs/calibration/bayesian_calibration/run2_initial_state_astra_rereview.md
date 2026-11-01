# Run 2 initial-state diagnostic — re-review

## Verdict: `APPROVED_WITH_CAVEATS`

**B1 is resolved. No unresolved launch-blocking findings were identified in the revised slice.**

This is a focused re-review of the classification repair, tests, prepared identities and execution controls. The unchanged mechanical/scoring assessment from the initial review is retained, with its stated limitations.

## Authority and reviewed identities

Authority: `docs/calibration/bayesian_calibration/convention_corrected_investigation.md:136–146`, including the separately authorized diagnostic and repair record. The historical six-parameter goal remains paused.

| Reviewed artifact | Recorded SHA-256 |
|---|---|
| `calibration/analysis/run2_initial_state_diagnostic.py` | `bfe1a3bb2baa552b4cf9798202463b9a6f773e95320ede966143a2d964443a98` |
| `calibration/tests/test_run2_initial_state_diagnostic.py` | `d74bf772995bac3c609ad45b7fa7888965f076006f7309a9ad981a5722908aa2` |
| `calibration/bo/run2_initial_state_diagnostic.sbatch` | `0cec79435645b95370f7962906c2213882af294c302689a7ef8957d1aa21a32e` |
| `run/diagnostics/run2_initial_state_a0084/inputs/manifest.json` | `9591f008aa762f429e2dae07dddad7ab181d065368d392d78d18fd392ec708ee` |
| Seed displacement CSV | `d5c39af5c5aead8a5b2c1d5aac0420ba139ebfb66f00927d88cdb196ec45a0a3` |
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |

The manifest retains the original a0084, target, sequence, evaluator/scoring and model-seed identities. The native seed remains SHA-256 `ad6a635b628017395b8b6f5557546528ebd6116e5c9cb8b41cee6ed1c35b9601`.

I inspected both manifest versions. The original manifest/CSV are retained under `run/diagnostics/run2_initial_state_a0084/inputs_pre_review_v1/`. Neither `flat/` nor `seeded/` existed at inspection.

**Evidence limit:** these are inspected recorded identities; I did not independently recompute hashes.

## B1 repair — cleared

`calibration/analysis/run2_initial_state_diagnostic.py:160–189` now refines only invalid, solver-phase infrastructure records with SIGABRT and matching evidence:

- Final native report plus terminal cycle-33 corrector exception.
- Matching recognized numerical status.
- `accepted=0`, finite energy/gradient, nonnegative gradient and sensible minimum work counts.
- Missing, malformed, unknown or mismatched evidence leaves the infrastructure classification unchanged.

The recognized statuses agree with `src/libshell/NativeHybridShellSolver.hpp:38–40` and the native engine’s cap statuses.

At `calibration/analysis/run2_initial_state_diagnostic.py:253–263`, the driver preserves the raw evaluator result, records diagnostic classification evidence separately, and uses that classification for continuation. Every invalid diagnostic discrepancy remains null. This repairs the premature-stop path without changing acceptance or treating a penalty as a physical error.

### Fixture adequacy

`calibration/tests/test_run2_initial_state_diagnostic.py:58–122` exercises the actual diagnostic driver with a mocked evaluator:

- Recognized indefinite-curvature failure → second start attempted.
- Controller timeout → second start attempted.
- Unknown abort or mismatched terminal status → stop.
- Raw classification/penalty preserved; diagnostic discrepancy null.

The original configuration-equivalence, seed reconstruction, mutation rejection and preparation-overwrite checks remain. This is adequate bounded coverage for B1; it is **not** compiled prescribed-start integration coverage.

The lead reports three tests, `py_compile` and batch syntax checks passing. I inspected their implementation and supplied attestation but did not rerun commands.

## Unchanged scientific and execution contract

The revised packet retains:

- Frozen a0084 parameters, including exact β/floor and zero bottom growth.
- Original 33-cycle Run 2 sequence, target, material frame, metric and acceptance.
- Position-only model seed imposed after cycle 32: no imported edge directors, scan coordinates, new constraint or intermediate re-equilibration.
- Final-only accumulation and zero slope feedback. The previously inspected source supports unchanged reference geometry, growth endpoint, boundary conditions and pass history. Different realized cutbacks remain part of the initialization response.
- Gradient ≤`5e-14` and independent positive-curvature verification.
- Flat then seeded, sequentially on one node; 8 CPUs, 16 GB, 02:40 walltime; 3900-second controller cap per start.
- No automatic restart, retry or overwrite.

`calibration/bo/run2_initial_state_diagnostic.sbatch:13` now sets `BLIS_NUM_THREADS=1` for both starts. This improves matched thread control but does not establish historical environment equivalence.

## Nonblocking caveats

1. Native 3600 seconds is **per continuation call**; the controller bounds the whole solver process. Trace export requires a native return, so controller termination can still leave incomplete traces.
2. No compiled prescribed-start integration fixture was exercised. Retain the existing imposed-position, history and terminal-load checks; a mismatch invalidates the comparison.
3. One ordered pair cannot establish failure probabilities or historical timeout causality. Small timing differences can include ordering, system-load and export/scoring overhead.
4. This remains an exploratory comparison on known Run 2 using a **position-only model seed**, not a checkpoint restart, calibration or unseen validation. Failures have no geometric discrepancy; identical outcomes do not rule out other initialization sensitivity.
5. Physical/model interpretation and parameter identifiability remain unresolved.

Approval applies only to this exact packet and procedure. It does not grant execution permission, additional attempts, budget extensions, criterion relaxation or scientific-test release, and does not assert runtime enforcement.