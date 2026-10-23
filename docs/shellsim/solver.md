# The equilibrium solver (`native_hybrid`)

Source: `src/libshell/NativeHybridShellSolver.hpp`, `src/libshell/ReferenceHybrid/`. Settings: `shellsim/configs/default_adaptive_cutback.json` (the only default; see `shellsim/README.md`). Written from the code; no benchmark was rerun for this note.

## What it does

`native_hybrid` minimizes the bilayer shell energy in two phases:

1. **L-BFGS warmup** (`-native_hybrid_warmup_attempts 2000`) gets near the equilibrium.
2. **Newton** steps with the exact Hessian reach the tight gradient gate (`-native_hybrid_grad_tol 5e-14`). The switch is fixed (`-native_hybrid_handoff fixed`); the controller can also request a Newton probe on a gradient plateau. A plateau is never treated as convergence.

At most 40 Hessian factorizations are allowed per solve (`-native_hybrid_max_hessians 40`). After the solve, an independent endpoint check tests the final gradient and curvature (`ReferenceHybrid/EndpointCheck.hpp`).

The older HLBFGS-only path used approximate curvature from a limited gradient history and stalls near the end on nearly flat energy directions.

## The Hessian

- Exact analytic Hessian of the bilayer SVK shell energy, from TinyAD forward-mode autodiff of the per-face energy (`ReferenceHybrid/Hessian.hpp`). No finite-difference noise floor.
- Sparse; 21 degrees of freedom per face, laid out `[x | y | z | edge directors]`.
- Faces may be assembled in parallel (`-hessian_threads 8`), but the scatter into the matrix is serial in face/row/column order, so thread count does not change the summation order.

## CHOLMOD

Each Newton step solves `H p = -g`. The native hybrid requires the CHOLMOD backend (`NativeHybridShellSolver.hpp`; `Config.newtonBackend="cholmod"`):

- supernodal LLᵀ factorization (`ReferenceHybrid/CholmodFactorization.hpp`); ordering is fixed (external Eigen AMD, CHOLMOD natural), one linear-solver thread;
- the pivots are checked after factorization; a matrix that is not positive definite is rejected;
- with `-native_hybrid_refine_linear_solve true`: one residual correction using the same factors, and diagonal shifts for indefinite matrices; a correction that worsens the residual is not accepted.

Eigen's SimplicialLLT/LDLT (`ShiftedFactorization.hpp`) is an alternative backend in the source, not the production path.

## Cutback

If Newton stalls (`-native_hybrid_on_newton_stall cutback`, window 5, ratio 0.85, at most 2 stalls, at most 3 indefinite steps), the load step is halved instead of falling back to L-BFGS from the stalled point. Steps start at 1.0 and have a floor of 1/64, up to 8 retries (`-sequence_*`). The load snapping and halving rules are in `src/libshell/ContinuationStepping.hpp`. Time, iterations and Hessian counts are charged to one resource ledger.

## Limits

- Determinism is of the numerics (fixed assembly order, ordering and thread count), not bitwise equality across machines or thread counts; that was not checked.
- The solver benchmark (`docs/history/`) mixed binaries; read its qualification claims with that caveat.
- The upstream HLBFGS kernels are kept under their non-commercial license (header of `ReferenceHybrid/Core.hpp`); relevant for publication.
