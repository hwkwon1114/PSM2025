# shellsim: simulate a toolpath

The C++ bilayer shell solver (`src/`, built to `bin/shell`) reads a toolpath as JSON. `shellsim` is the thin Python layer to write that JSON and run the solver.

```python
from shellsim import toolpath, simulate
seq = toolpath.sequence([('A', toolpath.zigzag(lv_mm=140, alpha_deg=9.13, n_strips=10, width_mm=10, gtop=0.003))])
state = simulate.run(seq, 'run/my_case', lx=0.1, ly=0.1, res=0.02, h_total=0.00077)
x, y, z = simulate.height(state)
```

- `toolpath.py`: build a sequence of zigzag passes (JSON schema: `docs/shellsim/zigzag_sequence_format.md`).
- `simulate.py`: run `bin/shell` with the shared solver settings and read the equilibrium state.
- `configs/default_adaptive_cutback.json`: the one solver configuration (adaptive load steps, native hybrid L-BFGS + exact Newton with cutback, gradient gate 5e-14). `solver_settings.solver_options()` reads it. It is the default for calibration, `dome_bo/` and `shellsim`; `docs/README.md` lists the deliberate exceptions in `legacy/`.
- `examples/notebooks/simulate_toolpath.ipynb`: toolpath, solve and plot in one notebook (run it from the repository root).
- `examples/trajectories/`: ready-made toolpath JSON files; `examples/printpaths/`: SVG-to-toolpath input converter.

Build the solver first: see the root `README.md`.
