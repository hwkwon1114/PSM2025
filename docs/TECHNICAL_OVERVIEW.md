# PSM 2025 technical overview and review guide

This is the canonical, code-grounded overview of the repository. It is organized
so mechanics, eigenstrain, optimization, runtime, and data-generation choices can
be reviewed independently. It describes the current workspace on branch `main` at
base commit `465618e`, including initialization and metric-continuation experiments
in the working tree.

Three labels are used:

- **Implemented**: behavior directly visible in the cited source.
- **Interpretation**: mathematical or modeling meaning inferred from that behavior;
  this needs mechanics review.
- **Review item**: a decision, ambiguity, or missing validation.

## 1. Scope and review map

The code solves quasistatic equilibrium shapes of triangulated monolayer and
bilayer shells. A simulation prescribes per-face reference geometry, and a local
energy minimizer changes the shell configuration to reduce stretching and bending
energy. The primary neural-operator use case is

\[
\text{prescribed eigenstrain/target metric field}
\longmapsto \text{equilibrium deformation field}.
\]

This is not automatically a one-to-one map. One target metric can admit multiple
local minima, and the selected shape can depend on initialization, continuation,
noise, mesh, and solver termination.

| Review section | Primary question | Status | Reviewer notes |
|---|---|---|---|
| Architecture and boundaries | Are mechanics and simulation policy separated? | Pending | |
| Discretization and kinematics | Are the discrete forms and signs correct? | Pending | |
| Constitutive mechanics | Are energy factors, laws, and thickness scaling correct? | Pending | |
| Eigenstrain/target metric | What is the canonical operator input? | Pending | |
| Boundary conditions and gauge | How are rigid modes removed for learning? | Pending | |
| Optimization | What qualifies as an accepted equilibrium? | Pending | |
| Initialization and continuation | Which branches should enter the dataset? | Pending | |
| Runtime and reproducibility | Can another machine reproduce a run? | Pending | |
| Outputs and lineage | Does every sample retain complete provenance? | Pending | |
| Verification | Do tests support the scientific claims? | Pending | |

Reviewers can update `Pending` and add decisions directly. The section checklists
are acceptance criteria.

## 2. Architecture and responsibility boundaries

### 2.1 Execution flow

```text
command-line key/value pairs
          |
          v
src/simulations/main.cpp
          | selects simulation and case
          v
geometry -> topology + rest/current configurations + boundary conditions
          |
          v
simulation-generated target forms (a_bar, b_bar)
          |
          v
CombinedOperator_Parametric
  current forms (a, b) -> strain -> per-face energy + analytic gradient
          |
          v
HLBFGS local minimization over vertices and edge directors
          |
          v
VTP geometry/fields + energy and solver diagnostic text files
```

The major ownership boundary is:

- [`src/libshell/`](../src/libshell/) owns mesh state, discrete kinematics,
  materials, energy/gradient evaluation, optimizer wrapping, geometry, and I/O.
- [`src/simulations/`](../src/simulations/) owns executable dispatch, prescribed
  growth fields, continuation schedules, initial conditions, case defaults, and
  output naming.
- [`external/hlbfgs/`](../external/hlbfgs/) is the vendored local optimizer.
- [`test/testshell/`](../test/testshell/) contains numerical and unit tests.
- [`scripts/`](../scripts/) and [`printpaths/`](../printpaths/) are historical
  experiment/toolpath utilities.
- [`notebooks/`](../notebooks/) contains Python-side experiment orchestration and
  analysis; it does not replace the C++ mechanics kernel.

The intended separation is explicit in
[`GrowthContinuation.hpp`](../src/simulations/GrowthContinuation.hpp):
`MetricContinuation` is numerical simulation policy, while
[`GrowthHelper.hpp`](../src/libshell/GrowthHelper.hpp) contains metric
representation and mechanics-adjacent transformations.

**Review checklist**

- [ ] Confirm `libshell` has no case-specific continuation or sampling policy.
- [ ] Decide whether file I/O belongs in the mechanics library or a separate layer.
- [ ] Define the stable API that Python/neural-operator tooling should call.

## 3. Mesh state and discrete shell kinematics

### 3.1 Mesh state

A monolayer `Mesh` combines a `DCSCurrentConfiguration` with a
`DCSRestConfiguration`; `BilayerMesh` uses a rest configuration with separate
bottom and top target metrics. See [`Mesh.hpp`](../src/libshell/Mesh.hpp) and
[`DCSConfigurations.hpp`](../src/libshell/DCSConfigurations.hpp).

At initialization, geometry supplies vertices, triangular faces, and vertex
constraints. The mesh constructs topology, copies the same initial vertices into
current and rest configurations, initializes scalar edge directors, constructs
boundary conditions, and updates derived geometric data.

The optimizer has

\[
N_{\mathrm{dof}}=3N_v+N_e
\]

variables: all vertex coordinates followed by one scalar director per edge. The
layout is defined by [`EnergyOperator.hpp`](../src/libshell/EnergyOperator.hpp) and
[`ConfigurationData.hpp`](../src/libshell/ConfigurationData.hpp). Fixed components
remain in storage, but their gradient contributions are omitted.

### 3.2 First fundamental form

For each triangle, the code uses

\[
e_0=v_1-v_0,\qquad e_1=v_2-v_1,\qquad e_2=v_0-v_2.
\]

The current first fundamental form is

\[
a=
\begin{bmatrix}
e_1\!\cdot e_1 & e_1\!\cdot e_2\\
e_1\!\cdot e_2 & e_2\!\cdot e_2
\end{bmatrix}.
\]

This convention is implemented in
[`ExtendedTriangleInfo.hpp`](../src/libshell/ExtendedTriangleInfo.hpp); edge
definitions are in [`TriangleInfo.hpp`](../src/libshell/TriangleInfo.hpp).

### 3.3 Second fundamental form

Each edge has a signed dihedral angle `theta` and an independent director angle
`phi`. With the triangle-edge orientation sign, the local angle is

\[
\alpha_e=\frac{1}{2}\theta_e+\operatorname{sign}_e\phi_e.
\]

The discrete second form is assembled from triangle heights and `sin(alpha_e)`,
with prescribed normals for clamped edges. Dihedrals use adjacent face normals and
a signed `atan2` expression. See
[`CreateExtendedTriangleInfos.hpp`](../src/libshell/CreateExtendedTriangleInfos.hpp)
and [`ExtendedTriangleInfo.hpp`](../src/libshell/ExtendedTriangleInfo.hpp).

Both forms have analytic derivatives. Stretching depends on the three vertices of
a face. Bending also depends on opposite vertices across adjacent edges and on
edge-director variables. Per-face work is parallelized with TBB and merged into the
global gradient by
[`MergePerFaceQuantities.hpp`](../src/libshell/MergePerFaceQuantities.hpp).

**Review checklist**

- [ ] Supply the intended reference/citation for `DCS` and define the acronym.
- [ ] Verify edge orientation and the sign of `theta`, `phi`, `alpha`, and `b`.
- [ ] Confirm the edge-director degree of freedom is the intended shell model.
- [ ] Verify behavior on boundaries, nonmanifold meshes, and poor-quality elements.

## 4. Non-Euclidean plate mechanics

### 4.1 Prescribed and current geometry

For every face the mechanics consumes:

- current first and second forms, `a` and `b`, recomputed from the deformed state;
- prescribed first and second forms, `a_bar` and `b_bar`, stored in the rest
  configuration; and
- face material properties.

The core operator is
[`CombinedOperator_Parametric`](../src/libshell/CombinedOperator_Parametric.hpp).
The monolayer growth cases overwrite only `a_bar`; the initially flat `b_bar`
remains unchanged. Thus those cases prescribe in-plane incompatibility, not
spontaneous curvature.

### 4.2 Isotropic strain and energy as implemented

The code defines

\[
S_a=\bar a^{-1}a-I,\qquad
S_b=\bar a^{-1}(b-\bar b).
\]

For either matrix `S`, the material norm is

\[
\lVert S\rVert_C^2 A_{\bar a}
=\left[c_1(\operatorname{tr}S)^2+c_2\operatorname{tr}(S^2)\right]
  \frac{1}{2}\sqrt{\det\bar a},
\]

where

\[
c_1=\frac{E\nu}{2(1-\nu^2)},\qquad
c_2=\frac{E}{2(1+\nu)}.
\]

For a monolayer, the per-face contributions are

\[
W_s=\frac{h}{4}\lVert S_a\rVert_C^2 A_{\bar a},\qquad
W_b=\frac{h^3}{12}\lVert S_b\rVert_C^2 A_{\bar a}.
\]

These are code-implemented conventions, not a replacement for a derivation.
`S_a` is twice the common metric-strain definition, which explains the `h/4`
scaling. Sources:
[`EnergyHelper_Parametric.hpp`](../src/libshell/EnergyHelper_Parametric.hpp) and
[`MaterialProperties.hpp`](../src/libshell/MaterialProperties.hpp).

The total operator reports

\[
E=E_{aa}+E_{bb}+E_{ab}.
\]

A monolayer has `E_ab=0`. Each bilayer layer uses half the monolayer stretching
and bending weights, plus a signed thickness-squared stretch-bend coupling:

\[
h_{aa}=h/8,\quad h_{bb}=h^3/24,\quad
h_{ab}=+h^2/8\;\text{(bottom)},\;-h^2/8\;\text{(top)}.
\]

### 4.3 Materials

Isotropic data contain `E`, `nu`, `h`, and density. Constant and per-face
containers are available. Orthotropic data contain `E1`, `E2`, `G12`, `nu1`,
`nu2`, `h`, and density, with reciprocity imposed as

\[
\nu_2=\nu_1E_2/E_1.
\]

Orthotropic energy is in
[`EnergyHelper_Parametric_Ortho.hpp`](../src/libshell/EnergyHelper_Parametric_Ortho.hpp).
The `basic_disk` path uses constant isotropic material: `E=1`, `h=0.01`, and
`nu=0.3` by default; validation fixes `nu=0.5`.

### 4.4 Diagnostic strain convention

For monolayer isotropic output, `computePerFaceStrains` reports

\[
\epsilon_s=\tfrac12\bar a^{-1}(a-\bar a),\qquad
\epsilon_b=-\bar a^{-1}(b-\bar b).
\]

The exported bending-strain sign differs from internal `S_b`. Quadratic monolayer
energy is sign-insensitive, but stress and branch interpretation may not be.
Orthotropic per-face strain/stress diagnostics are incomplete even though
orthotropic energy and gradients are implemented.

**Review checklist**

- [ ] Re-derive all area, `1/2`, thickness, and bilayer coupling factors.
- [ ] Confirm physical and exported bending-strain sign conventions.
- [ ] Confirm whether `nu=0.5` is intentional in validation.
- [ ] Document physical units and nondimensionalization for `E`, `h`, and geometry.
- [ ] Validate the orthotropic law and its admissible parameter range.
- [ ] Decide whether spontaneous curvature `b_bar` belongs in the operator input.

## 5. Eigenstrain and target-metric representation

### 5.1 What the solver consumes

The solver does not store a field named `eigenstrain`. Its effective input is the
symmetric positive-definite target metric `a_bar[f]` for every face. Growth factors
and directions are one parameterization used to construct it.

`DecomposedGrowthState` stores principal stretches `s1`, `s2` and orthonormal
tangent directions `v1`, `v2`. Given triangle reference basis `rxy`, it constructs
`a_bar` in that basis. It also decomposes a metric by a generalized eigenproblem.
See [`GrowthHelper.hpp`](../src/libshell/GrowthHelper.hpp).

Legacy helpers accept increments `g_i` and use stretches `1+g_i`, so metric
contributions scale as `(1+g_i)^2`. The `basic_disk` CLI follows this convention:
it adds one to both `-s1` and `-s2`.

### 5.2 Built-in monolayer target fields

`-growthcase ortho` assigns constant principal stretch increments in directions
rotated relative to radial/azimuthal directions. `-growthangle` is in units of
`pi`: zero is radial and `0.5` is azimuthal. Since direction is undefined at the
origin, this case normally requires `-innerR > 0`.

`-growthcase spherical` uses the isotropic stereographic factor

\[
s(r)=\frac{2}{1+r^2}
\]

on a radius-normalized disk. `-growthcase validation` uses unit radial stretch and
azimuthal stretch `sin(r)/r`.

### 5.3 Recommended neural-operator input

For a mesh-fixed first model, store the target metric relative to original metric
`a0`. A proposed coordinate-stable three-channel representation is

\[
G_f=\tfrac12\log(a_{0,f}^{-1/2}\bar a_fa_{0,f}^{-1/2}).
\]

This is a recommendation, not current solver behavior. It avoids principal
direction discontinuities when stretches are equal and distinguishes physical
input from its parameterization. Retain raw `a_bar`, `a0`, and all generating
parameters for auditability.

**Review checklist**

- [ ] Choose the term: eigenstrain, growth tensor, or target metric.
- [ ] Choose stored input: `a_bar`, growth parameters, or log strain.
- [ ] Define the local basis and tensor transformation convention.
- [ ] Verify every `a_bar` is finite and positive definite.
- [ ] Decide whether material, thickness, geometry, and boundary conditions are
  fixed context or additional operator inputs.

## 6. Boundary conditions, invariance, and output gauge

Vertex constraints are stored per coordinate. Edge constraints store fixed
directors and optional prescribed normals. Fully fixed endpoints can identify a
clamped edge; its prescribed normal enters the second form. See
[`BoundaryConditionsData.hpp`](../src/libshell/BoundaryConditionsData.hpp).

The current `basic_disk` geometry is free: it imposes no displacement constraints.
The energy therefore has rigid translation and rotation modes. Raw coordinates are
not a unique learning target even if the physical shape is unique.

For a neural-operator dataset, use one documented gauge. Recommended
post-processing is:

1. subtract the deformed centroid;
2. align vertices to the reference mesh by a proper-rotation Procrustes/Kabsch
   transform;
3. store aligned displacement `u=x_aligned-x_reference`; and
4. store the removed rigid transform so the operation is reversible.

Reflection must not be silently allowed. Mirror-related equilibria can be distinct
branches and should retain an explicit branch label.

**Review checklist**

- [ ] Choose constraints in the solve or deterministic post-alignment.
- [ ] Define whether mirror images are equivalent or separate branches.
- [ ] Test alignment near symmetric configurations where orientation is ambiguous.
- [ ] Ensure edge-director outputs, if retained, use a compatible gauge.

## 7. Optimization and equilibrium selection

### 7.1 Solver path

```text
Sim_MonoLayer_Growth::run_basic_disk
  -> Sim::minimizeEnergy
    -> HLBFGS_Energy::minimize
      -> vendored HLBFGS
        -> CombinedOperator_Parametric::compute
```

HLBFGS is a gradient-based local quasi-Newton method. It does not certify a global
minimum, uniqueness, or stability. Initial state and continuation can select
different basins.

### 7.2 Active settings

| Setting | Current value |
|---|---:|
| Limited-memory history `M` | 10 |
| Line-search `ftol` | `1e-4` |
| Line-search `xtol` | `1e-16` |
| Line-search `gtol` | `0.9` |
| Relative gradient tolerance | caller-supplied |
| Absolute gradient tolerance | `1e-16` |
| Strategy | M1QN3 (`INFO[3]=1`) |
| Accurate Hessian | disabled |
| Preconditioned CG | disabled |
| Update correction | Zhang-Xu (`INFO[13]=3`) |
| Maximum line-search evaluations | 20 |
| Default maximum iterations | `1,000,000,000` |

Values are set in
[`HLBFGS_Wrapper.hpp`](../src/libshell/HLBFGS_Wrapper.hpp) and bundled
[`HLBFGS.cpp`](../external/hlbfgs/HLBFGS.cpp).

### 7.3 Termination codes and caveats

| Raw code | Meaning | Wrapper classification |
|---:|---|---|
| 1 | line search failed | failure |
| 2 | relative gradient criterion met | success |
| 3 | absolute gradient criterion met | success |
| 4 | line search cannot improve further | success |
| 5 | maximum iterations exceeded | failure |

Important implemented behavior:

- `basic_disk` creates `eps=1e-2`, but non-stepwise minimization passes machine
  epsilon (about `2.22e-16`) as relative tolerance.
- Code 4 is accepted even though it does not establish a small gradient.
- The simulation ignores the wrapper result and still writes a shape.
- `main()` returns zero if it reaches the end, regardless of optimizer status.
- Maximum-iteration checking is off by one: `-maxiterations 20` can report 21.

Neither file existence nor process exit status establishes convergence.

### 7.4 Minimum dataset acceptance record

Every attempted solve should retain:

- raw HLBFGS code and stricter project accepted/rejected classification;
- initial/final energy, components, and final gradient norm;
- iteration and function-evaluation counts;
- finite-value and mesh quality/inversion checks;
- initialization, seed, continuation schedule, and preceding stage;
- mesh/topology identity and solver/code version; and
- optional lowest Hessian eigenvalues or perturbation stability check.

**Review checklist**

- [ ] Define tolerances scaled to energy and degrees of freedom.
- [ ] Decide whether code 4 needs an independent gradient threshold.
- [ ] Propagate solver status into structured output and process status.
- [ ] Correct or document maximum-iteration semantics.
- [ ] Add stability checks before calling a configuration a local minimum.
- [ ] Treat global optimality as unproven without a separate search.

## 8. Continuation, initialization, and branch sampling

### 8.1 Metric continuation

Each face has a `MetricContinuation` from initial to target SPD metric. For an
isotropic initial state, default interpolation is linear in principal stretches
while keeping target directions; `-interp_logeucl true` interpolates logarithms.
The general fallback interpolates metrics linearly or by matrix log/exp. Endpoints
are unit-tested.

The fixed `basic_disk` fractions are:

```text
0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
0.50, 0.60, 0.70, 0.80, 0.90, 1.00
```

`-maxstages N` takes the first `min(N,12)` stages. `0` performs no solve. Each stage
warm-starts from the preceding configuration, so schedule and path are
branch-selection inputs.

### 8.2 Starting shapes and noise

| Option | Default | Meaning |
|---|---:|---|
| `-initmode` | `legacy` | initialization policy |
| `-initseed` | `42` | nonnegative random seed |
| `-initamp` | `0.01*h` | height amplitude |
| `-initwaves` | `4` | wrinkle azimuthal wave count |
| `-initedgenoise` | `0` | initial director perturbation |
| `-stepnoise` | mode-dependent | per-stage vertex perturbation |
| `-stepedgenoise` | mode-dependent | per-stage director perturbation |

Explicit modes are `flat`, `random`, `dome_up`, `dome_down`, `saddle`, and
`wrinkle`; formulas are in
[`InitialShapePerturbation.hpp`](../src/simulations/InitialShapePerturbation.hpp).
Random, dome, and wrinkle vanish at the outer radius. Saddle does not.

Explicit modes default to no continuation noise, isolating the starting shape.
Legacy mode adds deterministic seeded vertex and director noise at every stage.

### 8.3 Using multiple designs

For each target metric, run a controlled ensemble across initialization families,
seeds, amplitudes, continuation rules, and possibly mesh refinement. After quality
filtering and rigid alignment:

- cluster shapes using geometry-aware distance, not raw coordinates;
- treat observed frequency as basin accessibility, not thermodynamic probability;
- use a branch-conditioned or multimodal operator when stable branches persist;
- retain rejected/unconverged attempts as solver diagnostics, not equivalent
  physical targets; and
- split train/test by target-metric family before expanding seeds, preventing
  near-duplicate leakage.

Categorical initialization, continuation, or termination labels can be embedded by
an LVGP in a separate surrogate for branch probability, failure probability, or
model discrepancy. `Converged` is mainly a quality outcome, not a physical input to
the deformation operator.

**Review checklist**

- [ ] Define epistemic uncertainty versus numerical branch search.
- [ ] Define a shape distance and branch-clustering threshold.
- [ ] Separate stable alternative equilibria from optimizer failures.
- [ ] Decide whether continuation schedules should be configurable.
- [ ] Test branch persistence under tighter tolerance and mesh refinement.

## 9. Simulation and command-line reference

### 9.1 CLI contract

Every argument is a `-key value` pair. There are no valueless flags; booleans must
be literal `true` or `false`; duplicate keys are rejected; unknown keys are not.
There is no `--help` schema. Options and defaults append to `argumentparser.log`.
See [`ArgumentParser.hpp`](../src/libshell/ArgumentParser.hpp).

Top-level modes in [`main.cpp`](../src/simulations/main.cpp) are:

| Command | Purpose |
|---|---|
| `-sim monolayer_growth` | monolayer growth cases |
| `-sim bilayer_4dfilaments` | printpath-driven bilayer cases |
| `-sim IO` | mesh/file conversion |

Monolayer cases are `cone`, `edgegrowth`, and `basic_disk`. The first two are
historical fixed-form experiments; `basic_disk` is the current initialization/UQ
surface.

### 9.2 `basic_disk` options

| Option | Default | Notes |
|---|---:|---|
| `-R` | `1.0` | outer radius |
| `-res` | `128` | disk resolution; annulus uses it to derive edge length |
| `-innerR` | `0.0` | nonzero selects an annulus |
| `-growthcase` | `ortho` | `ortho`, `spherical`, or `validation` |
| `-growthangle` | `0.0` | multiples of `pi`; orthotropic only |
| `-s1` | `1.0` | increment; actual stretch is `1+s1` |
| `-s2` | `0.0` | increment; actual stretch is `1+s2` |
| `-E` | `1.0` | Young modulus |
| `-h` | `0.01` | thickness |
| `-interp_logeucl` | `false` | metric continuation rule |
| `-initmode` | `legacy` | see initialization modes above |
| `-initseed` | `42` | must be nonnegative |
| `-initamp` | `0.01*h` | initial height amplitude |
| `-initwaves` | `4` | positive for `wrinkle` |
| `-initedgenoise` | `0` | one-time director noise |
| `-stepnoise` | mode-dependent | continuation vertex noise |
| `-stepedgenoise` | mode-dependent | continuation director noise |
| `-maxstages` | `12` | truncates fixed schedule |
| `-maxiterations` | `1e9` | positive HLBFGS limit; off by one |

Default `-s1 1.0` produces stretch factor 2.0 and deserves review. Most historical
runs supply a smaller increment.

A bounded review run is:

```bash
mkdir -p run/review_dome
cd run/review_dome
../../bin/shell \
  -sim monolayer_growth -case basic_disk \
  -R 1 -res 24 -innerR 0.15 \
  -growthcase ortho -growthangle 0.5 -s1 0.20 -s2 0 \
  -E 1 -h 0.01 -interp_logeucl false \
  -initmode dome_up -initseed 7 -initamp 0.03 \
  -initedgenoise 0 -stepnoise 0 -stepedgenoise 0 \
  -maxstages 1 -maxiterations 20
```

The 20-iteration cap is for plumbing checks, not accepted scientific data.

### 9.3 Bilayer printpath cases

Cases are `vtu2stl`, `catenoid_printpath`, `helicoid_printpath`,
`sombrero_printpath`, `logspiral_printpath`, `folding_flower_printpath`, and
`orchid_printpath`. Their workflow is:

1. construct case-specific plate geometry;
2. map bottom/top path data to per-face density and direction;
3. create layer material and target-metric fields;
4. apply a fixed swelling continuation;
5. minimize combined bottom/top energy; and
6. write VTP and energy components.

Common controls include `-E`, `-nu`, `-filthickness`, `-fname_bot`, `-fname_top`,
`-sigma`, `-dpath`, swelling factors, orthotropic parameters, and plane-penalty
settings. See
[`Sim_Bilayer_4DFilaments.cpp`](../src/simulations/Sim_Bilayer_4DFilaments.cpp).

Historical defaults contain absolute `/Users/wvanrees/...` paths. The repository
contains SVGs while C++ consumes converter-generated `.dat` files. The converter
uses Python 2 syntax and undeclared packages. These cases are not reproducible from
defaults without modernization.

### 9.4 File conversion

`-sim IO` reads OBJ, OFF, or VTP and selects output by extension:

```bash
../../bin/shell -sim IO -filename input.vtp -filename_out output.stl -ascii false
```

Outputs are STL, OBJ, Abaqus INP (`-element STRI3` by default), and cell-data CSV
as fallback. See [`Sim_IOops.cpp`](../src/simulations/Sim_IOops.cpp).

## 10. Outputs and data lineage

Run every experiment in a separate directory. Outputs have fixed names, and
`argumentparser.log`, diagnostics, and energy logs use append mode.

| `basic_disk` file | Contents |
|---|---|
| `argumentparser.log` | supplied options and recorded defaults |
| `test_basic_disk_init.vtp` | initial geometry and final target fields |
| `test_basic_disk_initialization.vtp` | explicit perturbed starting geometry |
| `test_basic_disk_final_00.vtp` | initialized zero-continuation geometry |
| `test_basic_disk_final_01.vtp`, ... | post-minimization stage geometries |
| `test_basic_disk_diagnostics.dat` | periodic and terminal HLBFGS records |
| `test_basic_disk_energies.dat` | appended component energies |
| `test_basic_disk_final_energies.dat` | validation-only final components |

`dumpOrthoNew` writes face scalars `rate1`, `rate2`, vectors `dir1`, `dir2`, and
curvatures `gauss`, `mean`; geometry is the current vertex state.

Critical caveat: `rate*` and `dir*` come from final target decomposition, not the
metric actually applied at an intermediate stage. Stage 0 and 1% files still
display final target parameters. Reconstruct the applied metric from schedule and
interpolation, or change the writer before using those fields as training input.

Text diagnostics are not self-describing enough for robust assembly: energy rows
lack stage IDs, reruns mix records, and process status does not identify optimizer
failure. A structured manifest should give each run/stage a unique ID and named
provenance.

**Review checklist**

- [ ] Decide whether output stores final target, applied stage metric, or both.
- [ ] Add run/stage identifiers and a schema version.
- [ ] Use truncate-or-unique files rather than implicit append.
- [ ] Verify writers report errors and no fields silently default to zero.
- [ ] Define the dataset manifest before producing a large ensemble.

## 11. Build and reproducibility

### 11.1 Targets and dependencies

CMake builds `libshell`, vendored `hlbfgs`, the `shell` executable, and aggregate
`testshell`. The project requires C++23; the nearby CMake comment saying C++14 is
stale. Eigen, TBB, and VTK are required. OpenMP and GSL are optional in code,
although some tests may need GSL. Triangle, libigl, and GoogleTest use
`FetchContent`; their `master`/`main` revisions are unpinned.

The local setup recipe is [`env_settings`](../env_settings). A condensed
Linux/Conda build is:

```bash
conda activate smcpp_vtk38

cmake -S . -B build_cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
  -DVTK_DIR="$CONDA_PREFIX/lib/cmake/$(ls "$CONDA_PREFIX/lib/cmake" | grep -E '^vtk-9\.' | head -n1)" \
  -DTBB_DIR="$CONDA_PREFIX/lib/cmake/TBB" \
  -DGSL_ROOT_DIR="$CONDA_PREFIX"

cmake --build build_cmake --parallel 2
ctest --test-dir build_cmake --output-on-failure
```

Once dependencies are populated, offline configuration can use
`-DFETCHCONTENT_FULLY_DISCONNECTED=ON`.

### 11.2 Build risks

- Moving dependency branches make rebuilds non-reproducible.
- CMake hard-codes Homebrew root hints although Linux setup uses Conda.
- `file(GLOB ...)` may require reconfiguration after adding source files.
- Source-root `bin/` and `lib/` can collide across build configurations.
- `-march=native -mtune=native` reduces release-binary portability.
- Tracked `build_cmake/` contents create noisy working-tree changes.

**Review checklist**

- [ ] Pin dependency commits and record compiler/library versions.
- [ ] Decide supported C++ standard and correct the stale comment.
- [ ] Remove machine-specific roots or make them cache options.
- [ ] Decide whether build products remain in the source tree.
- [ ] Establish clean builds on the lab machine and cluster.

## 12. Verification coverage

CMake registers one CTest entry, `test1`, which runs 21 GoogleTests covering:

- six basic discrete-gradient variants;
- bending and stretching convergence;
- isotropic/orthotropic energy gradients across mono/bilayer variants;
- zero energy in the undeformed rest state;
- five initialization-profile behaviors; and
- metric-continuation endpoints.

Sources:
[`Test_BasicGradients.cpp`](../test/testshell/Test_BasicGradients.cpp),
[`Test_StretchingOperator_Convergence.cpp`](../test/testshell/Test_StretchingOperator_Convergence.cpp),
[`Test_BendingOperator_Convergence.cpp`](../test/testshell/Test_BendingOperator_Convergence.cpp),
[`Test_EnergyOperatorDCS_Gradient.cpp`](../test/testshell/Test_EnergyOperatorDCS_Gradient.cpp),
[`test_energy_staticeq.cpp`](../test/testshell/test_energy_staticeq.cpp),
[`Test_InitialShapePerturbation.cpp`](../test/testshell/Test_InitialShapePerturbation.cpp),
and [`Test_MetricContinuation.cpp`](../test/testshell/Test_MetricContinuation.cpp).

Missing or weak coverage includes:

- end-to-end monolayer and bilayer cases;
- optimizer return propagation and acceptance classification;
- full continuation schedule;
- mesh validity/inversion and stability;
- geometry and boundary conditions;
- deterministic repeatability;
- VTP/STL/OBJ/INP round trips (`Test_WriteReadMesh.cpp` is effectively empty);
- CLI dispatch/validation; and
- legacy scripts and printpath conversion.

**Review checklist**

- [ ] Match each scientific claim to a unit, convergence, or integration test.
- [ ] Add a small deterministic production-path regression.
- [ ] Test optimizer failure/success and structured convergence assertions.
- [ ] Add mesh-quality, SPD-metric, and finite-output assertions.
- [ ] Repeat mechanics convergence tests over mesh families/refinements.

## 13. Known issues and decision register

1. `DCS` lacks a repository citation and acronym definition.
2. Bending-strain output and internal energy use opposite signs.
3. Free disks contain rigid modes; raw coordinates are not unique outputs.
4. HLBFGS is local and provides no global-optimum guarantee.
5. Effective `basic_disk` relative tolerance is machine epsilon, not `1e-2`.
6. Optimizer failures do not propagate to executable status.
7. Raw code 4 is success without an independent gradient threshold.
8. Maximum-iteration enforcement is off by one.
9. Intermediate VTP growth fields describe final target, not applied stage.
10. Logs append and lack explicit run/stage identifiers.
11. `-s1` defaults to increment 1.0, hence stretch factor 2.0.
12. Orthotropic growth is undefined at the disk origin.
13. No Hessian/stability check distinguishes minima from other stationary points.
14. Orthotropic per-face strain/stress diagnostics are incomplete.
15. Dependencies are unpinned and scripts/paths are historical.
16. The printpath converter is Python 2-era and dependencies are unspecified.
17. The bilayer script uses `-simulateplate`, while code parses `-simulateplane`.

## 14. Recommended review order

Review in dependency order:

1. **Kinematics:** triangle basis, forms, directors, and signs.
2. **Mechanics:** energy, thickness factors, bilayer coupling, and materials.
3. **Input semantics:** define eigenstrain and canonical tensor encoding.
4. **Gauge and boundary conditions:** make deformation targets unique up to intended
   physical branches.
5. **Optimization acceptance:** tolerance, status, stability, and mesh quality.
6. **Branch sampling:** initialization and continuation ensemble design.
7. **Data schema:** provenance, applied metrics, labels, and splits.
8. **Runtime/reproducibility:** dependencies, scripts, and end-to-end tests.

Record accepted decisions and required changes in the table at the top. Mechanics
decisions should precede large-scale neural-operator data generation; optimization
and schema decisions should precede labeling samples as reliable equilibria.
