# From permanent strain to shell shape

## 1. The model in one page

A linear-algebra-first guide to non-Euclidean shell mechanics and English-wheel modeling. The aim is to explain what the model does, what its optimization establishes, and what still needs physical calibration. This is a theory and implementation note, not a report of newly validated forming predictions.

### Two questions, two models

A process model asks: what permanent material change does a wheel pass create? A structural model asks: given that change, which configurations satisfy mechanical equilibrium? Our current simulation prescribes the first and computes the second. It is not yet a resolved wheel-contact/plasticity simulation.

The sheet has a current three-dimensional shape, but material points have two surface coordinates. Each triangle has a 2-by-2 metric and a 2-by-2 curvature form. These are local tensors, not one matrix describing the whole panel.

### Five objects to keep separate

- Original geometry: the initial positions, material coordinates and reference forms. A flat initial geometry has zero curvature, not zero metric.
- Current geometry: vertex positions X and edge-director angles phi. Both change during a solve. Current means the present configuration, before, during or after optimization.
- Preferred material geometry: the layer target metrics. They encode local preferred lengths and angles, which may not fit together into one stress-free surface.
- Elastic mismatch: the difference between the realized geometry and the locally preferred geometry. It drives elastic stresses and energy.
- Process history: permanent increments and any hardening variables. Our prescribed increments are a surrogate for plastic forming, not a calibrated yield-and-flow calculation.

### The recurring calculation

Prescribe a pass -> update the layer target metrics -> hold those targets fixed -> solve for X and phi -> check the residual -> retain geometry and material history for the next pass. Adaptive continuation subdivides a prescribed increment; it is not automatically a simulation of continuous roller motion.

> Main message: current curvature is included in the elastic response. The missing feedback is how the already curved, stressed sheet changes the permanent strain created by the next wheel pass.

Reading route: geometry (pages 2-3), plastic strain and incompatibility (4-5), bilayer mechanics (6-7), optimization (8-9), English-wheel interpretation (10), self-check and sources (11).

## 2. A metric measures geometry, not just stretch

### Start with dot products

Let r(u,v) be the current position of a material point. Its two tangent vectors are the partial derivatives with respect to u and v. Put them into the columns of a 3-by-2 matrix J. The metric is the Gram matrix of those tangent vectors:

$$ J=[\partial_u\mathbf{r},\partial_v\mathbf{r}],\qquad a=J^T J,\qquad ds^2=d\mathbf{u}^T a\,d\mathbf{u}. $$

Here d u is a small two-component coordinate increment; ds is the physical distance it represents. For a nondegenerate surface, a is positive definite: every nonzero coordinate displacement has positive squared length. A zero metric would collapse all local lengths. It does not describe an ordinary flat sheet.

### Worked flat-triangle example

Take perpendicular local edge vectors of length 5 mm. In this edge basis their Gram matrix has entries a11=a22=25 times 10^-6 square metres and a12=0. The sheet is flat and completely ungrown, yet its metric is nonzero. If the first edge stretches by 1%, a11 becomes 25.5025 times 10^-6 square metres. Curvature can still be zero.

$$ a_{11}^{\rm new}=(1.01)^2 a_{11}^{\rm old}. $$

Matrix entries depend on coordinates. With Cartesian material coordinates measured in metres, an initially flat sheet instead has a0=I. With dimensionless coordinates along 5 mm edges, the entries above have units of square metres. These describe the same geometric idea in different bases.

### Three different comparisons

- Current metric a versus initial metric a0: total geometric change from the original sheet; it does not by itself separate elastic and permanent deformation.
- Target metric abar versus a0: prescribed change of locally preferred geometry.
- Current metric a versus target metric abar: elastic mismatch in the shell constitutive model. A layer-specific target is needed in a bilayer.

A one-dimensional analogue makes the split exact: if the preferred length becomes (1+g)L0, but the actual length is L, the elastic stretch is L/[(1+g)L0]. Actual stretch relative to the original is L/L0. They are not the same quantity.

> Equal top and bottom growth is not what makes a flat metric nonzero. Every ordinary flat sheet already has a nonzero metric before growth.

## 3. Current curvature is geometry at any instant

The metric tells us about tangent lengths and angles. Curvature tells us how the surface normal changes as we move across the sheet. For a smooth surface, one sign convention uses the second fundamental form b below; reversing the normal reverses its sign.

$$ b_{ij}=\mathbf{n}\cdot\partial_i\partial_j\mathbf{r},\qquad S=a^{-1}b. $$

The eigenvalues of the shape operator S are the principal curvatures k1 and k2, in inverse length. The mean curvature H and Gaussian curvature K are:

$$ H=\tfrac12(k_1+k_2),\qquad K=k_1 k_2. $$

### A cylinder separates bending from stretching

A flat sheet can wrap around a cylinder without changing distances along its surface, ignoring thickness effects at the midsurface. The following mapping preserves the Cartesian metric a=I:

$$ \mathbf{r}(u,v)=(R\sin(u/R),\ v,\ R[1-\cos(u/R)]). $$

The cylinder has one principal curvature of magnitude 1/R and the other zero, so K=0 even though it bends. A sphere has K=1/R^2. A patch of a flat, inextensible sheet cannot become a smooth spherical patch without changing its metric. This is why stretching and bending must be distinguished.

### Current, stored reference and effective preferred curvature

Current curvature is computed from the present X and phi throughout optimization and after it ends. It is not merely the curvature before an optimization starts. Our discrete shell uses neighboring geometry and edge directors; it does not assign zero curvature to each rendered planar triangle and stop there.

Stored reference curvature bbar is an input to the constitutive energy. Keeping it zero does not force the current b to remain zero. Moreover, in a bilayer, the two target metrics together can supply an effective preferred curvature even if stored bbar is unchanged. Page 6 explains why these quantities must not be conflated.

> Check: a flat sheet and an isometrically rolled cylinder can have the same metric but different curvature. A metric alone is not the current 3D embedding.

## 4. Does plastic strain imply internal stress?

### The simplest constitutive example

For small uniaxial strains, with thermal strain omitted, separate total strain into elastic and plastic parts. Young's modulus is Y here, to avoid confusing it with total energy.

$$ \epsilon=\epsilon^e+\epsilon^p,\qquad \sigma=Y\epsilon^e=Y(\epsilon-\epsilon^p). $$

Plastic strain changes the material's preferred length. Stress depends on the remaining elastic mismatch, not on plastic strain alone. These equations are a teaching example; our finite metric update is not this additive formula applied blindly at large deformation.

### Same permanent strain, different stress

Suppose a bar has a uniform prescribed permanent extension of 0.1%, and Y=70 GPa. If it is free to adopt its new length, total strain equals permanent strain, elastic strain is zero and stress is zero. If held at its original length, elastic strain is -0.1% and the stress predicted by this simple law is -70 MPa, supported by end reactions.

$$ \epsilon^p=0.001:\quad \sigma_{\rm free}=0,\qquad \sigma_{\rm fixed}=-70\ {\rm MPa}. $$

This compares equilibria for a specified eigenstrain; it is not a prediction that a roller necessarily creates that strain or that a particular material remains elastic at every stress used in the example.

### Residual stress without external forces

If different bonded regions acquire incompatible permanent strains, they cannot all achieve their preferred lengths simultaneously. After external loads are removed, elastic stresses may remain and balance internally. Zero net force or moment does not require zero stress at every point. Bending can reduce this mismatch without eliminating it.

Uniform equal-layer expansion of a free homogeneous plate can be stress-free and flat. Spatially varying equal-layer growth is different: it can be incompatible and drive residual stress or buckling even without a top-bottom mismatch.

Plastic flow is irreversible and dissipative. More complete plasticity models may also store hardening energy or backstress. Our current elastic minimization does not calculate that entire dissipation/history process; it solves the elastic response to prescribed permanent metric changes.

> Answer: plastic strain can exist with or without residual stress. Ask whether the body can realize the preferred deformation compatibly under its constraints.

## 5. What non-Euclidean means here

The word does not mean that positions live outside ordinary 3D space. The sheet still occupies Euclidean space. It means the material's preferred intrinsic distances need not describe a stress-free flat sheet, and the preferred metric and curvature may not be jointly realizable as a surface.

### Local wishes need global compatibility

Imagine assigning preferred lengths to every small triangle. Each triangle can have perfectly valid positive lengths. Gluing all triangles together can nevertheless create conflicts. A smooth metric carries an intrinsic Gaussian curvature; the realized metric and curvature must obey surface compatibility equations, commonly called Gauss-Codazzi equations.

A nonflat target metric is not automatically impossible to embed in 3D: a spherical metric is locally realizable by a sphere. Frustration arises when the preferred forms, boundaries, topology or available embedding are incompatible. For example, a spherical preferred metric combined with a preference for zero bending cannot generally satisfy both preferences exactly.

### What minimization balances

For intuition, a reduced shell energy penalizes departures from a preferred metric and a preferred curvature. The symbols below are schematic positive quadratic forms, not the exact implemented bilayer energy.

$$ \mathcal{E}\sim h\int Q_s(a-\bar a)\,dA_0+h^3\int Q_b(b-\bar b)\,dA_0. $$

Here h is thickness; Qs and Qb include material stiffness and the appropriate coordinate contractions. The h versus h^3 scaling explains why a thin sheet often bends readily while resisting stretching. Zero stretching energy need not imply zero bending energy, or vice versa.

### The term non-Euclidean optimization

In this project, we optimize a discrete shell energy with non-Euclidean preferred geometry. This does not mean HLBFGS is a Riemannian optimization algorithm on the space of metrics. The optimization variables are ordinary coordinate/director degrees of freedom, subject to the chosen constraints.

Nonconvexity follows from geometry and the energy landscape: multiple stationary configurations may exist. Different loading histories can select different numerical paths through that landscape. That alone is not proof of physically stable path dependence.

> Better terminology for discussion: nonlinear equilibrium optimization of a shell with incompatible prescribed natural geometry.

## 6. Why a bilayer bends

Your beam analogy is useful: two bonded layers want different lengths but share a geometry. Bending lets one side become longer than the other. It is more precise to say that the mismatch creates a preferred bending deformation and elastic stress resultants than to assign separate unexplained moments to each free layer.

### Derive the beam intuition from a thickness profile

Take a homogeneous symmetric section with thickness coordinate z and convention epsilon(z)=epsilon0+z*kappa. For a prescribed inelastic strain profile epsilon-star(z), small-strain section integration gives:

$$ \epsilon_0^*=\frac{1}{h}\int_{-h/2}^{h/2}\epsilon^*(z)\,dz,\qquad \kappa^*=\frac{12}{h^3}\int_{-h/2}^{h/2}z\epsilon^*(z)\,dz. $$

The average strain drives extension. The first thickness moment drives preferred bending. These are strain measures, not stress moments. For a free uniform beam section, minimizing its quadratic elastic energy selects these preferred generalized strains. The actual bending moment is proportional to kappa minus kappa-star.

For piecewise constant top/bottom strains and equal half-thickness layers:

$$ \epsilon_0^*=\tfrac12(\epsilon_t^*+\epsilon_b^*),\qquad \kappa^*=\frac{3}{2h}(\epsilon_t^*-\epsilon_b^*). $$

Even this bending need not remove all through-thickness residual stress: a linear strain profile cannot exactly match a discontinuous two-layer eigenstrain profile. This is a small-strain beam illustration, not a complete shell or rolling derivation.

### The metric version

In the equal-material/equal-thickness bilayer reduction of van Rees et al. [1], with their bottom/top ordering and normal convention:

$$ \bar a_{\rm eff}=\tfrac12(\bar a_b+\bar a_t),\qquad \bar b_{\rm eff}=\frac{3}{4h}(\bar a_b-\bar a_t). $$

The sign convention for b differs from the beam kappa convention above; compare curvature magnitudes only after matching conventions. The effective bbar here is not the unchanged stored reference bbar in our direct bilayer implementation. Layer mismatch already supplies the bending drive through the energy.

> Do not add the same preferred curvature a second time by resetting stored bbar. That risks double-counting the bilayer mismatch.

## 7. What our implemented energy and pass update do

For each layer, the analytic shell operator forms the following strain-like matrices from the current metric a and curvature b. The symbols A and B here name constitutive ingredients; they are not themselves the complete elastic energy or a universal plastic-strain definition.

$$ A_\ell=\bar a_\ell^{-1}a-I,\qquad B_\ell=\bar a_\ell^{-1}(b-\bar b_{\rm ref}). $$

The bilayer energy includes quadratic stretching and bending contributions and layer-signed stretching-bending coupling. This coupling is important: writing only two independent metric penalties and a zero-preference bending penalty would not explain the implemented mismatch-driven curvature. See EnergyHelper_Parametric.hpp, StrainData_Stretching, StrainData_Bending and the bilayer energy terms.

### The prescribed process surrogate

For each material-space hit, we prescribe two directional growth increments. The history factor q is one when hardening is disabled. The parameter eta is the input called ortho, not a measured universal material constant.

$$ g_1=qg(1+\eta),\qquad g_2=qg(1-\eta). $$

A local growth tensor G is formed by rotating the directional stretch factors (1+g1,1+g2) into the material frame. If Dm contains the fixed material edge basis, the update is:

$$ T=D_m^{-1}GD_m,\qquad \bar a_{\rm new}=T^T\bar a_{\rm old}T. $$

For isotropic growth this multiplies the target metric by (1+qg)^2; when hardening is off, q=1. Top and bottom have separate updates. This is a basis-aware prescribed metric increment, not an elastoplastic return-mapping algorithm based on contact stresses.

### Important restrictions of the diagnostic recipes

- eta=0 gives equal directional increments. Rotating the path changes where material is treated, not the isotropy of each local increment.
- Top-only growth ties average expansion and layer mismatch together. More general independent layer tensors offer more control, but their physical meaning must be calibrated.
- Hits are selected in fixed material coordinates. Current curvature does not determine pressure, footprint or plastic yielding in this law.
- Keeping the previous X and phi provides a warm start. It does not turn the current stressed geometry into a new stress-free reference.

> The shell energy includes current curvature. The imposed growth law is the uncalibrated bridge to wheeling physics.

## 8. What the optimizer actually changes

Let x concatenate all vertex coordinates and edge-director angles. With Nv vertices and Ne edges, the unconstrained state has 3Nv+Ne entries. Directors help the discrete shell represent bending; retaining vertex positions alone does not preserve the complete state.

$$ \mathbf{x}=(\mathbf{X},\boldsymbol{\phi}),\qquad \min_{\mathbf{x}\in\mathcal{C}}\mathcal{E}(\mathbf{x};\bar a_t,\bar a_b,\bar b_{\rm ref}). $$

The feasible set C represents imposed boundary conditions and any gauge constraints. During one elastic solve, material targets, thickness, constitutive parameters, topology and material coordinates remain fixed. The optimizer changes X and phi. The next pass changes the material targets outside this solve.

### First-order equilibrium is not a minimum certificate

At an unconstrained stationary point, the energy gradient is zero. With constraints, the relevant condition is stationarity with respect to admissible variations: a free/projected gradient, or the appropriate constrained equilibrium conditions. Forces at fixed degrees of freedom can be reaction forces and need not vanish.

A small gradient means a small first-order residual in the chosen norm and units. It does not guarantee a nearby local minimum. A maximum, saddle or flat non-minimum point can have zero gradient. Even the distance to a stationary point requires additional conditioning assumptions.

::energy_diagram

These analytic sketches are illustrations, not solver data. At the origin, all three derivatives vanish. Only the first is a strict minimum. A two-variable saddle can likewise have zero gradient with descent directions in some orientations.

### Which algorithm is actually used?

The investigated runs use HLBFGS, which builds a limited-memory approximation from gradient information. Its production gradient is analytic. TinyAD can provide automatic derivatives and an exact discrete Hessian for other solver or diagnostic paths; it is not the Hessian driving these HLBFGS runs.

> Nonconvex optimization does not guarantee even a local minimum merely because the algorithm terminates or the gradient becomes small.

## 9. Hessians, stability and stopping criteria

Near a stationary point, a Taylor expansion explains why the Hessian matters. Here d is an admissible small perturbation, g the gradient and H the Hessian of the discrete energy.

$$ \mathcal{E}(\mathbf{x}+\mathbf{d})=\mathcal{E}(\mathbf{x})+\mathbf{g}^T\mathbf{d}+\tfrac12\mathbf{d}^TH\mathbf{d}+o(\|\mathbf{d}\|^2). $$

### What a second-order check can establish

- A negative eigenvalue in an admissible direction indicates a direction of decreasing energy at an exact stationary point: it is not a local minimum.
- Positive definiteness on all non-rigid admissible directions, at a stationary point of a smooth energy, is sufficient for a strict local minimum modulo rigid motions.
- Positive semidefiniteness alone is inconclusive when zero modes remain; higher-order terms can matter. Free translations and rotations must be handled properly.
- Numerical residuals and eigenvalue errors require tolerances. A stable minimum of the discrete model is not automatically the correct continuum or experimental solution.

TinyAD supplies derivatives, not a theorem that our solve landed at a minimum. Its Hessian still needs to be evaluated and checked on the correct constrained space. Even positive-definite approximate/projected Hessians used to generate steps do not certify the true energy Hessian. Our recent runs disabled final stability certification.

### Why the stopping-rule audit mattered

The old HLBFGS stopping test and the sequence acceptance gate used different quantities:

$$ \frac{\|\mathbf{g}\|}{\max(1,\|\mathbf{x}\|)}\leq\tau_{\rm rel},\qquad \|\mathbf{g}\|\leq\tau_{\rm gate}. $$

These are not interchangeable. The coordinate/director norm changes with the discretization and state. It is not a physically invariant nondimensionalization. The opt-in correction disables relative stopping, uses an absolute internal target below the gate, and recomputes the returned analytic gradient. It does not remove ill-conditioning, prove mesh convergence or calibrate the physical model.

### Four separate checks

Residual convergence -> local stability -> discretization/continuation sensitivity -> agreement with experiments. Each answers a different question. Passing one does not imply passing the others. Absolute residual norms themselves depend on units, variable scaling and mesh; a fully calibrated accuracy criterion needs more than a fixed number.

## 10. What to tell a professor about English wheeling

### A precise description of our method

We prescribe changes in the preferred in-plane metrics of two bonded layers, then solve a nonlinear shell equilibrium problem. Actual midsurface stretching and curvature enter the energy. Layer mismatch supplies an effective bending drive. The method is an eigenstrain response model, not yet a resolved roller-contact plasticity model.

### Which curvature objection is correct?

If the objection is that current midsurface curvature is absent from the elastic energy, that is not true of this implementation. If it is that existing curvature and prestress should change what the next wheel pass does, that is a real limitation: we do not resolve that contact/plasticity feedback.

A suitable reduced process law would predict permanent membrane and bending increments from some calibrated subset of current curvature, stress resultants, hardening, thickness, tool geometry, force/gap and direction. Curvature alone may not determine the response: identical shapes can carry different residual stresses and histories.

### Why complicated shapes are a separate question

Bilayer target metrics can encode complex shapes in the theory of [1], under its thinness, admissibility and controllability assumptions. That does not prove our isotropic top-only zigzags can reach every target, or that an optimizer finds a stable reachable shape. Representability, process reachability and predictive accuracy are distinct.

### A sensible modeling ladder

1. Keep the present shell as a response model; calibrate its prescribed increments over a stated operating range.
2. Add current-state/contact dependence to the per-pass law, and allow independent extension/bending channels when identifiable. These may use the existing two layer metrics rather than require a new shell.
3. Use elastoplastic shell contact for selected process-resolved checks, with suitable through-thickness behavior.
4. Use solid contact elements where transverse squeezing, shear or thickness plasticity cannot be represented adequately by the shell reduction.

More detailed FEA is not automatic ground truth. Constitutive calibration, contact/friction, mesh resolution, unloading and numerical acceleration all require checks. See [3] for a direct English-wheel shell-contact example and the focused model-choice review for its limitations.

> Suggested summary: our model includes curvature in the elastic mechanics, but the permanent-strain prescription does not yet account for curvature-dependent wheel contact. That is the physical modeling gap we should test.

## 11. Self-check, answer key and source map

### Try these before reading the answers

1. A sheet is flat, ungrown and stress-free. Is its metric zero? What about its curvature?
2. A free bar has uniform permanent extension. Must it carry stress? What changes if its ends are fixed?
3. Can a sheet bend without midsurface stretching? Can it become a smooth sphere by the same mechanism?
4. Can stored reference curvature remain zero while a bilayer bends? What supplies the drive?
5. Does a small gradient plus access to a TinyAD Hessian guarantee a local minimum?
6. Does warm-starting the next pass make the growth law depend on current curvature?

### Short answer key

1. Its metric is nonzero and positive definite in a regular coordinate chart; its curvature is zero. The metric measures geometry, not strain alone.
2. No: it can adopt its preferred length with zero elastic strain. Constraints can prevent that and create elastic stress. Nonuniform incompatible permanent strain can also cause self-balanced residual stress after unloading.
3. Yes, a plane can roll isometrically into a cylinder. A sphere has nonzero Gaussian curvature and requires a metric change from a flat patch.
4. Yes. Differential layer target metrics enter the coupled bilayer energy and create an effective preferred curvature.
5. No. Stationarity is only a first-order condition; the true Hessian must be assessed on admissible non-rigid directions. Access to a Hessian is not a performed stability test.
6. No. It changes the initial configuration for equilibrium optimization. A current-state-dependent process law is an additional modeling ingredient.

### Sources and qualifications

[1] van Rees, Vouga & Mahadevan (2017), Growth patterns for shape-shifting elastic bilayers, PNAS 114, 11597-11602. doi:10.1073/pnas.1709025114. Effective target forms: Eq. (4); conditional target-shape construction: Theorem 1. Not an English-wheel constitutive model.

[2] Efrati, Sharon & Kupferman (2009), Elastic theory of unconstrained non-Euclidean plates, JMPS 57, 762-775. doi:10.1016/j.jmps.2008.12.004. Foundation for incompatible preferred metrics and elastic response.

[3] Fann (2022), Finite Element Study on Forming Metal Sheets with an English Wheel, IOP MSE 1222, 012006. doi:10.1088/1757-899X/1222/1/012006. Direct shell/contact example, not a validated benchmark for our model.

Implementation map: src/libshell/EnergyHelper_Parametric.hpp (energy); GrowthHelper.hpp (metric updates); ExtendedTriangleInfo.hpp (forms); src/simulations/Sim_Bilayer_Growth.cpp (sequence); Sim.hpp and HLBFGS_Wrapper.hpp (stopping/acceptance). Companion notes: ../bilayer_zigzag_workflow.md, english_wheel_model_choices.md and absolute_gradient_stopping_audit.md. Beam and bar examples here are explanatory reductions, not simulation results. Existing literature evidence was reused; no new literature search or forming simulation was performed for this teaching note.
