# English-wheel mechanics: model choices and the curvature objection

**Focused decision review — 21 September 2026.** Seven selected references: three direct English-wheel studies, two adjacent forming studies, and two shell-mechanics foundations. Six primary texts were inspected at the locators below; Huang et al. is abstract-only. This is not a systematic survey or a new numerical validation. Existing reviews and code are preserved.

## 1. Suggested response to the professor

> I agree that the curvature of the already formed sheet can affect the next wheeling pass, and our current prescribed pass law does not resolve that contact/plasticity feedback. However, if “ignores mid-plane curvature” means curvature is absent from the elastic mechanics, that is not the case: the energy uses the current midsurface metric and curvature, and differential layer eigenstrain supplies a bending drive even while the stored reference curvature remains fixed. Therefore, inability to produce complicated shapes does not follow from fixed reference curvature. The more defensible concern is whether our restricted, uncalibrated eigenstrain increments can predict the shapes produced by real wheeling. I would retain the shell as a reduced response model, test current-state-dependent membrane and bending increments, and escalate to contact plasticity if those tests fail.

**Three different interpretations give three different answers:**

| Interpretation of the objection | Assessment |
|---|---|
| The current midsurface curvature is absent from elastic energy. | **Contradicted by inspected code.** Current curvature is evaluated and contributes to bending and layer coupling. |
| There is no independently calibrated evolution of plastic bending/natural curvature, or no current-curvature/contact dependence in the pass law. | **A valid limitation.** Layer mismatch already drives bending, but the particular process law may prescribe the wrong membrane–bending relationship and omit state-dependent plasticity. |
| The present runs have not demonstrated reliable, accurate complicated English-wheel shapes. | **Agreed.** Representability is not reachability, numerical convergence, stable-branch identification, or experimental validation. |

The recent HLBFGS normalization issue is a **separate numerical issue**. Neither its discovery nor its correction establishes or refutes physical model validity. Different computed shapes at the same target metrics do not yet certify distinct stable physical branches.

## 2. Four meanings of curvature/plastic bending that must remain separate

### A. Current midsurface curvature: already present

The current geometry supplies `a(X)` and `b(X,phi)`. The shape operator is `a⁻¹b`; the entries of `b` in a triangle basis are not themselves coordinate-independent principal curvatures.

Inspected implementation:

- [`EnergyHelper_Parametric.hpp`](../src/libshell/EnergyHelper_Parametric.hpp), lines **74–76**: stretching strain uses `abar⁻¹ a − I`.
- Lines **113–115**: bending strain uses `abar⁻¹ (b − bbar)`.
- Lines **320–355, 431–437, 469–479**: stretching–bending cross terms, opposite signs for the two layers, and nonzero bending-energy weights.

Thus changing current vertices/directors changes the energy through both actual forms. A fixed `bbar` does **not** impose `b=0` or freeze the solved curvature. This is geometrically nonlinear elastic response, not an elastoplastic flow rule.

### B. Stored reference curvature versus effective natural curvature

`bbar` is a stored reference second form. **Effective natural curvature is the curvature preference of the complete layer-integrated constitutive energy.** They are not interchangeable in this bilayer formulation.

In the equal-thickness/equal-material, flat-reference reduction of van Rees et al. [R1, Eq. (4)], the two independent layer metrics can be represented by an effective metric and curvature:

\[
\bar a_{\mathrm{eff}}=\tfrac12(\bar a_1+\bar a_2),
\qquad
\bar b_{\mathrm{eff}}=\frac{3}{4h}(\bar a_1-\bar a_2),
\]

where that paper labels layer 1 bottom and layer 2 top and `h` is total thickness. This is a relation in the cited reduced theory, **not a universal finite-plasticity law or a general identity for every curved-reference implementation**. Its decisive implication here is that layer mismatch supplies a curvature preference without a separate update of stored `bbar`.

Adding a curvature increment to `bbar` while retaining the same mismatch can therefore **double-count bending**. Any extra curvature state needs a consistent rederivation or a deliberately nonoverlapping physical interpretation.

### C. Through-thickness plastic strain moment versus actual bending moment

For intuition only, take a homogeneous, symmetric, small-strain elastic section and choose the convention `ε(z)=ε₀+zκ`. The zeroth and first moments of a prescribed inelastic strain field give

\[
\epsilon_0^*=\frac1h\int_{-h/2}^{h/2}\epsilon^p(z)\,dz,
\qquad
\kappa^*=\frac{12}{h^3}\int_{-h/2}^{h/2}z\epsilon^p(z)\,dz.
\]

These are **kinematic eigenstrain moments**, not stresses. The associated eigenmoment is

\[
M^*=\int z\,\mathsf C:\epsilon^p(z)\,dz,
\]

whereas the **actual** stress resultant is `M=∫zσ(z)dz`. For this simple section, `M=D:(κ−κ*)`; unequal moduli/thickness introduce membrane–bending coupling. These explanatory reductions are not a transcription of the code or a finite-strain return-mapping algorithm. In particular, Zhao et al.'s quantity called “bending strain” [R5, Eq. (3)] is a normalized first strain moment, **not a plastic bending moment**.

A two-layer eigenstrain field already retains a coarse first thickness moment. It does not retain an arbitrary plastic-strain profile, self-equilibrated higher-order residual stresses, backstress, or a full unloading/reversal history. Independently evolving membrane and natural-curvature tensors can therefore be a useful **process-state formulation**, but are not automatically a richer geometric representation than two unrestricted tensor-valued layer metrics.

### D. Current-curvature dependence of the next pass: missing from the present law

In [`Sim_Bilayer_Growth.cpp`](../src/simulations/Sim_Bilayer_Growth.cpp), lines **2272–2284**, hits are collected in persistent material coordinates. Lines **2402–2447** use a hit-count hardening factor and prescribed `gtop`, `gbot`, `ortho` to update layer metrics. Lines **2460–2475** check that reference curvature has not drifted. [`GrowthHelper.hpp`](../src/libshell/GrowthHelper.hpp), lines **740–819**, uses a fixed material chart for the increment.

The shell can start the next equilibrium solve from an already curved geometry, but **that is not the same as using curvature/contact to calculate the new plastic increment**. Current-surface mapping for visualization or tangency is likewise not rolling-contact mechanics. No explicit roller contact determines pressure, slip, contact width, or yield in this prescribed law.

A possible extension is [PROPOSAL]

\[
(\Delta\epsilon_0^*,\Delta\kappa^*,\Delta q)
=\mathcal F(\kappa_{\rm current},N,M,q,t,
\text{wheel geometry, force/gap, orientation, speed, history}),
\]

with objective local-frame quantities, measured inputs, and only the variables supported by calibration. Curvature alone need not be a sufficient state: equal shapes can have different residual stress and hardening. Material coordinates themselves are not the problem; prescribing contact-independent hits/increments is the approximation.

## 3. Can a metric/bilayer model make complicated shapes?

**Yes in principle; not every restricted pass schedule can do so, and accuracy is unproven here.**

- **Direct theoretical counterexample to categorical impossibility:** R1, Theorem 1 and Eq. (5), constructs bilayer metrics for a target surface within its thin-shell theory. The assumptions include topologically compatible parametrization, sufficiently small thickness, admissible positive-definite metrics, and independently controllable orthotropic growth in both layers. Its numerical examples include a flower, face and canyon. It also explicitly warns that intermediate growth trajectories can become trapped in metastable configurations (printed p. 11600).
- **Even without layer mismatch:** a spatially incompatible metric can drive out-of-plane bending/buckling while the reference bending preference is flat; Efrati et al. [R2, §3.1 and §4] is a foundational example. Nonzero Gaussian curvature generally requires intrinsic metric change/strain, not merely setting a curvature tensor at will. Actual forms must satisfy surface compatibility.
- **Important restriction in our diagnostics:** `ortho=0` gives equal directional increments locally. Rotating a path changes coverage, not local eigenstrain anisotropy. With top-only growth, average and differential growth are also tied together. This is a much smaller control family than R1's arbitrary two-layer orthotropic fields. Spatially varying isotropic growth can still make nontrivial shapes; R1 is not a proof that the current isotropic zigzags can realize every target.
- **Cylindrical-looking output is not proof of absent curvature mechanics.** It may reflect the chosen metric distribution, membrane/bending competition, insufficient actuation freedom, boundary conditions, or numerical solution selection. Existing project reports identify unresolved numerical sensitivity, not a verified physical explanation for every output.

## 4. Five model choices compared

These are not five mutually exclusive rungs. **Choice 2 concerns state/evolution; choice 3 concerns how a pass updates that state.** Both can often use the existing structural shell after a consistent reformulation.

| Model choice | What it can represent | Main missing physics / strongest objection | Calibration and appropriate role |
|---|---|---|---|
| **1. Prescribed bilayer eigenstrain/metric shell — current baseline** | Nonlinear current geometry, membrane–bending competition, effective bending from layer mismatch, residual incompatibility; complex shapes under sufficiently rich fields [R1–R2]. | Prescribed increments need not equal actual wheeling-induced plasticity. No pressure/slip/yield/contact evolution; two-layer thickness idealization. Current isotropic/top-only recipes restrict controllability. | Calibrate layer increments against more than final height. Retain as an inexpensive response/design model; do not call it a validated wheel model. |
| **2. Independently evolving membrane plastic strain and natural-curvature shell** | Separate zeroth/first inelastic moments, potentially directional hardening and loading/unloading history. Can decouple bending from extension when the present recipe ties them together [R1; adjacent R5]. | Simply storing another tensor supplies no evolution law. A low-order section state may still miss contact pressure and higher-order residual stress. Unrestricted layer metrics already encode comparable geometric degrees of freedom. | Identify both channels from multiple measurements; reparameterize existing bilayer states where possible. Use when measured extension and bending cannot be fitted with one prescribed layer ratio. Never add duplicate mismatch and curvature drives. |
| **3. Curvature/contact-informed reduced pass law** | Makes increments depend on present curvature/orientation, measured force or gap, estimated footprint, and relevant history; retains a cheap global shell. R3 supports shape-aware execution; R5 supports process-to-strain reduction, **not** a validated English-wheel curvature law. | Inter-pass scans alone do not identify plastic state. A local law can miss start/end, edges, crossing tracks, prestress and remote constraints. A curvature-only fit can absorb stress/history confounding. | Preferred next physical refinement. Start with the smallest identifiable state dependence; validate on withheld precurvature and repeated/crossed passes. Output both membrane and bending channels only if observations support them. |
| **4. Elastoplastic shell with moving roller contact** | Contact on the changing geometry, force/friction/slip, through-thickness integration-point plasticity, unloading and springback if modeled; direct numerical precedent R4. | A conventional plane-stress/Kirchhoff–Love shell does not resolve general transverse squeezing or arbitrary thickness gradients. Contact geometry, shell thickness treatment, hardening and tool compliance matter. R4 is preliminary, not a quantitative accuracy benchmark. | Use for process-resolved calibration/diagnosis where shell assumptions survive. Enhanced thickness-stretch/solid-shell formulations may be necessary; R6 is adjacent evidence, not automatic English-wheel validation. Higher setup/run cost than 1–3. |
| **5. 3D solid elastoplastic moving contact** | Finite-thickness compression/shear, nonuniform plastic zones, contact pressure and thickness evolution, with suitable constitutive law and resolution. Useful reference for reduced states. | Not ground truth: expensive thickness/contact refinement, friction/material/compliance identification, springback accuracy and numerical acceleration remain difficult. A poor or aggressively mass-scaled solid solution can be worse than a checked shell [R6]. | Selective short-pass anchors when transverse/contact mechanics are decision-critical or shells systematically miss force/thickness. No verified direct English-wheel solid benchmark was established in this bounded search. Highest expected computational burden; no project runtime claim. |

### What the direct English-wheel evidence actually establishes

- **Rossi & Nicholas [R3]:** physical robotic wheeling, varying tracking patterns, and scans of the changed sheet used to regenerate subsequent motion. This supports the professor's practical concern about handling an already curved sheet. It does **not** establish a constitutive derivative of plastic increment with respect to curvature, or justify resetting the stress-free reference to a scan.
- **Fann [R4]:** direct explicit shell FEA with rigid rollers, friction and imposed impression/motion. This is the closest inspected direct mechanics comparator. The paper uses accelerated motion, acknowledges hourglass-related wrinkling, and does not provide a physical validation or sufficient convergence evidence to become a trusted calibration oracle.
- **Huang et al. [R7]:** inspected abstract establishes robotic trajectory automation and improved trajectory execution/repeatability. “Simulation-based toolpaths” does not identify the simulation's element family; camera-tracked trajectories do not establish current-part shape feedback. Full-text mechanics and quantitative results remain inaccessible.

**Adjacent evidence is informative but conditional.** Zhao et al. [R5] concern concave–convex local rolling of ship plate, not English wheeling. Their baseline is an **S4R shell**, despite the phrase “three-dimensional elastoplastic” in the abstract. Their reduced model transports membrane and bending strain components from short contact calculations into a large-deformation elastic solve; it is not evidence that arbitrary curved/history-dependent passes share one fixed strain kernel. Moser et al. [R6] concern opposed-tool DSIF, with different support, sliding and path conditions. Their useful shell has **thickness stretch and full integration**, not ordinary plane stress; their shell–solid comparison is affected by excessive mass scaling in the solid case. Neither paper supports a universal claim that shells suffice or that solids are always more accurate.

## 5. Recommendation and decision gates

**[INFERENCE / RECOMMENDATION] Retain 1; prioritize a small, falsifiable version of 3; introduce 2 through a consistent state/evolution reformulation when needed. Reserve 4–5 as selective process anchors.**

1. **Do not replace the structural model merely because `bbar` stays fixed.** Its actual-curvature terms and layer coupling already address that objection.
2. **Do not impose top-only or isotropic permanent strain as established wheel physics.** Unequal roller geometry does not, by itself, identify the sign or magnitude of a top–bottom plastic mismatch. Net thinning/extension and plastic bending must be distinguished experimentally.
3. **First gate — missing process state:** if matched passes produce reproducibly different increments on differently curved/prestressed sheets, a constant material-hit law is insufficient. Add the measured state dependencies before assuming a new shell theory is required.
4. **Second gate — insufficient moment freedom:** if one fixed layer ratio cannot jointly fit extension and bending, independently evolve their reduced moments, potentially through the existing two metric tensors. Require a nonredundant mapping to effective natural forms.
5. **Third gate — inadequate shell/contact reduction:** if geometry fits but force, thickness change, or contact maintenance is systematically wrong, test enhanced shell contact or a selective solid anchor. Do not infer this need merely from visually complicated shapes.
6. **Separate numerical and physical gates.** Residual convergence, mesh/path-step sensitivity and branch/stability checks remain necessary, but are not experiments proving wheel-model accuracy. This review executed none of them and revalidated none of the earlier numerical outputs.

A geometry scan/restart should update measured configuration and tool execution, **not erase accumulated natural metrics or residual/plastic history**. Same geometry does not mean same material state.

## 6. Small discriminating experimental design — proposal only

**Aim:** distinguish a fixed per-hit law, independent membrane/bending evolution, and missing current-state/contact dependence without attempting a full production panel. Proposed first stage: **12 matched coupons**, one material batch/thickness, one wheel pair, one nominal normal-force setting and fixed speed/lubrication. Three replicates per condition are a pilot, not a powered validation study.

| Coupons / condition | Procedure and comparison |
|---|---|
| 3 initially flat | Scan and measure after one straight short pass, then after a repeat on the same track. The first-pass data calibrate all candidate reduced laws identically. The repeat is held out to challenge history/saturation. |
| 3 initially flat, crossing history | Same first pass, then an orthogonal crossing pass. Compare the intersection, isolated portions and direction-dependent increments; keep footprints/pass lengths documented. |
| 3 reversibly precurved `+κ`, 3 reversibly precurved `−κ` | Use virgin matched coupons with controlled elastic pre-bending below yield; make the same material-track pass and release the fixture before residual measurements. Measure no-roll fixture loading/unloading first to check reversibility. Compare with the flat first pass. |

**Control and measure:** choose force control **or** gap control, not both as independent fixed inputs; here force is controlled and actual gap is recorded. Record normal/tangential reactions, wheel/fixture motion, actual contact location, initial curvature and material orientation. Obtain registered unloaded full-field scans after each pass, both-surface strain measurements where feasible, and local thickness profiles. Avoid relying only on final height or a global quadratic curvature fit. Keep fixture/edge effects away from the measurement patch and report them.

**Identification caution:** released shape and surface strains are not direct measurements of a stress-free natural metric or a plastic moment. Use paired observables and an explicit inverse mechanics assumption; optional sacrificial release/slitting measurements can constrain residual stress but do not automatically recover the whole thickness profile. Elastic pre-bending also changes prestress and constraints: the `±κ` comparison tests **missing current-state dependence**, not a clean causal effect of curvature alone. Do not replace it with plastically preformed coupons and then attribute all differences to curvature.

**Predeclare decisions relative to measured uncertainty:**

- Reject the constant-pass-law claim if held-out residual strain/curvature increments differ reproducibly across current-state conditions beyond metrology and replicate scatter.
- Reject a fixed membrane-to-bending ratio if it cannot fit both observables across the held-out conditions; then test two independent channels.
- Refine beyond a reduced shape law if it fits shape but repeatedly misses thickness/contact/force behavior. Diagnose pressure/shear versus material/history error before choosing solid elements.
- If the simple law passes, retain it **only over the tested envelope**. Add a second force/gap level or another wheel radius as the next challenge, not an untested extrapolation claim.

No experiment, simulation, training, installation, or new numerical validation was performed for this report.

## 7. Verified references and inspected source locators

`full_text_inspected` below means selected load-bearing primary passages were read, not every page or a reproduction of the work. PDF page indices are physical 1-based pages, distinct from printed pagination.

| ID / reference | Relevance and inspected evidence | Access / limitation |
|---|---|---|
| **R1. van Rees, W. M., Vouga, E., & Mahadevan, L. (2017). [Growth patterns for shape-shifting elastic bilayers](https://doi.org/10.1073/pnas.1709025114). PNAS 114, 11597–11602.** | **Foundation, not rolling.** PDF pp. 1–4; printed 11598–11600, Eqs. (1)–(5), Theorem 1 and metastability discussion. Eqs. (2)–(5)/theorem pages visually checked. Establishes current-curvature energy, bilayer curvature preference and conditional shape capacity. | `full_text_inspected`, local published PDF: [`papers/...bilayers.pdf`](../papers/van-rees-et-al-2017-growth-patterns-for-shape-shifting-elastic-bilayers.pdf). Supplement not inspected; no claim of wheel plasticity or reachable current zigzags. Correct final page is **11602**, not 11604 as in the older review; older file left unchanged. |
| **R2. Efrati, E., Sharon, E., & Kupferman, R. (2009). [Elastic theory of unconstrained non-Euclidean plates](https://doi.org/10.1016/j.jmps.2008.12.004). JMPS 57, 762–775.** | **Foundation, not rolling.** Inspected [arXiv:0810.2411v1](https://arxiv.org/abs/0810.2411): PDF pp. 1, 15, 17, 19: §3.1 comments on stretching/bending energy and midsurface curvature; §4 spherical-metric example. Page 15 visually checked. | `partial_full_text`, local 26-page preprint [`papers/...plates.pdf`](../papers/efrati-sharon-kupferman-2009-elastic-theory-non-euclidean-plates.pdf), **not publisher pagination**. Front page says v1, 14 Oct 2008; generated manuscript footer says 1 Nov 2018. FastTrack returns year 2008; cited 2009 is journal issue year. No inference that the footer identifies a new scientific revision. |
| **R3. Rossi, G., & Nicholas, P. (2018). [Modelling A Complex Fabrication System: New design tools for doubly curved metal surfaces fabricated using the English Wheel](https://papers.cumincad.org/data/works/att/ecaade2018_298.pdf). eCAADe 36, vol. 1, 811–820.** | **Direct experiments / data-driven fabrication.** PDF pp. 3–6 / printed 813–816: empirical modelling choice, setup, feedback and tracking probes. Key PDF p. 5 / printed 815: after forming, scan current sheet and regenerate next-pass path. | `full_text_inspected` at cited passages; leaf checked p. 815 visually, coordinator re-read feedback text. Empirical scans are not a constitutive law. Augmented learning samples are not independent panels; no calibrated raw per-pass mechanics benchmark established. |
| **R4. Fann, K.-J. (2022). [Finite Element Study on Forming Metal Sheets with an English Wheel](https://doi.org/10.1088/1757-899X/1222/1/012006). IOP Conf. Ser.: Materials Science and Engineering 1222, 012006.** | **Direct numerical mechanics.** PDF p. 3 / article p. 2: deformable square shell sheet, rigid shell wheels, friction, imposed impression, explicit LS-DYNA. PDF pp. 5–6 / article pp. 4–5: accelerated motion and numerical-wrinkling caution. | `full_text_inspected`; IOP public PDF retrieved by leaf. Coordinator inspected methods; leaf inspected all article pages and visually checked methods. No physical validation, full constitutive/contact implementation detail or converged quasi-static benchmark established. Do not classify as 3D solid FEA. |
| **R5. Zhao, Y., Hu, C., Dong, H., & Yuan, H. (2017). [Automated local line rolling forming and simplified deformation simulation method for complex curvature plate of ships](https://doi.org/10.5194/ms-8-137-2017). Mechanical Sciences 8, 137–154.** | **Adjacent local line rolling.** PDF pp. 5–7 / printed 141–143, §3: S4R shell/contact, Explicit forming and Standard springback, Eq. (3) strain moments (visually checked). PDF pp. 11–12 / printed 147–148, §4: short-pass strain extraction and elastic initial-strain reduction. | `full_text_inspected` at cited passages, local [`papers/...rolling-forming.pdf`](../papers/copernicus-2017-automated-line-rolling-forming.pdf). “3D elastoplastic” does not mean solid elements here. The normalized bending strain is not a stress resultant. No direct validation of the proposed English-wheel current-curvature pass law. |
| **R6. Moser, N., Pritchet, D., Ren, H., Ehmann, K. F., & Cao, J. (2016). [An Efficient and General Finite Element Model for Double-Sided Incremental Forming](https://doi.org/10.1115/1.4033483). J. Manufacturing Science and Engineering 138, 091007.** | **Adjacent DSIF.** PDF pp. 4–7 / article 091007-4–7, §§3–3.2: direct shell/solid trial, thickness-stretch shell, contact-surface thickness update, mass-scaling confounding, full-integration improvement. | `full_text_inspected` at cited passages; [publisher PDF](https://asmedigitalcollection.asme.org/manufacturingscience/article-pdf/138/9/091007/6404970/manu_138_09_091007.pdf) retrieved. Coordinator re-read methods and limitations. Ordinary plane-stress-shell sufficiency is not established; precise springback geometry was outside that study's scope. |
| **R7. Huang, D., Suarez, D., Kang, P., Ehmann, K., & Cao, J. (2023). [Robot forming: Automated English wheel as an avenue for flexibility and repeatability](https://doi.org/10.1016/j.mfglet.2023.08.104). Manufacturing Letters 35, 342–349.** | **Direct robotic execution experiments.** FastTrack/OpenAlex abstract, bibliographic metadata corroborated by Crossref. Improved trajectory execution, not a verified final-shape mechanics model. | `abstract_only`. Publisher article/PDF returned 403 and text endpoint 400 in leaf retrieval; no usable public PDF found. No access bypass. Full-text control, feedback, FEA and quantitative geometry claims remain unresolved. |

### Search, access and integration boundary

- Existing sources inspected first: [sequential mechanics review](sequential_eigenstrain_mechanics_literature_review.md), [solver/English-wheel report](solver_characteristics_and_english_wheel.md), [bilayer workflow](bilayer_zigzag_workflow.md). Their numerical outputs were not rerun or independently revalidated.
- FastTrack direct-English-wheel and adjacent-forming searches plus DOI lookups were executed; local R1–R2 PDFs supplied foundation checks. No date/open-access/citation-count filter was applied to this focused question. Search responses and exact executed requests are retained in the run evidence.
- Consensus connection failed on secure credential-store writing; **no Consensus search succeeded**. Authentication/configuration was not changed. Recovery requires an explicitly authorized interactive host session, if further discovery is desired.
- Forward-citation sweeps, momentum/saturation analysis and unrelated broad discovery were intentionally omitted: this is a bounded model-choice decision, not a novelty or exhaustive-coverage claim. Henrard et al. (2011), DOI `10.1007/s00466-010-0563-4`, was a metadata-only adjacent lead and is not used as primary evidence.
- Both retrieval leaves returned partial runtime handoffs. Their saved evidence was recovered; the coordinator independently checked decisive primary passages and consolidated this report. The selected comparison is complete within scope despite the explicit R7 access limit. No report rendering or existing-review rewrite was requested.

**Recovery/evidence directory:** `/home/pxl1051/.pi/agent/literature-review-runs/review-TjTE20/` (`report.md`, `evidence.md`, `searches.jsonl`, source extractions and leaf artifacts). The reader-facing entry point is this file; references are inline to avoid creating a competing bibliography catalog.
