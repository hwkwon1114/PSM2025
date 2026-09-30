# Reference library — zigzag bilayer / english-wheel modeling

Papers backing the modeling choices in this codebase, grouped by the claim each
supports. DOIs verified via CrossRef / publisher; PDFs obtained from open-access
or legitimate author/repository copies only.

Sampling, functional-kernel, and neural-operator acquisition papers are
organized separately in [sampling/README.md](sampling/README.md).

## 1. Core framework — incompatible-metric (eigenstrain) elasticity

The simulation prescribes a per-layer target first fundamental form (`abar`) and
lets the elastic energy find the 3D shape. This is the "prescribe a non-Euclidean
metric" principle.

- **Klein, Efrati & Sharon (2007)** — *Shaping of Elastic Sheets by Prescription
  of Non-Euclidean Metrics*, Science 315, 1116. doi:10.1126/science.1135994.
  **PAYWALLED — no PDF here; pull via institutional Science access.** The
  foundational paper for the whole approach.
- `efrati-sharon-kupferman-2009-elastic-theory-non-euclidean-plates.pdf` —
  Efrati, Sharon & Kupferman (2009), *Elastic theory of unconstrained
  non-Euclidean plates*, J. Mech. Phys. Solids 57, 762. arXiv:0810.2411.
  The stretching+bending energy split the Koiter shell uses.
- `efrati-2010-mechanics-non-euclidean-plates-soft-matter.pdf` —
  Efrati (2010), *The mechanics of non-Euclidean plates*, Soft Matter 6, 5693.
  doi:10.1039/c0sm00479k.

## 2. Bilayer growth model — direct parent of this code

- `van-rees-et-al-2017-growth-patterns-for-shape-shifting-elastic-bilayers.pdf` —
  van Rees, Vouga & Mahadevan (2017), PNAS 114, 11597.
  doi:10.1073/pnas.1709025114. Top/bottom metric mismatch -> bending; the model
  this fork (derived from `growth_SM2018`) implements.

## 3. Plastic forming as prescribed eigenstrain — the english-wheel analog

Representing a tool pass' permanent (plastic) strain as a prescribed eigenstrain
field is the "inherent-strain method," standard in welding/line-heating/forming.
This is the process analog to the english wheel: local plastic strain deposited
along a path to curve a plate.

- `mdpi-2024-multipath-inherent-strain-hull-plate.pdf` — multipath inherent-strain
  method, sequential loading of hull plate. J. Marine Sci. Eng. 12, 654.
  doi:10.3390/jmse12040654.
- `copernicus-2017-automated-line-rolling-forming.pdf` — Zhao et al. (2017),
  automated line-rolling forming + simplified deformation simulation for curved
  ship plates. Mech. Sci. 8, 137. doi:10.5194/ms-8-137-2017.
- `rowan-2025-thermoelastic-plate-shot-peen-forming.pdf` — thermoelastic plate
  model for shot-peen forming via effective torque. arXiv:2505.05236.

## 4. Strain hardening (the Voce-type per-pass decay)

The hardening factor q = floor + (1-floor)*exp(-beta*hits) is a Voce saturation
law, repurposed to shrink the deposited eigenstrain increment across passes.

- **Voce (1948)** — *The Relationship Between Stress and Strain for Homogeneous
  Deformation*, J. Inst. Metals 74, 537. **PAYWALLED (1948, not digitized) — no
  PDF here; library scan if the primary source is needed.** Origin of the
  saturation hardening law.
- `bertin-2013-generalized-kocks-mecking-strain-hardening.pdf` — generalized
  Kocks-Mecking dislocation-density hardening law (maps to the Voce form).
  arXiv:1202.6006. Modern open-access citable reference.
- `imim-2024-multipoint-incremental-forming-formability.pdf` — work hardening
  reducing formability in multi-point incremental forming. Arch. Metall. Mater.
  Supports the *qualitative* premise (repeated working deforms less); note it is
  multi-POINT, not strictly multi-PASS.

## Honest status of the backing

- Sections 1-2 are **strongly supported** — the framework and bilayer model are
  established literature.
- Section 3 shows eigenstrain-as-plastic-forming has **real precedent**, though
  those works apply it to single plates (bending) rather than a top/bottom bilayer
  mismatch — a representational difference.
- Section 4: the Voce *form* is textbook, but the specific coupling used here
  (Voce decay of the eigenstrain increment, keyed on integer pass count, applied
  isotropically) is a **phenomenological choice not directly backed by a paper**.
  Its parameters (beta, floor) are uncalibrated and require experimental fitting.
