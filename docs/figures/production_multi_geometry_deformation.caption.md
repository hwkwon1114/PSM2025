# Figure 2: Equilibrium Deformations Across Diverse Shell Geometries

Comparative response of the production hybrid equilibrium solver (`-equilibrium_solver hybrid -hessian_threads 4`) across four distinct mechanical and geometric configurations subjected to the single-pass zigzag toolpath:
- **Row 1 (`rect_flat`)**: Planar rectangular structured plate ($0.2\times 0.2$ m). Bending moment induces symmetric double-curvature deflection ($U_z \in [-0.15, +0.23]\;\mu$m; maximum displacement $\|U\| = 6.06\;\mu$m).
- **Row 2 (`rect_curved`)**: Cylindrical shell panel with analytical transverse radius $R = 0.25$ m (initial out-of-plane camber $Z \in [0, 19.7]$ mm). Toolpath induces asymmetric deflection coupling with initial membrane curvature ($U_z \in [-0.12, +0.68]\;\mu$m).
- **Row 3 (`rect_allclamped`)**: Clamped boundary rectangular plate with edge displacement pinning (red markers indicate constrained edge DOFs). Boundary reaction forces restrict boundary deflection, concentrating curvature into the central domain ($U_z \in [-0.15, +0.23]\;\mu$m).
- **Row 4 (`rect_irreg`)**: Unstructured Delaunay triangular mesh (348 vertices, 636 faces) with random triangle orientations. The parallel solver preserves exact bitwise reproducibility without grid-alignment bias (maximum deflection $U_z = 0.38\;\mu$m; maximum displacement $\|U\| = 7.69\;\mu$m).
