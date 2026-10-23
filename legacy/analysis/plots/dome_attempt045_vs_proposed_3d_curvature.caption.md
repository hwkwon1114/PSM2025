# Figure: 3D Equilibrium Surface, Gaussian Curvature Classification, and Toolpath Architecture

**File:** `dome_attempt045_vs_proposed_3d_curvature.png` / `dome_attempt045_vs_proposed_3d_curvature.pdf`  
**Datasets:**
- Attempt 045 Baseline (3 discrete toolpaths, Stage 1 BO best compromise): `run/dome/bo_continuation_stage1/attempt_045/`
- Proposed Pitch-Graded Concentric Crosshatch: `run/dome/proposed_matched_analysis_v1/`

### Description:
A comprehensive 3-row side-by-side spatial comparison evaluating the transition from discrete multi-band toolpaths to a pitch-graded concentric cross-hatch architecture for target synclastic dome formation ($K > 0$).

- **Row 1: 3D Deformed Surface (Elevation $z$ [mm]):**
  - *(Left, Attempt 045):* Exhibits localized buckling near the central intersection with shallow dome height ($z = 4.03$ mm) and flaring boundary edges ($z < 0$).
  - *(Right, Proposed Crosshatch):* Exhibits a prominent, smooth global dome reaching $z = 7.50$ mm (+86% height gain). The out-of-plane dome structure extends coherently across the sheet footprint without edge ripple instabilities.
- **Row 2: Gaussian Curvature Classification Map (Material $uv$ Space):**
  - Mesh faces classified into Synclastic Dome ($K > 0$, Red), Anti-clastic Saddle ($K < 0$, Blue), and Flat/Cylindrical ($K \approx 0$, Gray).
  - *(Left, Attempt 045):* Severe saddle bands (blue, 53.9% saddle flux) cut across the entire width of the specimen, isolating the small central synclastic patch.
  - *(Right, Proposed Crosshatch):* Saddle flux fraction is reduced by 52% (down to 18.2%), producing an expansive, connected synclastic dome core (red) that spans 72.0% of the Central ROI ($|u| \le 60$ mm, $|v| \le 100$ mm).
- **Row 3: Toolpath Footprint Layout on Sheet ($200 \times 300$ mm):**
  - *(Left, Attempt 045):* Three separated zigzag blocks (A at center, B/C at margins) leave unprinted gaps where conflicting uniaxial bending moments induce saddle collapse.
  - *(Right, Proposed Crosshatch):* Concentric, orthogonal terrace capsules ($A$ at 0°, $B$ at 90°) with radial pitch grading outward ($p_i = p_c [1 + (f-1)(i/(n-1))^\alpha]$). Eliminates discrete boundaries and concentrates maximum biaxial eigenstrain at the center with strictly monotone decay.
