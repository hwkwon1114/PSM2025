# Overlapping Perpendicular Toolpaths A and B

Design layout for the overlapping perpendicular toolpath investigation on the 200 mm x 300 mm rectangular sheet (u in [-100, 100] mm, v in [-150, 150] mm).

- **Toolpath A** (Blue): Centered at (0, 0), rotation = 0 deg. 10 vertical strips, strip length Lv = 140 mm, pitch = 22.5 mm, angle alpha = 9.13 deg, width = 8 mm. Center-peak profile (top_end_ratio = 0.4, power = 2.0). Bounding box: u in [-94, 94] mm, v in [-74, 74] mm.
- **Toolpath B** (Vermilion): Centered at (0, 0) [shifted from previous +118 mm offset], rotation = 90 deg. 8 horizontal strips, strip length Lv = 196 mm, pitch = 9.79 mm, angle alpha = 2.86 deg, width = 4 mm. Uniform profile. Bounding box: u in [-100, 100] mm, v in [-31.38, 31.38] mm.

The two toolpaths form an orthogonal cross grid in the central region:
- Overlap bounding box: u in [-94, 94] mm, v in [-31.38, 31.38] mm (188 mm x 62.75 mm, ~19.7% of total sheet area).
- Number of intersections: 80 orthogonal crossing points (10 vertical strips of A x 8 horizontal strips of B).
- In this shared region, mesh elements experience plastic hits from both toolpaths. Under the Voce hardening model (beta = 0.22314), the sequence of application directly alters the accumulated plastic hit count and growth attenuation, creating genuine constitutive sequence dependence between A->B and B->A.
