# Toolpaths A and B Layout

Layout of the two perpendicular calibrated zigzag toolpaths on the 200 mm x 300 mm sheet (u in [-100, 100] mm, v in [-150, 150] mm).

- **Toolpath A** (Blue): Centered at (0, 0), rotation = 0 deg. 10 vertical strips, strip length Lv = 140 mm, pitch = 22.5 mm, angle alpha = 9.13 deg, width = 8 mm. Center-peak profile (top_end_ratio = 0.4, power = 2.0). Bounding footprint: u in [-94, 94] mm, v in [-74, 74] mm.
- **Toolpath B** (Vermilion): Offset to (0, +118 mm), rotation = 90 deg. 8 horizontal strips, strip length Lv = 196 mm, pitch = 9.79 mm, angle alpha = 2.86 deg, width = 4 mm. Uniform profile. Bounding footprint: u in [-100, 100] mm, v in [86.6, 149.4] mm.

There is a 12.6 mm vertical clearance gap between Toolpath A and Toolpath B, resulting in zero face-hit overlap between the two toolpaths (5,241 faces for A, 2,858 faces for B, intersection = 0).
