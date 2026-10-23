# Final Overlapping Perpendicular Toolpaths A and B (N=18)

Final frozen layout for the overlapping perpendicular toolpath investigation on the 200 mm x 300 mm sheet:

- **Toolpath A** (Blue): Centered at (0, 0), rotation = 0 deg. 10 vertical strips, Lv = 140 mm, pitch = 22.5 mm, angle alpha = 9.13 deg, width = 8 mm. Center-peak profile. Footprint: u in [-94, 94] mm, v in [-74, 74] mm.
- **Toolpath B** (Vermilion): Centered at (0, 0), rotation = 90 deg. 18 horizontal strips, Lv = 196 mm, pitch = 9.79 mm, angle alpha = 2.86 deg, width = 4 mm. Uniform profile. Footprint: u in [-100, 100] mm, v in [-80.3, 80.3] mm.

Key properties:
- **180 Crossing Nodes**: 10 vertical strips of A x 18 horizontal strips of B.
- **100% Coverage of Toolpath A**: Toolpath B completely covers Toolpath A across its entire 140 mm vertical length, with a 6.3 mm margin extending slightly past Toolpath A's turnarounds.
- **Shared Material Coupling**: Voce strain-hardening decay directly couples the two toolpaths at all 180 intersections, creating pronounced sequence-dependent final target metrics and shapes.
