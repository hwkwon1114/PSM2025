"""
Generate zig-zag print paths in the format Sim_Bilayer_4DFilaments actually consumes.

SVGHelper::SVGParser::parseFile does NOT parse SVG XML despite the name -- it reads a
plain whitespace-separated numeric format:

    nPaths
    for each path:
        nSegs
        for each segment:
            tStart tEnd segmentType   then the control points
                type 1 = line     : P0x P0y P1x P1y
                type 2 = quadratic: P0 P1 P2
                type 3 = cubic    : P0 P1 P2 P3
                type 4 = arc      : P0 P1 rx ry rotation large_arc sweep

with tStart/tEnd a contiguous parameterization running 0 -> 1 across the whole path.
The repo ships only the source .svg files, not the converted .dat files, so this script
produces usable input for a bilayer run.

Usage:
    python python/make_zigzag_path.py out_bot.dat out_top.dat
"""

import sys


def zigzag_segments(along, spacing, span_lo, span_hi, cross_lo, cross_hi):
    """
    Serpentine path: passes parallel to `along` ('x' or 'y'), stepping by `spacing`
    across the perpendicular direction, joined by short turnaround connectors.
    Returns a list of ((x0, y0), (x1, y1)) line segments.
    """
    segs = []
    n = int((cross_hi - cross_lo) / spacing) + 1
    forward = True
    for i in range(n):
        c = cross_lo + i * spacing
        if c > cross_hi:
            break
        a0, a1 = (span_lo, span_hi) if forward else (span_hi, span_lo)

        if along == "y":
            segs.append(((c, a0), (c, a1)))
        else:
            segs.append(((a0, c), (a1, c)))

        # turnaround into the next pass
        if i < n - 1:
            c_next = c + spacing
            if c_next <= cross_hi:
                if along == "y":
                    segs.append(((c, a1), (c_next, a1)))
                else:
                    segs.append(((a1, c), (a1, c_next)))
        forward = not forward
    return segs


def write_path_file(filename, segments):
    """Write one path made of line segments, parameterized by cumulative arc length."""
    lengths = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segments]
    total = sum(lengths)

    lines = ["1", str(len(segments))]
    t = 0.0
    for i, ((a, b), L) in enumerate(zip(segments, lengths)):
        t_start = t
        # force the final tEnd to be exactly 1 so the parser's contiguity check holds
        t_end = 1.0 if i == len(segments) - 1 else t + L / total
        lines.append(
            f"{t_start:.12f} {t_end:.12f} 1 "
            f"{a[0]:.12f} {a[1]:.12f} {b[0]:.12f} {b[1]:.12f}"
        )
        t = t_end

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    return total, len(segments)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    out_bot, out_top = sys.argv[1], sys.argv[2]
    spacing = 0.75          # matches max_filament_spacing in run_catenoid_printpath
    x_lo, x_hi = -4.0, 4.0
    y_lo, y_hi = -20.0, 20.0

    # bottom layer: filaments running along y ; top layer: along x.
    # Orthogonal alignment between layers is what produces the bending mismatch.
    bot = zigzag_segments("y", spacing, y_lo, y_hi, x_lo, x_hi)
    top = zigzag_segments("x", spacing, x_lo, x_hi, y_lo, y_hi)

    for name, segs in ((out_bot, bot), (out_top, top)):
        total, n = write_path_file(name, segs)
        print(f"{name}: {n} segments, total length {total:.2f}")


if __name__ == "__main__":
    main()
