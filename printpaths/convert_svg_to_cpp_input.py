"""
Convert SVG print paths into the numeric format SVGHelper::SVGParser::parseFile reads.

Output format (whitespace separated):

    nPaths
    for each path:
        nSegs
        for each segment:
            tStart tEnd
            segmentType            1 line, 2 quadratic, 3 cubic, 4 arc
            control points         line: P0 P1 | quad: P0 P1 P2 | cubic: P0..P3
                                   arc:  P0 P1 radius rotation large_arc sweep

Usage:
    python convert_svg_to_cpp_input.py [--flipy] file1.svg [file2.svg ...]

--flipy negates the y coordinate. The original script hardcoded this to True with the
note "True for orchid and folding flower"; it is now an explicit flag defaulting to off,
so the other paths (catenoid, helicoid, sombrero, logspiral) convert correctly without
editing the source.

Requires svgpathtools.
"""

import sys

import numpy as np
from svgpathtools import Arc, CubicBezier, Line, QuadraticBezier, svg2paths


def write_point(fout, point, flip_y, first=False):
    if not first:
        fout.write("\t")
    y = -np.imag(point) if flip_y else np.imag(point)
    fout.write("%10.10e %10.10e" % (np.real(point), y))


def convert(fname_in, flip_y):
    fname_out = fname_in.replace(".svg", ".dat")
    print("converting %s into %s" % (fname_in, fname_out))

    paths, _attributes = svg2paths(fname_in)

    # Paths that carry no usable geometry must be skipped BEFORE the count is written,
    # otherwise nPaths disagrees with the number of path blocks that follow and the C++
    # reader walks off into the next path's data.
    usable = []
    for i, path in enumerate(paths):
        if len(path) == 0:
            continue
        try:
            path.length()
        except Exception:
            continue
        usable.append(path)

    if len(usable) != len(paths):
        print("  skipped %d of %d paths (empty or zero length)"
              % (len(paths) - len(usable), len(paths)))

    with open(fname_out, "w") as fout:
        fout.write("%d\n" % len(usable))
        for i, path in enumerate(usable):
            n_segs = len(path)
            print("  path %d : length %10.10e, %d segments" % (i, path.length(), n_segs))
            fout.write("%d\n" % n_segs)

            for s, seg in enumerate(path):
                t_start = path.t2T(s, 0)
                t_end = path.t2T(s, 1)
                fout.write("%10.10e \t %10.10e\n" % (t_start, t_end))

                if isinstance(seg, Line):
                    fout.write("1\n")
                    write_point(fout, seg.start, flip_y, True)
                    write_point(fout, seg.end, flip_y)
                elif isinstance(seg, QuadraticBezier):
                    fout.write("2\n")
                    write_point(fout, seg.start, flip_y, True)
                    write_point(fout, seg.control, flip_y)
                    write_point(fout, seg.end, flip_y)
                elif isinstance(seg, CubicBezier):
                    fout.write("3\n")
                    write_point(fout, seg.start, flip_y, True)
                    write_point(fout, seg.control1, flip_y)
                    write_point(fout, seg.control2, flip_y)
                    write_point(fout, seg.end, flip_y)
                elif isinstance(seg, Arc):
                    fout.write("4\n")
                    write_point(fout, seg.start, flip_y, True)
                    write_point(fout, seg.end, flip_y)
                    write_point(fout, seg.radius, flip_y)
                    fout.write("\t %10.10e %d %d\n"
                               % (seg.rotation, seg.large_arc, seg.sweep))
                else:
                    raise TypeError("unknown segment type %r in %s (path %d, segment %d)"
                                    % (type(seg), fname_in, i, s))
                fout.write("\n")

    return fname_out


def main():
    args = sys.argv[1:]
    flip_y = "--flipy" in args
    files = [a for a in args if not a.startswith("--")]

    if not files:
        print(__doc__)
        sys.exit(1)

    for fname in files:
        convert(fname, flip_y)


if __name__ == "__main__":
    main()
