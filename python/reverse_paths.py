"""
Reverse the point order of every path in a toolpath .dat file.

This is a null test for the print-path -> per-face field projection. Reversing a path
traverses exactly the same geometry in the opposite direction, so the filament density
and the filament ORIENTATION AS A LINE FIELD (t (x) t) are physically unchanged. Any
difference in the resulting per-face target metrics -- or in the final shape -- means the
direction interpolation depends on traversal order, i.e. it is averaging signed tangent
vectors rather than the nematic tensor.

Only line segments (type 1) are handled, which is what the catenoid paths contain; the
script refuses anything else rather than silently mis-writing curved segments.

Usage:
    python python/reverse_paths.py in.dat out.dat
"""

import sys

NPTS = {1: 2, 2: 3, 3: 4}   # control points per segment type


def read_paths(fname):
    t = open(fname).read().split()
    i = 0
    n_paths = int(t[i]); i += 1
    paths = []
    for _ in range(n_paths):
        n_segs = int(t[i]); i += 1
        segs = []
        for _ in range(n_segs):
            t0, t1 = float(t[i]), float(t[i + 1]); i += 2
            styp = int(t[i]); i += 1
            if styp not in NPTS:
                raise SystemExit("segment type %d not supported by this null test" % styp)
            pts = []
            for _ in range(NPTS[styp]):
                pts.append((float(t[i]), float(t[i + 1]))); i += 2
            segs.append((t0, t1, styp, pts))
        paths.append(segs)
    if i != len(t):
        raise SystemExit("trailing tokens: parsed %d of %d" % (i, len(t)))
    return paths


def reverse_path(segs):
    """Traverse the same geometry backwards: reverse segment order, reverse the control
    points within each segment, and remap the parameter as t -> 1 - t."""
    out = []
    for (t0, t1, styp, pts) in reversed(segs):
        out.append((1.0 - t1, 1.0 - t0, styp, list(reversed(pts))))
    return out


def write_paths(fname, paths):
    with open(fname, "w") as f:
        f.write("%d\n" % len(paths))
        for segs in paths:
            f.write("%d\n" % len(segs))
            for (t0, t1, styp, pts) in segs:
                f.write("%10.10e \t %10.10e\n" % (t0, t1))
                f.write("%d\n" % styp)
                f.write("\t".join("%10.10e %10.10e" % p for p in pts))
                f.write("\n\n")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    paths = read_paths(sys.argv[1])
    rev = [reverse_path(s) for s in paths]

    # the endpoints of the whole path must simply swap
    a0 = paths[0][0][3][0]
    b1 = rev[0][-1][3][-1]
    assert abs(a0[0] - b1[0]) < 1e-9 and abs(a0[1] - b1[1]) < 1e-9, "endpoint check failed"

    write_paths(sys.argv[2], rev)
    print("%s -> %s : %d paths, %d segments, order reversed"
          % (sys.argv[1], sys.argv[2], len(rev), sum(len(s) for s in rev)))


if __name__ == "__main__":
    main()
