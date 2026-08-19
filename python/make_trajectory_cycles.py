#!/usr/bin/env python3
"""Generate trajectory-following (progressive-prefix) zigzag cycle files.

Instead of growing the whole zigzag pattern uniformly (cont30-style), activate
one strip at a time in traversal order -- mimicking the english wheel rolling
along the toolpath. Strip k's total growth G is split into m sub-increments
(continuation within the dwell), giving n_strips*m cycles in total.

Growth parity with the cont30 baseline: G = 30 * 0.009806 = 0.29418 per strip.
"""
import argparse, json, sys

def make_cycles(n_strips, g_total, m, lv_mm, alpha_deg, width_mm, order):
    toolpaths = []
    for k in order:
        gtop = [0.0] * n_strips
        gtop[k] = g_total / m
        toolpaths.append({
            "id": f"strip_{k:02d}",
            "enabled": True,
            "repeat": m,
            "operation": {
                "type": "zigzag",
                "lv_mm": lv_mm,
                "alpha_deg": alpha_deg,
                "n_strips": n_strips,
                "width_mm": width_mm,
                "gtop": gtop,
                "gbot": 0.0,
                "ortho": 0.0,
            },
        })
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": {"mode": "uniform"}},
        "toolpaths": toolpaths,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--n-strips", type=int, default=10)
    p.add_argument("--g-total", type=float, default=0.29418,
                   help="total growth per strip (default: cont30 parity)")
    p.add_argument("--increments", type=int, default=3,
                   help="sub-increments per strip dwell")
    p.add_argument("--lv-mm", type=float, default=120.0)
    p.add_argument("--alpha-deg", type=float, default=5.0)
    p.add_argument("--width-mm", type=float, default=16.0)
    p.add_argument("--reverse", action="store_true",
                   help="traverse strips in reverse order (order-dependence test)")
    a = p.parse_args()

    order = list(range(a.n_strips))
    if a.reverse:
        order = order[::-1]
    doc = make_cycles(a.n_strips, a.g_total, a.increments,
                      a.lv_mm, a.alpha_deg, a.width_mm, order)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)

    ncyc = a.n_strips * a.increments
    tot = sum(tp["repeat"] * sum(tp["operation"]["gtop"]) for tp in doc["toolpaths"])
    print(f"{a.out}: {len(doc['toolpaths'])} toolpaths x {a.increments} repeats "
          f"= {ncyc} cycles, per-strip total growth {a.g_total:.5f}, "
          f"sum over strips {tot:.5f}, order={'reversed' if a.reverse else 'forward'}")

if __name__ == "__main__":
    sys.exit(main())
