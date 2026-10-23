#!/usr/bin/env python3
"""Generate and score bounded zigzag toolpath sequences against target shapes.

This is a coarse inverse-design utility: it searches discrete orientation sequences,
then scores replayed VTP results on material coordinates. It does not alter simulator
mechanics or reference-curvature handling.
"""
import argparse, itertools, json, subprocess
from pathlib import Path
import numpy as np


def target_z(u, v, kind, amplitude_mm, lx, ly):
    x, y = u / lx, v / ly
    amp = 1e-3 * amplitude_mm
    if kind == "dome": return -amp * (x*x + y*y)
    if kind == "saddle": return amp * (x*x - y*y)
    if kind == "twist": return amp * (2.0*x*y)
    raise ValueError(f"unknown target {kind}")


def sequences(angles, length):
    """Small deterministic candidate family; exhaustive only for short sequences."""
    out = set()
    for a in angles:
        out.add(tuple([a] * length))
    for a in angles:
        b = (a + 90.0) % 180.0
        out.add(tuple(a if i % 2 == 0 else b for i in range(length)))
    for start in angles:
        out.add(tuple((start + 30.0*i) % 180.0 for i in range(length)))
    for start in angles:
        out.add(tuple((start - 30.0*i) % 180.0 for i in range(length)))
    return sorted(out)


def make_config(seq, args):
    per_pass = args.total_growth / len(seq)
    paths = []
    for i, angle in enumerate(seq, 1):
        paths.append({
            "id": f"pass_{i:02d}_{angle:g}deg",
            "enabled": True,
            "repeat": 1,
            "operation": {
                "type": "zigzag", "lv_mm": args.lv_mm,
                "alpha_deg": args.alpha_deg, "n_strips": args.n_strips,
                "width_mm": args.width_mm, "center_uv_mm": [args.center_u, args.center_v],
                "rotation_deg": angle, "gtop": per_pass, "gbot": 0.0,
                "ortho": args.ortho,
            },
        })
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": {"mode": "uniform"}},
        "toolpaths": paths,
    }


def write_manifest(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ang = list(np.arange(0.0, 180.0, 30.0))
    allseq = sequences(ang, args.length)
    records = []
    for i, seq in enumerate(allseq[:args.max_candidates]):
        cfg = out / f"candidate_{i:03d}.json"
        cfg.write_text(json.dumps(make_config(seq, args), indent=2) + "\n")
        records.append({"id": i, "config": str(cfg), "angles": seq})
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"args": vars(args), "candidates": records}, indent=2, default=str) + "\n")
    print(f"wrote {len(records)} candidates to {manifest}")
    return manifest


def read_vtp(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update()
    pd = r.GetOutput()
    xyz = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    u = vtk_to_numpy(pd.GetPointData().GetArray("material_u")).astype(float)
    v = vtk_to_numpy(pd.GetPointData().GetArray("material_v")).astype(float)
    return xyz, u, v


def kabsch_align(source, target):
    """Return source aligned to target using row-vector coordinates."""
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    h = source_centered.T.dot(target_centered)
    uu, _, vt = np.linalg.svd(h)
    r = uu.dot(vt)
    if np.linalg.det(r) < 0.0:
        uu[:, -1] *= -1.0
        r = uu.dot(vt)
    return source_centered.dot(r), target_centered


def self_test_kabsch():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    theta = np.deg2rad(37.0)
    rotation = np.array([[np.cos(theta), np.sin(theta), 0.0],
                         [-np.sin(theta), np.cos(theta), 0.0],
                         [0.0, 0.0, 1.0]])
    target = source.dot(rotation) + np.array([4.0, -2.0, 0.7])
    aligned, centered_target = kabsch_align(target, source)
    error = np.sqrt(np.mean(np.sum((aligned - (source - source.mean(axis=0))) ** 2, axis=1)))
    if error > 1e-12:
        raise AssertionError("Kabsch self-test error %.3e" % error)


def score_vtp(path, kind, amp, lx, ly):
    xyz, u, v = read_vtp(path)
    zt = target_z(u, v, kind, amp, lx, ly)
    target = np.column_stack((u, v, zt))
    aligned, target_centered = kabsch_align(xyz, target)
    return float(np.sqrt(np.mean(np.sum((aligned - target_centered)**2, axis=1))) * 1000.0)


def run_candidate(cfg, args, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(args.binary).resolve()), "-sim", "bilayer_growth", "-case", "custom",
           "-geometry", "rectangle", "-lx", str(args.panel_x), "-ly", str(args.panel_y),
           "-res", str(args.resolution), "-h_total", str(args.thickness),
           "-growth_type", "zigzag_sequence", "-cycle_file", str(cfg.resolve()),
           "-enable_passE", "false", "-nsteps", "1", "-tol", str(args.tolerance),
           "-minimizer", "hlbfgs", "-max_iter", str(args.max_iter), "-basename", run_dir.name,
           "-export_stl", "true"]
    with (run_dir / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    vtps = sorted(run_dir.glob("*_final.vtp"))
    if not vtps:
        raise RuntimeError(f"no final VTP in {run_dir}")
    return vtps[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("run/inverse_toolpath_search"))
    p.add_argument("--target", choices=("dome", "saddle", "twist"), default="dome")
    p.add_argument("--target-amplitude-mm", type=float, default=5.0)
    p.add_argument("--length", type=int, default=4)
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--total-growth", type=float, default=0.002)
    p.add_argument("--ortho", type=float, default=0.0)
    p.add_argument("--lv-mm", type=float, default=140.0); p.add_argument("--alpha-deg", type=float, default=9.13)
    p.add_argument("--n-strips", type=int, default=10); p.add_argument("--width-mm", type=float, default=10.0)
    p.add_argument("--center-u", type=float, default=0.0); p.add_argument("--center-v", type=float, default=0.0)
    p.add_argument("--panel-x", type=float, default=0.13); p.add_argument("--panel-y", type=float, default=0.16)
    p.add_argument("--thickness", type=float, default=0.0005); p.add_argument("--resolution", type=float, default=0.03)
    p.add_argument("--tolerance", type=float, default=1e-12); p.add_argument("--max-iter", type=int, default=50000)
    p.add_argument("--binary", default="bin/shell"); p.add_argument("--run", action="store_true")
    p.add_argument("--score-vtp", type=Path)
    args = p.parse_args()
    manifest = write_manifest(args)
    if args.score_vtp:
        print(f"target_error_mm={score_vtp(args.score_vtp, args.target, args.target_amplitude_mm, args.panel_x, args.panel_y):.6f}")
    if args.run:
        rows = []
        for rec in json.loads(manifest.read_text())["candidates"]:
            cfg = Path(rec["config"]); rd = args.output / f"candidate_{rec['id']:03d}"
            try:
                vtp = run_candidate(cfg, args, rd)
                err = score_vtp(vtp, args.target, args.target_amplitude_mm, args.panel_x, args.panel_y)
                rows.append({"id": rec["id"], "angles": rec["angles"], "error_mm": err, "vtp": str(vtp)})
                print(f"candidate {rec['id']:03d} {rec['angles']} error={err:.4f} mm")
            except Exception as exc:
                rows.append({"id": rec["id"], "angles": rec["angles"], "error_mm": None, "error": str(exc)})
                print(f"candidate {rec['id']:03d} FAILED: {exc}")
        rows.sort(key=lambda r: float("inf") if r["error_mm"] is None else r["error_mm"])
        (args.output / "ranking.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[:5], indent=2))

if __name__ == "__main__":
    main()
