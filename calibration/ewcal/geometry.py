"""VTK mesh I/O and exact nearest-triangle distances, in millimetres."""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def read_mesh(path, scale=1.0):
    reader = vtk.vtkSTLReader() if str(path).lower().endswith('.stl') else vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path)); reader.Update()
    tf = vtk.vtkTriangleFilter(); tf.SetInputData(reader.GetOutput()); tf.Update()
    mesh = vtk.vtkPolyData(); mesh.DeepCopy(tf.GetOutput())
    if mesh.GetNumberOfPoints() < 3 or mesh.GetNumberOfPolys() < 1:
        raise ValueError(f'No triangular surface in {path}')
    pts = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float) * scale
    if not np.isfinite(pts).all():
        raise ValueError(f'Nonfinite coordinates in {path}')
    mesh.GetPoints().SetData(numpy_to_vtk(pts, deep=True))
    mesh.Modified()
    return mesh

def points(mesh):
    return vtk_to_numpy(mesh.GetPoints().GetData()).copy()

def set_points(mesh, p):
    out = vtk.vtkPolyData(); out.DeepCopy(mesh)
    out.GetPoints().SetData(numpy_to_vtk(np.asarray(p, float), deep=True)); out.Modified()
    return out

def triangle_mesh(p, triangles):
    mesh = vtk.vtkPolyData(); vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.asarray(p, float), deep=True)); mesh.SetPoints(vp)
    cells = vtk.vtkCellArray()
    arr = np.column_stack([np.full(len(triangles), 3), triangles]).astype(np.int64).ravel()
    cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(arr, deep=True)); mesh.SetPolys(cells)
    return mesh

def write_mesh(mesh, path):
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(mesh)
    if w.Write() != 1:
        raise OSError(f'Failed writing {path}')

class SurfaceDistance:
    def __init__(self, mesh):
        self.mesh = mesh
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(mesh); self.locator.BuildLocator()
    def closest(self, p):
        cp = np.empty_like(p, dtype=float); d = np.empty(len(p))
        cell, sub, d2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
        q = [0.0, 0.0, 0.0]
        for i, x in enumerate(p):
            self.locator.FindClosestPoint(x, q, cell, sub, d2)
            cp[i] = q; d[i] = np.sqrt(max(0.0, float(d2)))
        return d, cp

def rigid_fit(source, target):
    """Proper rigid Kabsch transform for matched points; row-vector convention."""
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    correction = np.eye(3); correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    r = u @ correction @ vt
    return r, b - a @ r

def remove_solver_rigid_motion(mesh, initial):
    p, q = points(mesh), points(initial)
    if p.shape != q.shape:
        raise ValueError('Initial and final vertex correspondence changed')
    r, t = rigid_fit(p, q)
    return set_points(mesh, p @ r + t)

def pose_points(p, pose):
    return p @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]

def u_to_rotvec(u, angle):
    """Smooth bijection from R^3 to open ball B(0, angle); ||rotvec||_2 < angle strictly."""
    norm_u = float(np.linalg.norm(u))
    if norm_u < 1e-14:
        return np.zeros(3)
    return angle * (u / np.sqrt(1.0 + norm_u**2))

def rotvec_to_u(rot, angle):
    """Inverse mapping from rotvec in B(0, angle) to R^3."""
    norm_r = float(np.linalg.norm(rot))
    if norm_r < 1e-14:
        return np.zeros(3)
    s = min(0.99999999, norm_r / angle)
    norm_u = s / np.sqrt(1.0 - s**2)
    return rot * (norm_u / norm_r)

def reference_dual_areas(positions_mm, faces):
    """Barycentric dual areas on triangulated reference surface."""
    p = np.asarray(positions_mm, dtype=float)
    f = np.asarray(faces, dtype=int)
    if len(p) < 3 or len(f) < 1:
        raise ValueError("Invalid mesh dimensions for dual areas")
    v0 = p[f[:, 0]]
    v1 = p[f[:, 1]]
    v2 = p[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    if np.any(face_areas <= 0.0) or not np.all(np.isfinite(face_areas)):
        raise ValueError("Non-positive or non-finite reference face areas")
    dual_areas = np.zeros(len(p), dtype=float)
    for i in range(3):
        np.add.at(dual_areas, f[:, i], face_areas / 3.0)
    if not np.all(dual_areas > 0.0) or not np.all(np.isfinite(dual_areas)):
        raise ValueError("Non-positive or non-finite dual areas")
    return dual_areas

def physical_step(old_x, new_x, angle, sample_points):
    """Compute physical rotational, translational, and point-displacement step increments."""
    old_rot = Rotation.from_rotvec(u_to_rotvec(old_x[:3], angle))
    new_rot = Rotation.from_rotvec(u_to_rotvec(new_x[:3], angle))
    rel_rot = new_rot * old_rot.inv()
    rot_angle_deg = float(np.rad2deg(rel_rot.magnitude()))
    trans_disp_mm = float(np.linalg.norm(new_x[3:] - old_x[3:]))
    old_pts = pose_points(sample_points, np.r_[u_to_rotvec(old_x[:3], angle), old_x[3:]])
    new_pts = pose_points(sample_points, np.r_[u_to_rotvec(new_x[:3], angle), new_x[3:]])
    motion_rms_mm = float(np.sqrt(np.mean(np.sum((new_pts - old_pts)**2, axis=1))))
    return dict(
        rotation_step_deg=rot_angle_deg,
        translation_step_mm=trans_disp_mm,
        point_motion_rms_mm=motion_rms_mm
    )

def interpolate_mesh_z(mesh_pts, faces, xy):
    """Interpolate z on the exact simulation triangle connectivity (no retriangulation).
    Fails closed if any query (x, y) lies outside the simulation mesh footprint."""
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    pts = np.asarray(mesh_pts, float)
    f = np.asarray(faces, int)
    q_xy = np.asarray(xy, float)
    tri = Triangulation(pts[:, 0], pts[:, 1], f)
    z = np.asarray(LinearTriInterpolator(tri, pts[:, 2])(q_xy[:, 0], q_xy[:, 1]), float)
    if not np.all(np.isfinite(z)):
        raise ValueError('Shared footprint points lie outside predicted mesh footprint')
    return z

_REF_TRI_CACHE = {}

def sample_mesh_on_shared_footprint(mesh_pts, ref_pts, faces, xy):
    """Sample the predicted triangular surface on the shared footprint using actual mesh faces.
    Fails closed if any query point lies outside the reference sheet footprint. Uses vertical
    interpolation on deformed triangles where defined, and material-coordinate barycentric
    interpolation on the same shared footprint where large curling contracts the (x, y) projection."""
    import hashlib
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    pts = np.asarray(mesh_pts, float)
    ref = np.asarray(ref_pts, float)
    f = np.asarray(faces, int)
    q_xy = np.asarray(xy, float)
    key = (ref.shape, f.shape, hashlib.blake2b(np.ascontiguousarray(ref[:, :2]).view(np.uint8), digest_size=8).digest(),
           hashlib.blake2b(np.ascontiguousarray(f).view(np.uint8), digest_size=8).digest())
    cached = _REF_TRI_CACHE.get(key)
    if cached is None:
        ref_tri = Triangulation(ref[:, 0], ref[:, 1], f)
        rf = ref_tri.get_trifinder()
        if len(_REF_TRI_CACHE) >= 8:
            _REF_TRI_CACHE.clear()
        _REF_TRI_CACHE[key] = (ref_tri, rf)
    else:
        ref_tri, rf = cached
    if np.any(rf(q_xy[:, 0], q_xy[:, 1]) < 0):
        raise ValueError('Shared footprint points lie outside predicted mesh footprint')
    ok = np.zeros(len(q_xy), dtype=bool)
    out = np.empty((len(q_xy), 3), dtype=float)
    try:
        def_tri = Triangulation(pts[:, 0], pts[:, 1], f)
        z_vert = np.asarray(LinearTriInterpolator(def_tri, pts[:, 2])(q_xy[:, 0], q_xy[:, 1]), float)
        ok = np.isfinite(z_vert)
        out[ok, 0] = q_xy[ok, 0]
        out[ok, 1] = q_xy[ok, 1]
        out[ok, 2] = z_vert[ok]
    except Exception:
        ok[:] = False
    if not np.all(ok):
        miss = ~ok
        mx, my = q_xy[miss, 0], q_xy[miss, 1]
        mat_x = np.asarray(LinearTriInterpolator(ref_tri, pts[:, 0], trifinder=rf)(mx, my), float)
        mat_y = np.asarray(LinearTriInterpolator(ref_tri, pts[:, 1], trifinder=rf)(mx, my), float)
        mat_z = np.asarray(LinearTriInterpolator(ref_tri, pts[:, 2], trifinder=rf)(mx, my), float)
        if not (np.all(np.isfinite(mat_x)) and np.all(np.isfinite(mat_y)) and np.all(np.isfinite(mat_z))):
            raise ValueError('Shared footprint points lie outside predicted mesh footprint')
        out[miss, 0] = mat_x
        out[miss, 1] = mat_y
        out[miss, 2] = mat_z
    return out

def shared_footprint_faces(xy, max_edge_factor=1.75, min_area_factor=1e-6):
    """Triangulate 2D material-plane points without bridging missing-data gaps.
    Retains only triangles with max edge <= max_edge_factor * h and area > min_area_factor * h^2,
    where h is the positive median nearest-neighbor spacing, failing closed if none survive."""
    from scipy.spatial import Delaunay, cKDTree
    pts2 = np.asarray(xy, float)
    if len(pts2) < 3:
        raise ValueError('At least 3 points required for surface triangulation')
    d_nn, _ = cKDTree(pts2).query(pts2, k=2)
    pos = d_nn[:, 1]
    pos = pos[np.isfinite(pos) & (pos > 0.0)]
    if len(pos) == 0:
        raise ValueError('Shared footprint has no positive point spacing')
    h = float(np.median(pos))
    tri = Delaunay(pts2).simplices
    v0 = pts2[tri[:, 0]]
    v1 = pts2[tri[:, 1]]
    v2 = pts2[tri[:, 2]]
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e01, e12), e20)
    area = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    keep = (max_e <= float(max_edge_factor) * h) & (area > float(min_area_factor) * (h ** 2))
    if not np.any(keep):
        raise ValueError('No valid shared-footprint triangles within edge/area threshold')
    return tri[keep]

def compute_bidirectional_metrics(mesh, aligned_target_mm, ref_positions_mm, faces, fwd_distances_mm):
    """Historical point-cloud reverse metric for bounded 3-D registration.

    Unlike material-frame scoring, this route does not require a planar reference mesh.
    """
    from scipy.spatial import cKDTree
    dual_areas = reference_dual_areas(ref_positions_mm, faces)
    tot_area = float(np.sum(dual_areas))
    rev_dist, _ = cKDTree(aligned_target_mm).query(points(mesh))
    fwd_mse = float(np.mean(fwd_distances_mm**2))
    rev_mse = float(np.sum((dual_areas / tot_area) * rev_dist**2))
    return dict(
        rms_target_to_sim_mm=float(np.sqrt(fwd_mse)),
        rms_sim_to_target_weighted_mm=float(np.sqrt(rev_mse)),
        rms_symmetric_mm=float(np.sqrt(0.5 * (fwd_mse + rev_mse))),
        reference_area_quadrature_rms_mm=float(np.sqrt(rev_mse)),
        reference_area_mm2=tot_area,
        p95_target_to_sim_mm=float(np.quantile(fwd_distances_mm, 0.95)),
        max_target_to_sim_mm=float(np.max(fwd_distances_mm)),
        p95_sim_to_target_mm=float(np.quantile(rev_dist, 0.95)),
        max_sim_to_target_mm=float(np.max(rev_dist))
    )


def compute_shared_footprint_metrics(mesh, aligned_target_mm, ref_positions_mm, faces, fwd_distances_mm):
    """Compute double-sided surface-to-surface shape discrepancy on the shared footprint
    using actual simulation connectivity so identical shapes yield zero error in both directions."""
    dual_areas = reference_dual_areas(ref_positions_mm, faces)
    tot_area = float(np.sum(dual_areas))

    target_pts = np.asarray(aligned_target_mm, float)
    scan_faces = shared_footprint_faces(target_pts[:, :2])
    scan_mesh = triangle_mesh(target_pts, scan_faces)
    v0 = target_pts[scan_faces[:, 0], :2]
    v1 = target_pts[scan_faces[:, 1], :2]
    v2 = target_pts[scan_faces[:, 2], :2]
    tri_areas = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    scan_areas = np.zeros(len(target_pts), dtype=float)
    for k in range(3):
        np.add.at(scan_areas, scan_faces[:, k], tri_areas / 3.0)
    area_sum = float(np.sum(scan_areas))
    if area_sum <= 0.0 or not np.isfinite(area_sum):
        raise ValueError('Non-positive shared-footprint scan surface area')
    scan_weights = scan_areas / area_sum

    sim_pts = points(mesh)
    sim_shared = sample_mesh_on_shared_footprint(sim_pts, ref_positions_mm, faces, target_pts[:, :2])
    rev_weights = scan_weights

    rev_dist, _ = SurfaceDistance(scan_mesh).closest(sim_shared)

    fwd_mse = float(np.mean(fwd_distances_mm**2))
    rev_mse_weighted = float(np.sum(rev_weights * (rev_dist**2)))
    sym_rms = float(np.sqrt(0.5 * (fwd_mse + rev_mse_weighted)))

    return dict(
        rms_target_to_sim_mm=float(np.sqrt(fwd_mse)),
        rms_sim_to_target_weighted_mm=float(np.sqrt(rev_mse_weighted)),
        rms_symmetric_mm=sym_rms,
        reference_area_quadrature_rms_mm=float(np.sqrt(rev_mse_weighted)),
        reference_area_mm2=tot_area,
        p95_target_to_sim_mm=float(np.quantile(fwd_distances_mm, 0.95)),
        max_target_to_sim_mm=float(np.max(fwd_distances_mm)),
        p95_sim_to_target_mm=float(np.quantile(rev_dist, 0.95)),
        max_sim_to_target_mm=float(np.max(rev_dist))
    )

def align_target(mesh, target, settings, ref_positions_mm=None, faces=None):
    """Register target to prediction with a bounded, proper rigid transform and physical stopping.
    Rotation is strictly constrained inside ||rotvec||_2 < angle via smooth manifold bijection.
    Evaluates physical convergence (point motion and cost stagnation), optimizer success, and bounds."""
    surface = SurfaceDistance(mesh)
    angle = np.deg2rad(settings.get('rotation_bound_deg', 5.0))
    trans_bounds = np.asarray(settings.get('translation_bounds_mm', [5.0, 5.0, 20.0]), float)
    lb = np.r_[np.full(3, -np.inf), -trans_bounds]
    ub = np.r_[np.full(3, np.inf), trans_bounds]
    n = min(len(target), settings.get('fit_points', 2000))
    idx = np.linspace(0, len(target)-1, n, dtype=int)
    fit = target[idx]

    max_iter = int(settings.get('icp_iterations', 60))
    x_opt = np.zeros(6)
    trace = []
    outer_converged = False
    stop_reason = "iteration_limit"
    stagnation_count = 0

    for i in range(max_iter):
        current_pose = np.r_[u_to_rotvec(x_opt[:3], angle), x_opt[3:]]
        _, match = surface.closest(pose_points(fit, current_pose))
        def residual(x):
            rot = u_to_rotvec(x[:3], angle)
            trans = x[3:]
            return (pose_points(fit, np.r_[rot, trans]) - match).ravel()
        result = least_squares(residual, x_opt,
                               bounds=(lb, ub), max_nfev=60,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)

        step_diag = physical_step(x_opt, result.x, angle, fit)
        x_opt = result.x
        fd, _ = surface.closest(pose_points(fit, np.r_[u_to_rotvec(x_opt[:3], angle), x_opt[3:]]))
        curr_rms = float(np.sqrt(np.mean(fd**2)))

        delta_rms = None
        if i > 0 and len(trace) > 0 and trace[-1]["subset_rms_mm"] is not None:
            delta_rms = abs(curr_rms - trace[-1]["subset_rms_mm"])

        trace.append(dict(
            iteration=int(i + 1),
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            nfev=int(result.nfev),
            cost=float(result.cost),
            optimality=float(result.optimality),
            delta_translation_mm=float(step_diag["translation_step_mm"]),
            delta_rotation_deg=float(step_diag["rotation_step_deg"]),
            delta_point_motion_rms_mm=float(step_diag["point_motion_rms_mm"]),
            delta_rms_mm=float(delta_rms) if delta_rms is not None else None,
            subset_rms_mm=float(curr_rms)
        ))

        # Check 1: Physical point motion stabilization
        if (step_diag["point_motion_rms_mm"] < 0.005 and
            step_diag["rotation_step_deg"] < 0.005 and
            step_diag["translation_step_mm"] < 0.005):
            outer_converged = True
            stop_reason = "physical_motion"
            break

        # Check 2: Objective cost stagnation
        if delta_rms is not None and delta_rms < 1e-4:
            stagnation_count += 1
            if i >= 5 and stagnation_count >= 2:
                outer_converged = True
                stop_reason = "objective_stagnation"
                break
        else:
            stagnation_count = 0

    pose = np.r_[u_to_rotvec(x_opt[:3], angle), x_opt[3:]]
    aligned = pose_points(target, pose)
    distances, closest = surface.closest(aligned)

    rot_angle_rad = float(np.linalg.norm(pose[:3]))
    rot_near_bound = (rot_angle_rad > 0.95 * angle) or bool(np.any(np.abs(pose[:3]) > 0.95 * angle))
    trans_near_bound = bool(np.any(np.abs(pose[3:]) > 0.95 * trans_bounds))
    pose_near_bound = bool(rot_near_bound or trans_near_bound)

    pose_finite = bool(isinstance(pose, (list, np.ndarray)) and len(pose) == 6 and np.all(np.isfinite(pose)))
    distances_finite = bool(np.isfinite(distances).all())
    fwd_mse = float(np.mean(distances**2))
    fwd_rms = float(np.sqrt(fwd_mse))
    finite = bool(pose_finite and distances_finite and np.isfinite(fwd_rms))

    converged = bool(finite and outer_converged and len(trace) > 0 and all(t.get("success", False) for t in trace))
    qualified = bool(converged and not pose_near_bound)

    metrics = {
        'rms_target_to_sim_mm': fwd_rms,
        'median_mm': float(np.median(distances)),
        'p95_mm': float(np.quantile(distances, .95)),
        'max_mm': float(distances.max()),
        'pose': pose.tolist() if hasattr(pose, 'tolist') else list(pose),
        'pose_finite': pose_finite,
        'registration_converged': converged,
        'rotation_angle_deg': float(np.rad2deg(rot_angle_rad)),
        'pose_near_bound': pose_near_bound,
        'outer_converged': outer_converged,
        'stop_reason': stop_reason,
        'iterations_used': len(trace),
        'registration_qualified': qualified,
        'trace': trace
    }

    if ref_positions_mm is not None and faces is not None:
        bi_metrics = compute_bidirectional_metrics(mesh, aligned, ref_positions_mm, faces, distances)
        metrics.update(bi_metrics)
        metric_choice = settings.get('metric', 'symmetric')
        if metric_choice == 'symmetric':
            metrics['rms_mm'] = bi_metrics['rms_symmetric_mm']
        elif metric_choice == 'sim_to_target':
            metrics['rms_mm'] = bi_metrics['rms_sim_to_target_weighted_mm']
        elif metric_choice == 'target_to_sim':
            metrics['rms_mm'] = fwd_rms
        else:
            raise ValueError(f"Unknown metric choice '{metric_choice}'; expected 'symmetric', 'sim_to_target', or 'target_to_sim'")
    else:
        metric_choice = settings.get('metric', 'directional')
        if metric_choice in ('symmetric', 'sim_to_target'):
            raise ValueError(f"Metric '{metric_choice}' requires validated reference geometry (ref_positions_mm and faces)")
        metrics['rms_mm'] = fwd_rms

    return metrics, aligned, distances, closest


