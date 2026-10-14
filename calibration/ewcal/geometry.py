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

def compute_bidirectional_metrics(mesh, aligned_target_mm, ref_positions_mm, faces, fwd_distances_mm):
    """Compute bidirectional shape discrepancy combining forward (scan-to-sim)
    and reference-area-weighted reverse (sim-to-scan) metrics."""
    from scipy.spatial import cKDTree
    dual_areas = reference_dual_areas(ref_positions_mm, faces)
    tot_area = float(np.sum(dual_areas))
    weights = dual_areas / tot_area

    sim_pts = points(mesh)
    kdtree = cKDTree(aligned_target_mm)
    rev_dist, _ = kdtree.query(sim_pts)

    fwd_mse = float(np.mean(fwd_distances_mm**2))
    rev_mse_weighted = float(np.sum(weights * (rev_dist**2)))
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


