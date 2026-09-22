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

def align_target(mesh, target, settings):
    """Register target to prediction with a bounded, proper rigid transform only."""
    surface = SurfaceDistance(mesh)
    angle = np.deg2rad(settings['rotation_bound_deg'])
    bounds = np.r_[np.full(3, angle), settings['translation_bounds_mm']]
    n = min(len(target), settings['fit_points'])
    idx = np.linspace(0, len(target)-1, n, dtype=int)
    fit = target[idx]
    pose = np.zeros(6)
    # Deterministic ICP: closest triangle points, then bounded least squares.
    for _ in range(settings['icp_iterations']):
        _, match = surface.closest(pose_points(fit, pose))
        result = least_squares(lambda x: (pose_points(fit, x)-match).ravel(), pose,
                               bounds=(-bounds, bounds), max_nfev=60,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        delta = np.linalg.norm(result.x - pose)
        pose = result.x
        if delta < 1e-7:
            break
    aligned = pose_points(target, pose)
    distances, closest = surface.closest(aligned)
    return {'rms_mm': float(np.sqrt(np.mean(distances**2))),
            'median_mm': float(np.median(distances)),
            'p95_mm': float(np.quantile(distances, .95)),
            'max_mm': float(distances.max()), 'pose': pose.tolist(),
            'pose_near_bound': bool(np.any(np.abs(pose) > .98*bounds))}, aligned, distances, closest
