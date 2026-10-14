"""Targeted unit tests for unified geometry, registration, and symmetric metrics."""
import unittest
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from calibration.ewcal.geometry import (
    triangle_mesh, points, reference_dual_areas, physical_step,
    compute_bidirectional_metrics, align_target, u_to_rotvec, rotvec_to_u
)

class TestGeometry(unittest.TestCase):
    def setUp(self):
        # Create a simple 2D reference rectangle in z=0: [0, 20] x [0, 30] mm
        # 4 vertices, 2 triangles
        self.ref_pos = np.array([
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 30.0, 0.0],
            [0.0, 30.0, 0.0]
        ], dtype=float)
        self.faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=int)
        self.mesh = triangle_mesh(self.ref_pos, self.faces)

    def test_reference_dual_areas(self):
        dual_areas = reference_dual_areas(self.ref_pos, self.faces)
        total_area = np.sum(dual_areas)
        self.assertAlmostEqual(total_area, 600.0, places=6)
        self.assertTrue(np.all(dual_areas > 0))
        # Vertices 1 and 3 belong to 1 triangle each (area 300 / 3 = 100)
        # Vertices 0 and 2 belong to 2 triangles each (area 300/3 + 300/3 = 200)
        self.assertAlmostEqual(dual_areas[1], 100.0, places=6)
        self.assertAlmostEqual(dual_areas[3], 100.0, places=6)
        self.assertAlmostEqual(dual_areas[0], 200.0, places=6)
        self.assertAlmostEqual(dual_areas[2], 200.0, places=6)

    def test_rotvec_mappings(self):
        angle = np.deg2rad(5.0)
        u_zero = np.zeros(3)
        self.assertTrue(np.allclose(u_to_rotvec(u_zero, angle), 0.0))
        self.assertTrue(np.allclose(rotvec_to_u(u_zero, angle), 0.0))

        # Check roundtrip for intermediate rotation
        rot_orig = np.array([0.02, -0.03, 0.01])
        self.assertLess(np.linalg.norm(rot_orig), angle)
        u_val = rotvec_to_u(rot_orig, angle)
        rot_recovered = u_to_rotvec(u_val, angle)
        self.assertTrue(np.allclose(rot_orig, rot_recovered, atol=1e-12))

    def test_physical_step(self):
        angle = np.deg2rad(5.0)
        x_old = np.zeros(6)
        x_new = np.array([0.0, 0.0, 0.0, 1.0, 2.0, -1.0])
        step = physical_step(x_old, x_new, angle, self.ref_pos)
        self.assertAlmostEqual(step["rotation_step_deg"], 0.0, places=6)
        self.assertAlmostEqual(step["translation_step_mm"], np.sqrt(6.0), places=5)
        self.assertAlmostEqual(step["point_motion_rms_mm"], np.sqrt(6.0), places=5)

    def test_align_target_identity(self):
        # Target point cloud identical to mesh vertices plus boundary grid
        grid_x, grid_y = np.meshgrid(np.linspace(0, 20, 21), np.linspace(0, 30, 31))
        target_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(len(grid_x.ravel()))])

        settings = {
            "rotation_bound_deg": 5.0,
            "translation_bounds_mm": [5.0, 5.0, 20.0],
            "fit_points": 200,
            "icp_iterations": 30,
            "reject_at_bound": True,
            "metric": "symmetric"
        }

        metrics, aligned, dist, closest = align_target(
            self.mesh, target_pts, settings,
            ref_positions_mm=self.ref_pos, faces=self.faces
        )

        self.assertTrue(metrics["outer_converged"])
        self.assertTrue(metrics["registration_qualified"])
        self.assertFalse(metrics["pose_near_bound"])
        self.assertLess(metrics["rms_symmetric_mm"], 1e-4)
        self.assertLess(metrics["rms_target_to_sim_mm"], 1e-4)
        self.assertLess(metrics["rms_sim_to_target_weighted_mm"], 1e-4)
        self.assertAlmostEqual(metrics["rms_mm"], metrics["rms_symmetric_mm"], places=6)

    def test_align_target_near_bound_rejection(self):
        # Target point cloud shifted in z by 19.5 mm (exceeds 95% of 20 mm bound)
        grid_x, grid_y = np.meshgrid(np.linspace(1, 19, 10), np.linspace(1, 29, 15))
        target_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(150, 19.5)])

        settings = {
            "rotation_bound_deg": 5.0,
            "translation_bounds_mm": [5.0, 5.0, 20.0],
            "fit_points": 200,
            "icp_iterations": 30,
            "reject_at_bound": True,
            "metric": "symmetric"
        }

        metrics, aligned, dist, closest = align_target(
            self.mesh, target_pts, settings,
            ref_positions_mm=self.ref_pos, faces=self.faces
        )

        self.assertTrue(metrics["pose_near_bound"])
        self.assertFalse(metrics["registration_qualified"])

    def test_compute_bidirectional_metrics(self):
        # Simulate target points at z = 2.0 mm
        grid_x, grid_y = np.meshgrid(np.linspace(0, 20, 5), np.linspace(0, 30, 5))
        target_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(25, 2.0)])
        fwd_dist = np.full(len(target_pts), 2.0)

        # Sim vertices at z = 0.0 mm
        metrics = compute_bidirectional_metrics(
            self.mesh, target_pts, self.ref_pos, self.faces, fwd_dist
        )

        self.assertAlmostEqual(metrics["rms_target_to_sim_mm"], 2.0, places=6)
        self.assertAlmostEqual(metrics["rms_sim_to_target_weighted_mm"], 2.0, places=6)
        self.assertAlmostEqual(metrics["rms_symmetric_mm"], 2.0, places=6)

if __name__ == "__main__":
    unittest.main()
