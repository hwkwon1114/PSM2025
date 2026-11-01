"""Bounded solver-free controls for nominal material-reference comparisons."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.analysis.direction_frame_gallery import direction_frame, check_layout
from calibration.analysis.diagnose_free_translation import evaluate_saved_pose
from calibration.ewcal.geometry import triangle_mesh


def fixture():
    x, y = np.meshgrid(np.linspace(-100,100,9), np.linspace(-150,150,11))
    ref = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    faces = []
    for j in range(10):
        for i in range(8):
            a = j*9+i; faces += [[a,a+1,a+9], [a+1,a+10,a+9]]
    p = ref.copy(); p[:,2] = 0.002*ref[:,1]**2
    return p, ref, np.array(faces)


class DirectionFrameTests(unittest.TestCase):
    def test_arbitrary_whole_sheet_rigid_gauge_removed(self):
        p, ref, faces = fixture()
        baseline, _ = direction_frame(p,ref,faces)
        for rot in ([.7,-.4,1.2],[np.pi,0,0]):
            r = Rotation.from_rotvec(rot).as_matrix()
            shifted = p @ r.T + [450,-600,70]
            q, transform = direction_frame(shifted,ref,faces)
            np.testing.assert_allclose(q,baseline,atol=1e-10,rtol=0)
            self.assertAlmostEqual(transform['determinant'],1)

    def test_opposite_curvature_not_flipped_or_rejected(self):
        p,ref,faces = fixture()
        pos,_ = direction_frame(p,ref,faces)
        negative = p.copy(); negative[:,2] *= -1
        neg,_ = direction_frame(negative,ref,faces)
        np.testing.assert_allclose(pos[:,:2],neg[:,:2],atol=1e-10)
        np.testing.assert_allclose(pos[:,2],-neg[:,2],atol=1e-10)
        self.assertGreater(np.mean((pos[:,2]-neg[:,2])**2),100)
        metric,_ = evaluate_saved_pose(triangle_mesh(pos,faces),neg,ref,faces,np.zeros(6))
        self.assertTrue(np.isfinite(metric['rms_symmetric_mm']))
        self.assertGreater(metric['rms_symmetric_mm'],1)

    def test_shape_and_inputs_preserved(self):
        p,ref,faces = fixture(); old = p.copy()
        q,transform = direction_frame(p,ref,faces)
        np.testing.assert_array_equal(p,old)
        for a,b in [(0,-1),(5,17),(40,68)]:
            self.assertAlmostEqual(np.linalg.norm(q[a]-q[b]),np.linalg.norm(p[a]-p[b]))
        self.assertLess(transform['max_edge_length_change_mm'],1e-9)

    def test_point_centroid_offset_invariance(self):
        p,ref,faces = fixture(); q,_ = direction_frame(p,ref,faces)
        scan = p.copy(); scan0 = scan - scan.mean(0)
        shifted = scan + [300,-700,41]
        np.testing.assert_allclose(scan0,shifted-shifted.mean(0),atol=1e-10)
        first,_ = evaluate_saved_pose(triangle_mesh(q,faces),scan0,ref,faces,np.zeros(6))
        second,_ = evaluate_saved_pose(triangle_mesh(q,faces),shifted-shifted.mean(0),ref,faces,np.zeros(6))
        self.assertAlmostEqual(first['rms_symmetric_mm'],second['rms_symmetric_mm'],places=9)

    def test_bad_reference_rejected_not_high_rms(self):
        p,ref,faces = fixture()
        for bad in [np.zeros_like(p),np.full_like(p,np.nan),p[:-1]]:
            with self.assertRaises(ValueError):
                direction_frame(bad,ref,faces)

    def test_identical_shape_on_staggered_inset_footprint_has_zero_objective(self):
        from calibration.ewcal.geometry import interpolate_mesh_z
        from calibration.ewcal.material_frame import score_material_frame
        p, ref, faces = fixture()
        sx, sy = np.meshgrid(np.linspace(-85, 85, 18), np.linspace(-135, 135, 28))
        s_xy = np.column_stack([sx.ravel(), sy.ravel()])
        sz = interpolate_mesh_z(p, faces, s_xy)
        scan = np.column_stack([s_xy[:, 0], s_xy[:, 1], sz]) + np.array([12.0, -34.0, 5.5])
        metrics, _, _, dist, _ = score_material_frame(triangle_mesh(p, faces), scan, ref, faces)
        self.assertLess(metrics['rms_target_to_sim_mm'], 1e-12)
        self.assertLess(metrics['rms_sim_to_target_weighted_mm'], 1e-12)
        self.assertLess(metrics['rms_symmetric_mm'], 1e-12)
        self.assertLess(float(np.max(dist)), 1e-12)

    def test_nonseparable_patch_opposite_diagonals_zero_floor(self):
        from calibration.ewcal.geometry import interpolate_mesh_z, rigid_fit
        from calibration.ewcal.material_frame import score_material_frame
        xs, ys = np.arange(-20.0, 20.1, 2.0), np.arange(-30.0, 30.1, 2.0)
        gx, gy = np.meshgrid(xs, ys)
        ref = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        ny, nx = gx.shape
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v00 = j * nx + i; v10 = v00 + 1; v01 = (j + 1) * nx + i; v11 = v01 + 1
                if (i + j) % 2 == 0:
                    faces.extend([[v00, v10, v11], [v00, v11, v01]])
                else:
                    faces.extend([[v00, v10, v01], [v10, v11, v01]])
        faces = np.asarray(faces, int)
        p = ref.copy()
        p[:, 2] = 0.015 * ref[:, 0] * ref[:, 1] + 0.004 * ref[:, 0]**2 - 0.003 * ref[:, 1]**2
        # Sample cell centers (where opposite diagonals of z=x*y differ)
        sx, sy = np.meshgrid(np.arange(-17.0, 17.1, 2.0), np.arange(-27.0, 27.1, 2.0))
        s_xy = np.column_stack([sx.ravel(), sy.ravel()])
        q0, _ = direction_frame(p, ref, faces)
        flat_s = np.column_stack([s_xy[:, 0], s_xy[:, 1], np.zeros(len(s_xy))])
        for _ in range(6):
            sz = interpolate_mesh_z(q0, faces, s_xy)
            r, t = rigid_fit(np.column_stack([s_xy[:, 0], s_xy[:, 1], sz]), flat_s)
            q0 = q0 @ r + t
        scan = np.column_stack([s_xy[:, 0], s_xy[:, 1], interpolate_mesh_z(q0, faces, s_xy)])
        metrics, _, _, dist, _ = score_material_frame(triangle_mesh(p, faces), scan, ref, faces)
        self.assertLess(metrics['rms_target_to_sim_mm'], 1e-12)
        self.assertLess(metrics['rms_sim_to_target_weighted_mm'], 1e-12)
        self.assertLess(metrics['rms_symmetric_mm'], 1e-12)
        self.assertLess(float(np.max(dist)), 1e-12)

    def test_production_2mm_inset_footprint_with_hole_and_asymmetric_tilt(self):
        from calibration.ewcal.geometry import interpolate_mesh_z, rigid_fit, shared_footprint_faces
        from calibration.ewcal.material_frame import score_material_frame
        xs, ys = np.arange(-100.0, 100.1, 2.0), np.arange(-150.0, 150.1, 2.0)
        gx, gy = np.meshgrid(xs, ys)
        ref = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        ny, nx = gx.shape
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v00 = j * nx + i; v10 = v00 + 1; v01 = (j + 1) * nx + i; v11 = v01 + 1
                if (i + j) % 2 == 0:
                    faces.extend([[v00, v10, v11], [v00, v11, v01]])
                else:
                    faces.extend([[v00, v10, v01], [v10, v11, v01]])
        faces = np.asarray(faces, int)
        p = ref.copy()
        x, y = ref[:, 0], ref[:, 1]
        p[:, 2] = 0.05 * x - 0.035 * y + 0.0004 * x * y + 0.0003 * x**2 - 0.0002 * y**2
        sx, sy = np.meshgrid(np.arange(-95.0, 83.1, 2.0), np.arange(-131.0, 145.1, 2.0))
        s_xy = np.column_stack([sx.ravel(), sy.ravel()])
        s_xy = s_xy[~((np.abs(s_xy[:, 0] - 11.0) < 0.5) & (np.abs(s_xy[:, 1] + 23.0) < 0.5))]
        s_xy = s_xy - s_xy.mean(0)
        tri_scan = shared_footprint_faces(s_xy)
        edges = np.linalg.norm(s_xy[tri_scan[:, [0, 1, 2]]] - s_xy[tri_scan[:, [1, 2, 0]]], axis=-1)
        self.assertLessEqual(float(edges.max()), 3.5)
        q0, _ = direction_frame(p, ref, faces)
        flat_s = np.column_stack([s_xy[:, 0], s_xy[:, 1], np.zeros(len(s_xy))])
        for _ in range(6):
            sz = interpolate_mesh_z(q0, faces, s_xy)
            r, t = rigid_fit(np.column_stack([s_xy[:, 0], s_xy[:, 1], sz]), flat_s)
            q0 = q0 @ r + t
        scan = np.column_stack([s_xy[:, 0], s_xy[:, 1], interpolate_mesh_z(q0, faces, s_xy)]) + np.array([25.0, -40.0, 9.0])
        rot = Rotation.from_rotvec([0.4, -0.3, 0.7]).as_matrix()
        p_moved = p @ rot.T + np.array([-80.0, 110.0, -25.0])
        metrics, _, _, dist, _ = score_material_frame(triangle_mesh(p_moved, faces), scan, ref, faces)
        self.assertLess(metrics['rms_target_to_sim_mm'], 1e-12)
        self.assertLess(metrics['rms_sim_to_target_weighted_mm'], 1e-12)
        self.assertLess(metrics['rms_symmetric_mm'], 1e-12)
        self.assertLess(float(np.max(dist)), 1e-12)

    def test_incomplete_mesh_coverage_fails_closed(self):
        from calibration.ewcal.material_frame import score_material_frame
        p, ref, faces = fixture()
        # Scan extending outside [-100, 100] x [-150, 150] must fail closed, not fall back to full mesh
        sx, sy = np.meshgrid(np.linspace(-130, 130, 10), np.linspace(-180, 180, 10))
        scan = np.column_stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)])
        with self.assertRaises(ValueError):
            score_material_frame(triangle_mesh(p, faces), scan, ref, faces)

    def test_expanded_deformed_mesh_outside_reference_fails_closed(self):
        from calibration.ewcal.geometry import sample_mesh_on_shared_footprint
        ref = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        expanded = ref.copy()
        expanded[:, :2] *= 2.0
        # Query (1.5, 0.0) lies inside expanded deformed mesh [-2, 2]^2 but outside reference [-1, 1]^2
        with self.assertRaises(ValueError):
            sample_mesh_on_shared_footprint(expanded, ref, faces, np.array([[0.0, 0.0], [1.5, 0.0]]))
        # Contracted deformed mesh [-0.5, 0.5]^2 falls back via reference barycentric coordinates for (0.8, 0.0) in [-1, 1]^2
        contracted = ref.copy()
        contracted[:, :2] *= 0.5
        contracted[:, 2] = 1.25
        pts = sample_mesh_on_shared_footprint(contracted, ref, faces, np.array([[0.0, 0.0], [0.8, 0.0]]))
        self.assertTrue(np.all(np.isfinite(pts)))
        self.assertAlmostEqual(float(pts[1, 0]), 0.4, places=12)
        self.assertAlmostEqual(float(pts[1, 2]), 1.25, places=12)

    def test_finer_grid_hole_and_min_area_filter(self):
        from calibration.ewcal.geometry import shared_footprint_faces
        gx, gy = np.meshgrid(np.arange(-3.0, 3.1, 1.0), np.arange(-3.0, 3.1, 1.0))
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        # Remove interior point (0, 0) on 1 mm grid: 2 mm chords across hole must be rejected by 1.75 * h = 1.75 mm
        xy_hole = xy[~((np.abs(xy[:, 0]) < 1e-9) & (np.abs(xy[:, 1]) < 1e-9))]
        tri = shared_footprint_faces(xy_hole)
        edges = np.linalg.norm(xy_hole[tri[:, [0, 1, 2]]] - xy_hole[tri[:, [1, 2, 0]]], axis=-1)
        self.assertLessEqual(float(edges.max()), 1.75)
        # Add nearly collinear splinter point that creates a degenerate triangle with area < 1e-6 * h^2
        xy_deg = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5 + 1e-9]])
        tri_deg = shared_footprint_faces(xy_deg)
        v0, v1, v2 = xy_deg[tri_deg[:, 0]], xy_deg[tri_deg[:, 1]], xy_deg[tri_deg[:, 2]]
        areas = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
        self.assertTrue(np.all(areas > 1e-6))

    def test_layout_known_defects(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Deliberate off-canvas text must not enter constrained-layout's solver.
        fig,ax = plt.subplots(figsize=(6,4))
        check_layout(fig)
        a = ax.text(.5,.5,'same',transform=ax.transAxes)
        b = ax.text(.5,.5,'same',transform=ax.transAxes)
        with self.assertRaises(AssertionError): check_layout(fig)
        b.remove(); a.set_position((100,100))
        with self.assertRaises(AssertionError): check_layout(fig)
        a.remove(); check_layout(fig); plt.close(fig)


if __name__ == '__main__':
    unittest.main()
