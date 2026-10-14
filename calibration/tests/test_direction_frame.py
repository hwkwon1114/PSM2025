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
