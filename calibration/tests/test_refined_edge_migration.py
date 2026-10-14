"""Bounded solver-free controls for the selected refined scan-frame migration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from calibration.analysis import refined_edge_migration as migration


class RefinedEdgeTests(unittest.TestCase):
    def test_selected_fixed_rotation(self):
        angle = .05586775269924428
        observed, rotation = migration.fixed_rotation({'additional_yaw_deg':angle}, angle)
        self.assertEqual(observed, angle)
        p = np.array([[1.,2.,3.],[-4.,5.,6.]])
        np.testing.assert_array_equal((p @ rotation.T)[:,2], p[:,2])
        np.testing.assert_allclose(np.linalg.norm(p @ rotation.T,axis=1), np.linalg.norm(p,axis=1))
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_policy_not_chosen_from_score(self):
        angle = .05586775269924428
        for policy, observed in [(0.,angle), (angle,0.), (float('nan'),angle), (angle,float('nan'))]:
            with self.subTest(policy=policy, observed=observed), self.assertRaises(ValueError):
                migration.fixed_rotation({'additional_yaw_deg':policy}, observed)

    def test_raw_edges_known_yaw(self):
        # Dense synthetic actual-size sheet; geometry only, no real scan tuning.
        x,y = np.meshgrid(np.arange(-100.,100.01,.333),np.arange(-150.,150.01,.333))
        xy = np.column_stack((x.ravel(),y.ravel()))
        for yaw in (0., .25, -.25):
            q = np.column_stack((xy,np.zeros(len(xy)))) @ migration.yaw_matrix(yaw).T
            estimate = migration.raw_edges(q[:,:2])
            self.assertLess(abs(estimate['mean_angle_deg']+yaw), .03)
            self.assertGreaterEqual(estimate['rows'],20)

    def test_fragmented_or_small_boundary_refused(self):
        with self.assertRaises(ValueError):
            migration.raw_edges(np.array([[0.,0.],[1.,1.],[2.,2.]]))

    def test_reviewed_input_drift_refused(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); root=base/'review'; root.mkdir()
            p=base/'source.txt';p.write_text('original')
            migration.save(root/'rescore_bindings.json',{'input_sha256':{'source.txt':migration.sha(p)}})
            with patch.object(migration,'ROOT',root), patch.object(migration,'BASE',base), patch.object(migration,'baseline',return_value=({},{})):
                migration.checked_bindings();p.write_text('changed')
                with self.assertRaisesRegex(ValueError,'Reviewed migration input changed'):
                    migration.checked_bindings()

    def test_partial_output_not_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); root=base/'review'; root.mkdir()
            with patch.object(migration,'ROOT',root), patch.object(migration,'BASE',base), patch.object(migration,'OUT',base/'out'), patch.object(migration,'baseline',return_value=({},{})):
                with self.assertRaisesRegex(FileExistsError,'Preserve previous/partial'):
                    migration.prepare()

    def test_finished_preparation_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); migration.save(root/'prepared.json', {'status':'prepared'})
            with patch.object(migration,'ROOT',root),patch.object(migration,'OUT',root/'out'),patch.object(migration,'baseline',return_value=({},{})):
                with self.assertRaisesRegex(FileExistsError,'already finalized'):
                    migration.finish()

    def test_nonclear_review_cannot_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            migration.save(root/'review_clearance.json', {'verdict':'CHANGES_REQUESTED','launch_blockers':['blocker']})
            with patch.object(migration,'ROOT',root),patch.object(migration,'checked_bindings',return_value=({},{})):
                with self.assertRaisesRegex(ValueError,'Missing reviewed clearance'):
                    migration.rescore()

    def test_changed_review_cannot_launch(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);review=base/'review.md';review.write_text('approved')
            migration.save(base/'review_clearance.json',{'verdict':'APPROVED_WITH_CAVEATS','launch_blockers':[],
                'reviewed_artifact_sha256':{'review.md':migration.sha(review)}})
            review.write_text('changed')
            with patch.object(migration,'ROOT',base),patch.object(migration,'BASE',base),patch.object(migration,'checked_bindings',return_value=({},{})):
                with self.assertRaisesRegex(ValueError,'Review/packet changed'):
                    migration.rescore()


if __name__=='__main__':
    unittest.main()
