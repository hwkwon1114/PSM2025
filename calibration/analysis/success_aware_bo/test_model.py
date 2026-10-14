import copy
import itertools
import unittest

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from calibration.analysis.success_aware_bo.model import (
    CompletionGP, SuccessAwareLogEI, joint_base_samples, posterior_draws,
    log_successful_improvement, repair_batch, training_records)


def record(i, valid=True, failure=None):
    r = {'attempt_id': i, 'normalized_parameters': [.1*i, .2, .3, .4], 'valid': valid}
    if valid:
        r['objective_mm'] = 30.*i  # Poor fits remain real losses.
    else:
        r.update(failure_class=failure, error='fixture failure', failure_phase='solver',
                 solver_termination={'timeout': failure == 'timeout', 'numerical': failure == 'solver',
                                     'limit_seconds': 3600})
    return r


class StandardGaussian(torch.nn.Module):
    """Independent unit-normal posterior gives an analytically known EI integral."""
    num_outputs = 1
    batch_shape = torch.Size([])
    mode = 'laplace'

    def posterior(self, X):
        mean = X[..., 0]*0
        covariance = torch.eye(X.shape[-2], dtype=X.dtype, device=X.device)
        return GPyTorchPosterior(MultivariateNormal(mean, covariance_matrix=covariance))


class SuccessAwareTests(unittest.TestCase):
    def test_actual_losses_and_verified_completion_labels(self):
        records = [record(1), record(2, False, 'timeout'), record(3, False, 'solver'),
                   record(4, False, 'infrastructure'), record(5, False, 'registration')]
        before = copy.deepcopy(records)
        losses, labels, excluded = training_records(records)
        self.assertEqual([r['objective_mm'] for r in losses], [30.])
        self.assertEqual([(r['attempt_id'], y) for r, y in labels], [(1, 1), (2, 0), (3, 0)])
        self.assertEqual([r['attempt_id'] for r in excluded], [4, 5])
        self.assertEqual(records, before)

    def test_unverified_and_other_timeout_caps_excluded(self):
        for cap in (True, 60, 7200):
            r = record(1, False, 'timeout')
            r['solver_termination']['limit_seconds'] = cap
            self.assertFalse(training_records([r])[1])
        r = record(1, False, 'solver')
        r['solver_termination']['numerical'] = False
        self.assertFalse(training_records([r])[1])

    def test_duplicates_malformed_and_contradictory_rejected(self):
        with self.assertRaises(ValueError):
            training_records([record(1), record(1)])
        r = record(1)
        r['failure_class'] = 'timeout'
        with self.assertRaises(ValueError):
            training_records([r])
        for y in ([2.], [float('nan')]):
            with self.assertRaises(ValueError):
                CompletionGP([[.1]*4], y)

    def test_beta_uncertainty_no_label_single_class(self):
        for labels in ([], [0, 0, 0], [1, 1, 1]):
            gp = CompletionGP([[.2]*4]*len(labels), labels)
            self.assertEqual(gp.mode, 'exchangeable_beta')
            p = gp.probabilities(torch.zeros(4, 4, dtype=torch.double))
            self.assertLess(p['p05'][0], p['mean'][0])
            self.assertLess(p['mean'][0], p['p95'][0])
            _, uniform = joint_base_samples(4, 128, 123, beta_rate=True)
            self.assertTrue(torch.isfinite(gp.beta_logits_from_uniform(uniform)).all())

    def test_laplace_covariance_orientation_and_restore(self):
        X = torch.tensor([[.05]*4, [.1]*4, [.9]*4, [.95]*4], dtype=torch.double)
        gp = CompletionGP(X, [0, 0, 1, 1])
        probe = torch.tensor([[.08]*4, [.92]*4], dtype=torch.double, requires_grad=True)
        posterior = gp.posterior(probe)
        self.assertTrue((torch.linalg.eigvalsh(posterior.mvn.covariance_matrix) > 0).all())
        self.assertLess(gp.probabilities(probe)['mean'][0], gp.probabilities(probe)['mean'][1])
        posterior.mean.sum().backward()
        self.assertTrue(torch.isfinite(probe.grad).all())
        restored = CompletionGP(X, [0, 0, 1, 1], fit=False)
        restored.load_state_dict(gp.state_dict())
        torch.testing.assert_close(gp.posterior(probe).mean, restored.posterior(probe).mean)
        torch.testing.assert_close(gp.posterior(probe).variance, restored.posterior(probe).variance)

    def test_closed_form_matches_all_bernoulli_outcomes(self):
        improvements = torch.tensor([4., 1., 3., 2.], dtype=torch.double)
        p = torch.tensor([.2, .8, .4, .6], dtype=torch.double)
        expected = 0.
        for mask in itertools.product((0, 1), repeat=4):
            probability = torch.prod(torch.where(torch.tensor(mask).bool(), p, 1-p))
            expected += probability*max(float(v)*m for v, m in zip(improvements, mask))
        actual = log_successful_improvement(improvements, torch.logit(p), 0.).exp()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_failed_member_cannot_erase_successful_improvement(self):
        y = torch.tensor([100., 2., 1., -1.], dtype=torch.double)
        h = torch.tensor([-100., 100., 100., 100.], dtype=torch.double)
        torch.testing.assert_close(log_successful_improvement(y, h, 0.).exp(),
                                   torch.tensor(2., dtype=torch.double), atol=1e-6, rtol=1e-6)
        self.assertGreater(log_successful_improvement(y, torch.zeros_like(h), 0.).exp(), 1.)

    def test_all_success_all_fail_negative_extremes_and_finite_gradients(self):
        draws = torch.tensor([4., 1., 3., 2.], dtype=torch.double, requires_grad=True)
        h = torch.full_like(draws, 100., requires_grad=True)
        value = log_successful_improvement(draws, h, 0.)
        torch.testing.assert_close(value.exp(), torch.tensor(4., dtype=torch.double), atol=1e-6, rtol=1e-6)
        value.backward()
        self.assertTrue(torch.isfinite(draws.grad).all())
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(log_successful_improvement(-1000*draws.detach(), -1000*torch.ones_like(h), 0.)))

    def test_acquisition_shapes_gradients_and_sampler_restore(self):
        X = torch.tensor([[.1]*4, [.3]*4, [.7]*4, [.9]*4], dtype=torch.double)
        loss = SingleTaskGP(X, torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.double))
        loss.eval()
        for labels in ([0, 0, 1, 1], [1, 1, 1, 1]):
            success = CompletionGP(X, labels)
            acq = SuccessAwareLogEI(loss, success, 4., samples=32, seed=31)
            probe = torch.rand(2, 4, 4, dtype=torch.double, requires_grad=True)
            result = acq(probe)
            self.assertEqual(result.shape, (2,))
            result.sum().backward()
            self.assertTrue(torch.isfinite(probe.grad).all())
            twin = SuccessAwareLogEI(loss, success, 4., samples=32, seed=31)
            torch.testing.assert_close(result, twin(probe), atol=1e-12, rtol=1e-12)

    def test_joint_gaussian_quadrants_and_cross_moment(self):
        for q in (1, 4):
            for seed in (31, 41, 51):
                f, h = joint_base_samples(q, 8192, seed)
                for i in range(q):
                    quadrant = ((f[:, i] > 0) & (h[:, i] > 0)).double().mean()
                    self.assertAlmostEqual(float(quadrant), .25, places=12)
                    self.assertLess(abs(float((f[:, i]*h[:, i]).mean())), .015)

    def test_joint_beta_quadrants_and_shared_batch_rate(self):
        for seed in (31, 41, 51):
            f, uniform = joint_base_samples(4, 8192, seed, beta_rate=True)
            self.assertEqual(uniform.shape, (8192, 1))
            quadrant = ((f[:, 0] > 0) & (uniform[:, 0] > .5)).double().mean()
            self.assertAlmostEqual(float(quadrant), .25, places=12)
            beta_gp = CompletionGP([], [])
            rate = beta_gp.beta_logits_from_uniform(uniform).sigmoid()
            torch.testing.assert_close(rate, uniform[:, 0], atol=1e-12, rtol=1e-12)
            acq = SuccessAwareLogEI(StandardGaussian(), beta_gp, 0., samples=128, seed=seed)
            acq(torch.zeros(4, 4, dtype=torch.double))
            self.assertEqual(acq.completion_base.shape[-1], 1)

    def test_known_gaussian_logistic_integral(self):
        # E[Z_+ sigmoid(H)] = E[Z_+] * .5 for independent standard normals.
        expected = .5 / np.sqrt(2*np.pi)
        for seed in (31, 41, 51):
            acq = SuccessAwareLogEI(StandardGaussian(), StandardGaussian(), 0., samples=8192, seed=seed)
            actual = float(acq(torch.zeros(1, 4, dtype=torch.double)).exp())
            self.assertLess(abs(actual-expected), .001)

    def test_known_gaussian_beta_integral(self):
        # Includes no-label Uniform rate and single-class Beta(3,1)/Beta(1,3).
        for labels in ([], [1, 1], [0, 0]):
            success = CompletionGP([[.2]*4]*len(labels), labels)
            a, b = success.beta_shapes.numpy()
            expected = a/(a+b) / np.sqrt(2*np.pi)
            for seed in (31, 41, 51):
                acq = SuccessAwareLogEI(StandardGaussian(), success, 0., samples=8192, seed=seed)
                actual = float(acq(torch.zeros(1, 4, dtype=torch.double)).exp())
                self.assertLess(abs(actual-expected), .001)

    def test_posterior_transform_preserves_covariance_and_common_batch_samples(self):
        covariance = torch.tensor([[1., .6], [.6, 1.]], dtype=torch.double)
        posterior = GPyTorchPosterior(MultivariateNormal(torch.zeros(2, dtype=torch.double), covariance))
        f, _ = joint_base_samples(2, 8192, 31)
        draws = posterior_draws(posterior, f)
        torch.testing.assert_close(torch.cov(draws.T), covariance, atol=.015, rtol=.015)
        self.assertLess(float(draws.mean(0).abs().max()), .005)
        batched = GPyTorchPosterior(MultivariateNormal(torch.zeros(3, 2, dtype=torch.double), covariance))
        batch_draws = posterior_draws(batched, f)
        for i in range(3):
            torch.testing.assert_close(batch_draws[:, i], draws, atol=1e-12, rtol=1e-12)

    def test_repair_preserves_all_attempt_and_pairwise_separation(self):
        records = [record(1), record(2, False, 'timeout'), record(3, False, 'infrastructure')]
        raw = np.array([records[0]['normalized_parameters']]*4)
        final, changed = repair_batch(raw, records, 2026)
        self.assertEqual(changed, [0, 1, 2, 3])
        old = np.array([r['normalized_parameters'] for r in records])
        self.assertGreaterEqual(np.linalg.norm(final[:, None]-old, axis=-1).min(), .08)
        distances = np.linalg.norm(final[:, None]-final, axis=-1) + np.eye(4)*100
        self.assertGreaterEqual(distances.min(), .08)
        self.assertTrue(np.all((final >= 0) & (final <= 1)))


if __name__ == '__main__':
    unittest.main()
