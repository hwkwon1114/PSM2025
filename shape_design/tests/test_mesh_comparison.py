"""Solver-free known-answer checks for nested-grid mapping and PCA projection."""
import numpy as np
import pytest

from shape_design import analysis, family
from shape_design.analyze_mesh_check import match_vertices


def test_reference_grid_mapping_and_linear_field():
    coarse, _ = family.solver_grid(2.0)
    fine, _ = family.solver_grid(1.0)
    idx, error = match_vertices(coarse, fine)
    assert error < 1e-10
    assert np.allclose(fine[idx], coarse, rtol=0, atol=1e-10)
    h = lambda q: .003*q[:, 0]**2 - .002*q[:, 1]**2 + 7
    assert np.allclose(h(fine)[idx], h(coarse), rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match='not nested'):
        match_vertices(coarse + [0.1, 0, 0], fine)


def test_projection_uses_coarse_mean_and_basis():
    k = np.arange(-4, 5)
    field = np.array([1., -1., 2., -2.])
    X = k[:, None] * field
    result = analysis.pca(X, k)
    projected = (X - result['mean']) @ result['components'][:2].T
    assert np.allclose(projected, result['scores'][:, :2], atol=1e-12)
    assert projected[0, 0] < projected[-1, 0]
    assert result['explained_ratio'][0] == pytest.approx(1.0)
