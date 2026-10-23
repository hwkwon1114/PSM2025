"""Metric-basis fixture: identical material treatment on a coarse and a fine mesh gives identical C-I."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_square_verification as asv


def mesh(n):
    g = np.linspace(-0.1, 0.1, n + 1)
    uv = np.array([[x, y] for y in g for x in g])
    f = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            f += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return uv, np.array(f)


def state(n, c):
    uv, f = mesh(n)
    u0, u1, u2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    dm = np.stack([u2 - u1, u0 - u2], axis=2)
    abar = np.einsum('nji,jk,nkl->nil', dm, c, dm)      # Dm^T C Dm
    return {'material_uv': uv, 'faces': f, 'abar_top': abar}


def test_metric_deviation_is_mesh_independent():
    c = np.array([[1.0003, 0.0001], [0.0001, 1.0]])
    raw = [np.trace(state(n, c)['abar_top'], axis1=1, axis2=2).mean() for n in (10, 20)]
    assert raw[0] / raw[1] > 3.5                          # raw abar scales with edge length^2 (the defect found by review)
    for n in (10, 20):
        _, d = asv.material_metric_deviation(state(n, c), 'abar_top')
        assert np.allclose(d, [0.0003, 0.0001, 0.0], atol=1e-12)
