"""Saved-page checks must inspect rendered tick positions, not mutate 3-D projections."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from calibration.analysis import attempt209_publication as pub


@pytest.mark.parametrize('projection', [None, '3d'])
def test_save_checks_rendered_tick_positions(tmp_path, monkeypatch, projection):
    monkeypatch.setattr(pub, 'OUT', tmp_path)
    with plt.rc_context(pub.STYLE):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection=projection)
        if projection:
            ax.set(xlim=(-105, 105), ylim=(-90, 220), zlim=(-5, 5))
            ax.set_xticks([-100, 0, 100])
            ax.set_yticks([-85, 65, 215])
            ax.view_init(elev=23, azim=-65)
            ax.set_box_aspect((210, 310, 80))
        pub.save(fig, 'layout', 'Fixture')
    assert (tmp_path / 'layout.pdf').is_file()


def test_save_still_rejects_off_page_text(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, 'OUT', tmp_path)
    fig = plt.figure(figsize=(4, 3))
    fig.text(-1, 0, 'Outside the page')
    try:
        with pytest.raises(ValueError, match='text outside page'):
            pub.save(fig, 'invalid', 'Fixture')
    finally:
        plt.close(fig)
    assert not (tmp_path / 'invalid.pdf').exists()
