"""No-solver checks for the saved-candidate Pareto figure."""
import matplotlib.pyplot as plt
import pytest

from calibration.analysis.plot_fresh_pareto_tradeoff import (
    check_annotation_layout, load_results, nondominated,
)


def test_nondominated_minimizes_both_and_keeps_ties():
    points = [(1, 3), (2, 2), (3, 1), (4, 4), (1, 4), (2, 2), (2, 3)]
    assert nondominated(points) == [True, True, True, False, False, True, False]
    assert nondominated([]) == []


def test_saved_fresh_campaign_frontier():
    records, valid, best, _ = load_results()
    assert (len(records), len(valid), best['attempt']) == (88, 86, 84)
    assert {r['attempt'] for r in valid if r['pareto']} == {53, 59, 72, 75, 78, 83, 84, 85, 87}
    assert {r['attempt'] for r in records if not r['valid']} == {2, 9}


def test_annotation_layout_detects_overflow_and_overlap():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    one = ax.text(.2, .5, 'One', transform=ax.transAxes)
    check_annotation_layout(fig)
    two = ax.text(.2, .5, 'Two', transform=ax.transAxes)
    with pytest.raises(ValueError, match='Overlapping'):
        check_annotation_layout(fig)
    two.set_visible(False)
    point = ax.scatter([.2], [.5])
    with pytest.raises(ValueError, match='plotted point'):
        check_annotation_layout(fig)
    point.set_visible(False)
    one.set_position((1.01, .5))
    with pytest.raises(ValueError, match='outside'):
        check_annotation_layout(fig)
    plt.close(fig)
