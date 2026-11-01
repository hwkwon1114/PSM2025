"""Shared matplotlib styling and layout helpers for shape_design figures."""
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def format_sheet_ax(ax, title=None, roi_color='#333333'):
    """Standard 2D plan-view sheet formatting: equal aspect, ticks, and ROI outline."""
    ax.set_aspect('equal')
    ax.set_xlim(-110, 110)
    ax.set_ylim(3, 127)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([13, 65, 117])
    if title is not None:
        ax.set_title(title)
    if roi_color:
        ax.plot([-100, 100, 100, -100, -100], [13, 13, 117, 117, 13],
                color=roi_color, lw=0.7, ls='--')


def curvature_cmap_and_norm():
    """Consistent 3-color palette and boundary norm for Gaussian curvature sign."""
    cmap = ListedColormap(['#0072B2', '#d6d6d6', '#D55E00'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    return cmap, norm


def curvature_legend_handles():
    """Legend handles matching curvature_cmap_and_norm."""
    return [
        Patch(color='#0072B2', label='Negative K'),
        Patch(color='#D55E00', label='Positive K'),
        Patch(color='#d6d6d6', label='Zero or invalid'),
    ]
