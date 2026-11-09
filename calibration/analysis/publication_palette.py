"""Shared Attempt 209 publication colours; diverging maps use blue–white–orange."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

BLUE = '#0072B2'
ORANGE = '#D55E00'
DIVERGING = 'attempt209_blue_orange'

def register():
    if DIVERGING not in mpl.colormaps:
        mpl.colormaps.register(LinearSegmentedColormap.from_list(DIVERGING, [BLUE, '#ffffff', ORANGE]))
