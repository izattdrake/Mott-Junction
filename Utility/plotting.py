import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox 
 
# ----------------------------------------------------------------------
# LaTeX-friendly style
# ----------------------------------------------------------------------
# Set USE_TEX = True to typeset every label with your actual LaTeX install
# (perfect font match with the document). Requires latex + dvipng + cm-super
# on your PATH. If that's not available, False uses matplotlib's bundled
# Computer Modern, which is very close.
USE_TEX = False
 
mpl.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,      # cmr10 has no unicode minus glyph
 
    # Sizes are in points and land in the PDF at these *absolute* sizes,
    # so pick figsize below to equal the final width in the document and
    # include it at that width (no \includegraphics scaling).
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
 
    "lines.linewidth": 2.0,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
 
    "pdf.fonttype": 42,               # embed real fonts, not Type-3
    "ps.fonttype": 42,
})
 
 
def save_axes_centered(fig, ax, filename, pad=0.03):
    """Save so that the *plot box* sits at the center of the image.
 
    A normal bbox_inches='tight' crop hugs the labels, so the y-label makes
    the left margin wider than the right and \\centering in LaTeX ends up
    centering the labels, not the data. This pads the short side instead,
    making the saved bounding box symmetric about the axes center.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)                                 # inches
    axbb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
 
    cx = 0.5 * (axbb.x0 + axbb.x1)
    cy = 0.5 * (axbb.y0 + axbb.y1)
    hw = max(cx - tight.x0, tight.x1 - cx) + pad
    hh = max(cy - tight.y0, tight.y1 - cy) + pad
 
    bbox = Bbox.from_extents(cx - hw, cy - hh, cx + hw, cy + hh)
    fig.savefig(f"Figures/{filename}", bbox_inches=bbox, pad_inches=0.0)