# Toric code geometry plot (periodic square lattice with star/plaquette highlights)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=r'\usepackage{amsfonts, amssymb, amsmath, physics}')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 12


def draw_toric_code(Lx=6, Ly=6, spacing=1.0, savepath="toric_code_geometry.png", pbc=True):
    """
    Draw a toric-code style lattice:
      - magenta square lattice
      - blue diamond plaquettes (Z_p)
      - vertices as white circles with magenta edges
      - highlight one star operator X_s near the center
      - annotate one plaquette with Z_p
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_aspect('equal', adjustable='box')

    def grid_to_xy(i, j):
        return i * spacing, j * spacing

    # grid lines
    magenta = (0.8, 0, 0.5)
    for j in range(1, Ly + 1):
        x0, y = grid_to_xy(0, j)
        x1, _ = grid_to_xy(Lx, j)
        ax.plot([x0, x1], [y, y], color=magenta, lw=2)
    for j in range(1):
        x0, y = grid_to_xy(0, j)
        x1, _ = grid_to_xy(Lx, j)
        ax.plot([x0, x1], [y, y], color=magenta, lw=2, ls='--')

    for i in range(Lx):
        x, y0 = grid_to_xy(i, 0)
        _, y1 = grid_to_xy(i, Ly)
        ax.plot([x, x], [y0, y1], color=magenta, lw=2)

    ls = '-' if (pbc) else '--'
    for i in range(Lx, Lx + 1):
        x, y0 = grid_to_xy(i, 0)
        _, y1 = grid_to_xy(i, Ly)
        ax.plot([x, x], [y0, y1], color=magenta, lw=2, ls=ls)

    # blue diamonds (plaquettes)
    blue = (0.64, 0.84, 1.0)
    for i in range(Lx):
        for j in range(Ly):
            cx, cy = grid_to_xy(i + 0.5, j + 0.5)
            r = 0.5 * spacing
            ax.add_patch(Polygon(
                [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
                closed=True, facecolor=blue, edgecolor='none', alpha=0.95))

    # vertices
    for i in range(Lx):
        for j in range(Ly):
            x, y = grid_to_xy(i, j + 0.5)
            ax.add_patch(Circle((x, y), 0.075*spacing,
                                facecolor='white', edgecolor='k', lw=2, zorder=5))
    for i in range(Lx, Lx + 1):
        for j in range(Ly):
            x, y = grid_to_xy(i, j + 0.5)
            ax.add_patch(Circle((x, y), 0.075*spacing,
                                facecolor='white', edgecolor='k', lw=2, zorder=5, ls=ls))
    for i in range(Lx):
        for j in range(1, Ly + 1):
            x, y = grid_to_xy(i + 0.5, j)
            ax.add_patch(Circle((x, y), 0.075*spacing,
                                facecolor='white', edgecolor='k', lw=2, zorder=5))
        for j in range(1):
            x, y = grid_to_xy(i + 0.5, j)
            ax.add_patch(Circle((x, y), 0.075*spacing,
                                facecolor='white', edgecolor='k', lw=2, zorder=5, ls='--'))

    # highlight one star operator X_s

    if (not pbc):
        for i, j in [[2, 0.5], [2, 1.5], [1.5, 1], [2.5, 1]]:
            x, y = grid_to_xy(i, j)
            ax.add_patch(Circle((x, y), 0.05*spacing,
                                facecolor=magenta, edgecolor=magenta, lw=2, zorder=5))

        px, py = grid_to_xy(2, 1)
        t = ax.text(px - 0.14*spacing, py - 0.05*spacing, r"$A_v$",
                    fontsize=30, color=magenta, weight='bold')
        t.set_bbox(dict(facecolor='white', alpha=1, linewidth=0))
    else:
        for i, j in [[4, 0.5], [4, 1.5], [3.5, 1]]:
            x, y = grid_to_xy(i, j)
            ax.add_patch(Circle((x, y), 0.05*spacing,
                                facecolor=magenta, edgecolor=magenta, lw=2, zorder=5))

        px, py = grid_to_xy(4, 1)
        t = ax.text(px - 0.14*spacing, py - 0.05*spacing, r"$A_v$",
                    fontsize=30, color=magenta, weight='bold')
        t.set_bbox(dict(facecolor='white', alpha=1, linewidth=0))

    # annotate one plaquette with Z_p
    px, py = grid_to_xy(0.5, Ly - 0.5)
    ax.text(px - 0.14*spacing, py - 0.05*spacing, r"$B_p$",
            fontsize=30, color=(0.05, 0.25, 0.6), weight='bold')
    for i, j in [[0, 1.5], [1, 1.5], [0.5, 1], [0.5, 2]]:
        x, y = grid_to_xy(i, j)
        ax.add_patch(Circle((x, y), 0.05*spacing,
                            facecolor=blue, edgecolor=blue, lw=2, zorder=5))

    # tidy
    xmin, ymin = grid_to_xy(0, 0)
    xmax, ymax = grid_to_xy(Lx, Ly)
    ax.set_xlim(xmin - 0.2*spacing, xmax + 0.2*spacing)
    ax.set_ylim(ymin - 0.2*spacing, ymax + 0.2*spacing)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close()
    return savepath


# Example usage
for pbc in [True, False]:
    draw_toric_code(Lx=4, Ly=2, spacing=1.0,
                    savepath="figs/toric_code_geometry_%d.pdf" % pbc, pbc=pbc)
