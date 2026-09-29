"""Draw the mood surface, the 2D plane, and an arc over the surface."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

SURFACE_CMAP = "magma"
ARC_CMAP = "winter"


def draw_surface(ax, GX, GY, Z, order, positions, alpha=0.9):
    ax.plot_surface(GX, GY, Z, cmap=SURFACE_CMAP, vmin=0, vmax=1, linewidth=0,
                    antialiased=True, alpha=alpha, rstride=2, cstride=2)
    ax.set_zlim(0, 1)
    ax.set_zticks(positions)
    ax.set_zticklabels(order, fontsize=7)
    ax.set_xticklabels([]); ax.set_yticklabels([])


def draw_arc(ax, points, reading_pos, n_paragraphs, lift=0.02, width=1.4):
    """A polyline over the surface, coloured by reading position."""
    pts = np.column_stack([points[:, 0], points[:, 1], points[:, 2] + lift])
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=ARC_CMAP, linewidths=width, zorder=6)
    lc.set_array(np.asarray(reading_pos[:-1], dtype=float))
    lc.set_clim(0, n_paragraphs - 1)
    ax.add_collection3d(lc, autolim=False)
    ax.scatter(*pts[0], c="black", s=26, marker="o", depthshade=False, zorder=8)
    ax.scatter(*pts[-1], c="black", s=32, marker="X", depthshade=False, zorder=8)


def draw_runs(ax, runs, n_paragraphs, lift=0.02, width=1.4):
    """Draw geodesic runs (each a points/reading-pos pair) with start/end marks."""
    first = last = None
    for P, t in runs:
        pts = np.column_stack([P[:, 0], P[:, 1], P[:, 2] + lift])
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(seg, cmap=ARC_CMAP, linewidths=width, zorder=6)
        lc.set_array(t[:-1])
        lc.set_clim(0, n_paragraphs - 1)
        ax.add_collection3d(lc, autolim=False)
        if first is None:
            first = pts[0]
        last = pts[-1]
    if first is not None:
        ax.scatter(*first, c="black", s=26, marker="o", depthshade=False, zorder=8)
        ax.scatter(*last, c="black", s=32, marker="X", depthshade=False, zorder=8)


def reading_bar(fig, n_paragraphs):
    sm = plt.cm.ScalarMappable(cmap=ARC_CMAP,
                               norm=plt.Normalize(0, n_paragraphs - 1))
    sm.set_array([])
    cax = fig.add_axes([0.34, 0.08, 0.32, 0.012])
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label(
        "reading position (paragraph)", fontsize=9)


def plane(coords, mood, order, positions, title, plane_name):
    """The 2D layout, coloured by reading order and by mood, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    lab = ("PC1", "PC2") if plane_name == "pca" else ("UMAP-1", "UMAP-2")

    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)),
                         cmap="plasma", s=12, linewidths=0)
    fig.colorbar(sc, ax=axes[0], fraction=0.046).set_label("reading position")
    axes[0].set_title("by reading order")

    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=mood, cmap=SURFACE_CMAP,
                          vmin=0, vmax=1, s=12, linewidths=0)
    cb = fig.colorbar(sc2, ax=axes[1], fraction=0.046, ticks=list(positions))
    cb.ax.set_yticklabels(order)
    cb.set_label("mood")
    axes[1].set_title("by mood")

    for ax in axes:
        ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
