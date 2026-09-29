"""The narrative arc in 3-D: windows projected to three components and joined in
reading order.

The same drawing as plot_2d -- gray path, direction cues, O at the opening, X at
the ending, windows colored by a ColorSpec, optional fitted curve -- with a third
projected axis instead of a flat plane, as a matplotlib PNG from one fixed
viewpoint (--elev / --azim).

How to read it: for PCA the third axis is the next direction of variance and the
shape is real geometry (within the variance captured). For UMAP and t-SNE all
three axes are non-metric; only which windows sit near which is signal, not
distances, angles or the overall shape.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

PATH_GRAY = "#8c8c8c"


def direction_cues(coords, n_cues=12, scale=0.04):
  """Tails and vectors for short arrows along the path, pointing forward.

  In 2-D a chevron is a fixed fraction of its segment. In 3-D that makes
  foreshortened segments' arrows vanish, so each cue is instead a fixed fraction
  of the layout's extent -- capped at half the segment, so it never overshoots
  the windows on either side.
  """
  n = len(coords)
  idx = np.linspace(0, n - 2, min(n_cues, n - 1)).round().astype(int)
  seg = coords[idx + 1] - coords[idx]
  seg_len = np.linalg.norm(seg, axis=1, keepdims=True)
  unit = seg / np.clip(seg_len, 1e-12, None)
  extent = np.ptp(coords, axis=0).max()
  length = np.minimum(scale * extent, 0.5 * seg_len)
  mids = (coords[idx] + coords[idx + 1]) / 2.0
  return mids, unit * length


# --------------------------------------------------------------------------
# static (matplotlib)
# --------------------------------------------------------------------------

def draw_path_3d(ax, coords, faint=False):
  ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="0.6",
          linewidth=0.7, alpha=0.35 if faint else 0.75)

  tails, vecs = direction_cues(coords)
  tails = tails - vecs / 2.0  # centre each arrow on its segment midpoint
  ax.quiver(tails[:, 0], tails[:, 1], tails[:, 2],
            vecs[:, 0], vecs[:, 1], vecs[:, 2],
            color="0.45", linewidth=0.9, arrow_length_ratio=0.6,
            normalize=False)

  ax.scatter(*coords[0], s=140, facecolor="none", edgecolor="black",
             linewidths=1.6, marker="o", depthshade=False)
  ax.text(*coords[0], "O", fontsize=9, fontweight="bold",
          ha="center", va="center")
  ax.scatter(*coords[-1], s=160, color="black", marker="X", depthshade=False)


def draw_smooth_curve_3d(ax, fit, show_control=False):
  segments = np.stack([fit.curve[:-1], fit.curve[1:]], axis=1)
  lc = Line3DCollection(segments, cmap="plasma", linewidth=2.6, alpha=0.9)
  lc.set_array(fit.t_norm[:-1])
  ax.add_collection3d(lc)
  if show_control:
    X = fit.control_points
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="black", linestyle="--",
            linewidth=0.7, marker="s", markersize=3, alpha=0.6)


def draw_points_3d(ax, coords, spec, size=40):
  return ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=spec.values,
                    cmap=spec.cmap, vmin=spec.vmin, vmax=spec.vmax, s=size,
                    alpha=0.9, linewidths=0.3, edgecolors="white",
                    depthshade=False)


def draw_arc_3d(fig, ax, proj, spec, title, fit=None, elev=22, azim=-60,
                show_control=False, key=True):
  coords = proj.coords
  draw_path_3d(ax, coords, faint=fit is not None)
  if fit is not None:
    draw_smooth_curve_3d(ax, fit, show_control)
  sc = draw_points_3d(ax, coords, spec)

  ax.set_xlabel(proj.labels[0], fontsize=8)
  ax.set_ylabel(proj.labels[1], fontsize=8)
  ax.set_zlabel(proj.labels[2], fontsize=8)
  ax.tick_params(labelsize=7)
  ax.set_title(title)
  ax.view_init(elev=elev, azim=azim)
  if key:
    fig.colorbar(sc, ax=ax, label=spec.label, shrink=0.6, pad=0.1)


def arc_plot_3d(proj, spec, title, output_path, fit=None, elev=22, azim=-60,
                show_control=False):
  fig = plt.figure(figsize=(9, 8))
  ax = fig.add_subplot(projection="3d")
  draw_arc_3d(fig, ax, proj, spec, title, fit, elev, azim, show_control)
  fig.tight_layout()
  fig.savefig(output_path, dpi=250)
  plt.close(fig)
  print(f"  wrote {output_path}")


def arc_grid_3d(projections, spec, title, output_path, fits=None, elev=22,
                azim=-60, show_control=False):
  n = len(projections)
  fits = fits or [None] * n
  fig = plt.figure(figsize=(7.5 * n, 7))
  for i, (proj, fit) in enumerate(zip(projections, fits)):
    ax = fig.add_subplot(1, n, i + 1, projection="3d")
    draw_arc_3d(fig, ax, proj, spec, proj.name, fit, elev, azim, show_control)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")
