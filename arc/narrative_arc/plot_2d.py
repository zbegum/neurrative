"""The flat narrative arc: windows projected to 2-D and joined in reading order.

A gray polyline through the windows, short chevrons along it for the direction
of travel, the opening marked O and the ending X, and the windows drawn on top
colored by a ColorSpec. Optionally a fitted B-spline (a curves.Fit) colored by
progression, with its control polygon.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def draw_smooth_curve(ax, fit, show_control=False):
  """Overlay the fitted curve, colored by progression so its direction reads."""
  segments = np.stack([fit.curve[:-1], fit.curve[1:]], axis=1)
  lc = LineCollection(segments, cmap="plasma", zorder=2, linewidth=2.6, alpha=0.9)
  lc.set_array(fit.t_norm[:-1])
  ax.add_collection(lc)
  if show_control:
    X = fit.control_points
    ax.plot(X[:, 0], X[:, 1], color="black", linestyle="--", linewidth=0.7,
            marker="s", markersize=3.5, alpha=0.6, zorder=2)


def draw_path(ax, coords, faint=False):
  """The trajectory itself: a gray line through the windows in reading order,
  with a few arrowheads along it so the direction of travel is legible, and the
  opening/ending marked. Colored points are drawn on top by the caller. When a
  fitted curve is overlaid, `faint` drops the raw polyline to a backdrop.
  """
  ax.plot(coords[:, 0], coords[:, 1], color="0.6",
          linewidth=0.5, alpha=0.35 if faint else 0.7, zorder=1)

  # A handful of direction cues, evenly spaced, pointing the way the story
  # moves. Each is a SHORT chevron centered on the segment midpoint, not an
  # arrow spanning the whole segment -- a full-length arrow draws a second line
  # over the segment and makes those segments look heavier than the rest. Same
  # gray and weight as the polyline, so no segment reads as bolder.
  n = len(coords)
  for k in np.linspace(0, n - 2, min(12, n - 1)).round().astype(int):
    mid = (coords[k] + coords[k + 1]) / 2.0
    step = (coords[k + 1] - coords[k]) * 0.12  # 24% of the segment, centered
    ax.annotate("", xy=mid + step, xytext=mid - step, zorder=2,
                arrowprops=dict(arrowstyle="->", color="0.6",
                                lw=0.5, alpha=0.7, mutation_scale=7))

  ax.scatter(*coords[0], s=140, facecolor="none", edgecolor="black",
             linewidths=1.6, marker="o", zorder=4)
  ax.annotate("O", coords[0], fontsize=9, fontweight="bold",
              ha="center", va="center", zorder=5)
  ax.scatter(*coords[-1], s=160, color="black", marker="X", zorder=4)


def draw_points(ax, coords, spec, size=45):
  """The windows, colored by `spec`; returns the mappable for the colorbar."""
  return ax.scatter(coords[:, 0], coords[:, 1], c=spec.values, cmap=spec.cmap,
                    vmin=spec.vmin, vmax=spec.vmax, s=size, alpha=0.9,
                    linewidths=0.3, edgecolors="white", zorder=3)


def draw_arc(fig, ax, proj, spec, title, fit=None, show_control=False, key=True):
  """One complete arc panel on `ax`: path, optional fit, points, and -- unless
  `key` is False, for dense grids -- the colorbar."""
  coords = proj.coords
  draw_path(ax, coords, faint=fit is not None)
  if fit is not None:
    draw_smooth_curve(ax, fit, show_control)
  sc = draw_points(ax, coords, spec)

  ax.set_xlabel(proj.labels[0])
  ax.set_ylabel(proj.labels[1])
  ax.set_title(title)
  if key:
    fig.colorbar(sc, ax=ax, label=spec.label)


def arc_plot(proj, spec, title, output_path, fit=None, show_control=False):
  fig, ax = plt.subplots(figsize=(8, 8))
  draw_arc(fig, ax, proj, spec, title, fit, show_control)
  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)
  print(f"  wrote {output_path}")


def arc_grid(projections, spec, title, output_path, fits=None,
             show_control=False):
  """All projections side by side, so agreement between them can be read.
  `fits` is one Fit (or None) per projection."""
  n = len(projections)
  fits = fits or [None] * n
  fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.6), squeeze=False)
  for ax, proj, fit in zip(axes[0], projections, fits):
    draw_arc(fig, ax, proj, spec, proj.name, fit, show_control)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")
