"""
One grid: the 3-D arc and its three flat views, for several window sizes and
projection methods at once.

  rows      window settings (default 10:5, 20:10, 40:20, 80:40)
  columns   for each method (default PCA, UMAP), four panels:
              3-D     the arc in three components
              1-2     looking down axis 3 (axis 1 across, axis 2 up)
              1-3     looking along axis 2
              2-3     looking along axis 1

The three flat views are the same 3-D points with one coordinate dropped, drawn
on the same axis ranges as the 3-D panel, so a loop that seems to cross itself
in one view can be checked against the other two. For PCA the 1-2 view is
exactly the ordinary 2-D PCA arc; for UMAP and t-SNE it is not (a 3-component
layout is computed differently from a 2-component one).

The fitted curve is a 3-D fit, projected onto each plane with the points. Its
control points are scaled to the number of windows (about half, between 8 and
40) with lambda 0.01 -- the setting the fit sweep found to follow the windows
closely -- so every row is fitted equally tightly whatever its window count.

Examples:

python grid_3d.py
python grid_3d.py --methods pca tsne
python grid_3d.py --windows 20:10 40:20 --methods pca umap tsne
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from narrative_arc import cli, paths
from narrative_arc import windows as W
from narrative_arc.colors import resolve_colors
from narrative_arc.curves import FitOpts, fit_curve
from narrative_arc.plot_2d import draw_arc
from narrative_arc.plot_3d import draw_arc_3d
from narrative_arc.projections import METHODS, project

# (axis a, axis b) for each flat view, in column order after the 3-D panel.
PLANES = ((0, 1), (0, 2), (1, 2))
PANEL = 3.6      # inches per flat panel
WIDE_3D = 1.3    # the 3-D panel is this much wider than a flat one
GAP = 0.08       # blank column between methods, as a fraction of a panel


def grid_fit_opts(n_windows):
  return FitOpts(n_control=int(np.clip(n_windows // 2, 8, 40)), degree=3,
                 lambda_=0.01, solver="dn", max_iter=300)


def perplexity_for(n_windows):
  """min(30, n/4), floored at 5: see sweep.adaptive_perplexity."""
  return float(min(30, max(5, n_windows // 4)))


def limits(proj, fit):
  """Per-axis (lo, hi) covering the points and the fitted curve, padded 6%."""
  pts = proj.coords if fit is None else np.vstack([proj.coords, fit.curve])
  lo, hi = pts.min(axis=0), pts.max(axis=0)
  pad = 0.06 * (hi - lo)
  return lo - pad, hi + pad


def flat_view(proj, fit, a, b):
  """The projection and fit with only axes a and b kept."""
  view = proj._replace(coords=proj.coords[:, [a, b]],
                       labels=(proj.labels[a], proj.labels[b]))
  if fit is None:
    return view, None
  return view, fit._replace(curve=fit.curve[:, [a, b]],
                            control_points=fit.control_points[:, [a, b]])


def short_axis(label):
  """`PC1 (19.1% var)` -> `PC1`, `UMAP-2` -> `UMAP-2`."""
  return label.split(" ")[0]


def main():
  parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  cli.add_target_args(parser)
  parser.add_argument("--windows", nargs="+",
                      default=["10:5", "20:10", "40:20", "80:40"],
                      help="size:stride pairs, one row each.")
  parser.add_argument("--methods", nargs="+", default=["pca", "umap"],
                      choices=METHODS, help="Four columns per method.")
  parser.add_argument("--no-fit", action="store_true",
                      help="Skip the fitted curve.")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--elev", default=22.0, type=float)
  parser.add_argument("--azim", default=-60.0, type=float)
  args = parser.parse_args()

  for book, model in cli.targets(args, parser):
    args.book, args.model = book, model
    draw_grid(args)


def draw_grid(args):
  """One grid figure for args.book / args.model."""
  settings = [tuple(int(x) for x in w.split(":")) for w in args.windows]
  n_rows = len(settings)

  # Per method: [3-D, 1-2, 1-3, 2-3], then a narrow blank column before the next
  # method so the groups read as separate blocks.
  ratios = []
  for m in range(len(args.methods)):
    ratios += [WIDE_3D, 1, 1, 1] + ([GAP] if m < len(args.methods) - 1 else [])
  fig = plt.figure(figsize=(PANEL * sum(ratios), PANEL * n_rows))
  grid = GridSpec(n_rows, len(ratios), figure=fig, width_ratios=ratios,
                  wspace=0.38, hspace=0.42)

  for r, (size, stride) in enumerate(settings):
    series = W.build(args.data_dir, args.book, args.model, size, stride)
    n = len(series)
    print(f"\nw{size} s{stride}: {n} windows")
    spec = resolve_colors(["progression"], series)[0]

    for m, method in enumerate(args.methods):
      proj = project(series.pooled, [method], n_components=3,
                     perplexity=perplexity_for(n), seed=args.seed)[0]
      fit = None if args.no_fit else fit_curve(proj.coords, grid_fit_opts(n))
      lo, hi = limits(proj, fit)
      col0 = 5 * m  # 4 panels + 1 gap per preceding method

      ax = fig.add_subplot(grid[r, col0], projection="3d")
      draw_arc_3d(fig, ax, proj, spec,
                  f"w{size} s{stride} ({n} windows)\n{proj.name} 3-D",
                  fit, elev=args.elev, azim=args.azim, key=False)
      ax.set_xlim(lo[0], hi[0])
      ax.set_ylim(lo[1], hi[1])
      ax.set_zlim(lo[2], hi[2])
      # mplot3d leaves a wide margin inside its box; zoom in to use it.
      ax.set_box_aspect(None, zoom=1.25)
      # The flat views carry the numbers; here tick labels only collide with
      # the neighbouring panel, so keep just the axis names.
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_zticklabels([])
      ax.set_xlabel(short_axis(proj.labels[0]), fontsize=8, labelpad=-10)
      ax.set_ylabel(short_axis(proj.labels[1]), fontsize=8, labelpad=-10)
      ax.set_zlabel(short_axis(proj.labels[2]), fontsize=8, labelpad=-10)
      ax.title.set_fontsize(10)

      for p, (a, b) in enumerate(PLANES):
        view, view_fit = flat_view(proj, fit, a, b)
        ax = fig.add_subplot(grid[r, col0 + p + 1])
        draw_arc(fig, ax, view, spec,
                 f"{proj.name}: {short_axis(proj.labels[a])} vs "
                 f"{short_axis(proj.labels[b])}",
                 view_fit, key=False)
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontsize(7)
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(10)

  names = " / ".join(proj_name.upper() if proj_name == "pca" else
                     {"umap": "UMAP", "tsne": "t-SNE"}[proj_name]
                     for proj_name in args.methods)
  fig.suptitle(
    f"Narrative arc in 3-D and its flat views -- {args.book} / {args.model} -- "
    f"{names}\n"
    "each row: one window size; for each method: 3-D, then the same points "
    "seen along each axis. Color = reading order (dark -> light), "
    "O = opening, X = ending", fontsize=15, y=0.995)
  fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.05)

  out = paths.out_dir(args.output_dir, args.book, args.model, "grids")
  stem = (f"grid3d_views_{'-'.join(args.methods)}_"
          + "_".join(f"w{s}s{t}" for s, t in settings))
  path = os.path.join(out, stem + ".png")
  fig.savefig(path, dpi=110)
  plt.close(fig)
  paths.stamp(out, __file__, args)
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
