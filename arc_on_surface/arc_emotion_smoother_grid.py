"""
The mood surface under each smoother and bandwidth -- bandwidth_grid applied to
the emotion-axis surface.

Same layout as surface/kernel/output/<book>/<model>/bandwidth_grid: rows are the four smoothers
from geometry/smoothers.py, columns are that method's bandwidth ladder (small ->
its CV-tuned value). But the panel here is not one emotion's landscape: it is the
combined *mood* surface (the six emotions collapsed onto one labelled axis, as in
emotion_axis_3d.png) with the story arc riding it. So the grid shows how the
choice of smoother and bandwidth reshapes the mood landscape the arc travels.

Each cell fits the repository's per-paragraph mood (surface/mood/mood.py) with
that (smoother, bandwidth) and lifts the arc onto it. loess and local_linear fit
a weighted regression per grid point, so this is the slow grid -- run it in the
background.

Lands in arc_on_surface/output/<book>/<model>/arc_on_surface/<variant>/emotion_axis_smoother_grid.png,
the variant naming the windows and the blend (the swept smoothers and bandwidths
are fixed in METHODS below, so they cannot differ between runs).

Example:

  python arc_on_surface/arc_emotion_smoother_grid.py --book alice_wonderland --model bge-m3
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
from smooth_common import Standardizer, load_pca, fit_grid
from geometry.smoothers import gaussian_nw, epanechnikov_nw, local_linear, loess
from arc_emotion_axis import lifted_arc, DEFAULT_ORDER
from surface.mood import mood as mood_mod
from windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds

# One row per smoother; three columns = that method's bandwidth ladder. The
# ladders match the bandwidth_grid README (Epanechnikov's cutoff needs ~4x the
# Gaussian's decay scale; loess's frac is a neighbour fraction, not a distance).
METHODS = [
  ("gaussian_nw", gaussian_nw,
   [{"hx": 0.1, "hy": 0.1}, {"hx": 0.2, "hy": 0.2}, {"hx": 0.4, "hy": 0.4}],
   ["h=0.1", "h=0.2", "h=0.4"]),
  ("epanechnikov_nw", epanechnikov_nw,
   [{"hx": 0.4, "hy": 0.4}, {"hx": 0.8, "hy": 0.8}, {"hx": 1.6, "hy": 1.6}],
   ["h=0.4", "h=0.8", "h=1.6"]),
  ("local_linear", local_linear,
   [{"hx": 0.4, "hy": 0.4, "degree": 1}, {"hx": 0.8, "hy": 0.8, "degree": 1},
    {"hx": 1.6, "hy": 1.6, "degree": 1}],
   ["h=0.4", "h=0.8", "h=1.6"]),
  ("loess", loess,
   [{"frac": 0.1, "degree": 1}, {"frac": 0.2, "degree": 1}, {"frac": 0.4, "degree": 1}],
   ["frac=0.1", "frac=0.2", "frac=0.4"]),
]


def fit_surface(predict, params, X_raw, y_raw, res, margin, floor):
  """Emotion landscape over the PCA plane for an arbitrary smoother; returns the
  masked grid plus the standardizers, so the arc can be evaluated in the same
  frame."""
  xs, ys = Standardizer().fit(X_raw), Standardizer().fit(y_raw)
  gx, gy, Zz, _, _ = fit_grid(predict, params, xs.forward(X_raw),
                              ys.forward(y_raw), res, margin, floor)
  Z = ys.inverse(Zz)
  raw = xs.inverse(np.column_stack([gx.ravel(), gy.ravel()]))
  return raw[:, 0].reshape(gx.shape), raw[:, 1].reshape(gy.shape), Z, xs, ys


def surface_at(predict, params, xs, ys, X_raw, y_raw, Q):
  vals, _ = predict(xs.forward(X_raw), ys.forward(y_raw), xs.forward(Q), **params)
  return np.clip(ys.inverse(vals), 0.0, 1.0)


def variant_params(args):
  """The knobs that change this grid. METHODS fixes the swept smoothers and
  bandwidths in code, so only the windows and the blend vary."""
  return {"w": args.size, "s": args.stride, "blend": args.blend}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=DEFAULT_SIZE, type=int)
  ap.add_argument("--stride", default=DEFAULT_STRIDE, type=int)
  ap.add_argument("--blend", default="banded", choices=mood_mod.BLENDS)
  ap.add_argument("--arc-smooth", default=1.0, type=float)
  ap.add_argument("--resolution", default=60, type=int)
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  m, order, positions, _ = mood_mod.mood(matrix, emotions,
                                         order=args.order or DEFAULT_ORDER, blend=args.blend)

  windows = window_bounds(n, args.size, args.stride)
  arc_xy = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])

  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          variant_params(args))
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="swept: " + ", ".join(name for name, *_ in METHODS),
              spectrum=order)
  nr, nc = len(METHODS), 3
  print(f"=== emotion-axis smoother grid: {args.book} / {args.model} ===")
  print(f"  {nr} smoothers x {nc} bandwidths, res {args.resolution} "
        f"(loess/local_linear are slow)")

  fig = plt.figure(figsize=(5.0 * nc, 4.4 * nr))
  for r, (name, predict, param_cols, labels) in enumerate(METHODS):
    for c, (params, label) in enumerate(zip(param_cols, labels)):
      GX, GY, mood_grid, xs, ys = fit_surface(predict, params, X_raw, m, args.resolution,
                                              args.margin, args.density_floor)
      height = lambda Q: surface_at(predict, params, xs, ys, X_raw, m, Q)

      ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
      ax.plot_surface(GX, GY, mood_grid, cmap="magma", vmin=0, vmax=1,
                      linewidth=0, antialiased=True, alpha=0.4, rstride=2, cstride=2)
      pts = lifted_arc(arc_xy, height, args.arc_smooth)
      seg = np.stack([pts[:-1], pts[1:]], axis=1)
      lc = Line3DCollection(seg, cmap="plasma", linewidth=2.4, zorder=5)
      lc.set_array(np.linspace(0, 1, len(pts) - 1))
      ax.add_collection3d(lc)
      ax.scatter(*pts[0], color="black", s=40, marker="o", depthshade=False, zorder=7)
      ax.scatter(*pts[-1], color="black", s=50, marker="X", depthshade=False, zorder=7)
      ax.set_zlim(0, 1)
      ax.set_zticks(positions)
      ax.set_zticklabels(order if c == 0 else [], fontsize=7)
      ax.set_xticklabels([]); ax.set_yticklabels([])
      ax.view_init(elev=30, azim=-52)
      ax.set_title(label, fontsize=10, pad=0)
      if c == 0:
        ax.text2D(-0.22, 0.5, name, transform=ax.transAxes, rotation=90,
                  va="center", ha="center", fontsize=12, fontweight="bold")
      print(f"    {name} {label} done")

  fig.suptitle(f"Emotion-axis mood surface by smoother and bandwidth -- "
               f"{args.book} / {args.model}\n"
               "rows: smoother | columns: bandwidth | height = mood, "
               "color = reading order", fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  p = os.path.join(out_dir, "emotion_axis_smoother_grid.png")
  fig.savefig(p, dpi=150); plt.close(fig)
  print(f"  wrote {p}")


if __name__ == "__main__":
  main()
