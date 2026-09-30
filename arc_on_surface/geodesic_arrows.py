"""
Geodesics pointing toward an emotion.

For every consecutive pair of paragraphs i -> i+1 whose score rises, draw the
path from i to i+1 -- but along the emotion landscape rather than straight
across the latent plane. Climbing costs distance, so the path bends around the
terrain the emotion defines.

The surface is Gaussian Nadaraya-Watson at hx = hy = 0.2 (see the bandwidth_grid
README for why: it is the only smoother that is differentiable everywhere, leaves
no holes in the mesh, and cannot leave the [0, 1] score range).

--alpha is vertical exaggeration and it decides whether there is anything to see.
Larger alpha exaggerates the relief, so the geodesics bend more; the sweep shows
a row of increasing alphas side by side.

The exact solver cannot take a perfectly flat mesh (alpha = 0): a coplanar
regular grid is a degeneracy for the MMP wavefront and it either aborts or
returns wrong paths, so alpha = 0 is left out of the sweep and there is no flat control.

Example:

python arc_on_surface/geodesic_arrows.py --model bge-m3 --emotions wonder
python arc_on_surface/geodesic_arrows.py --model bge-m3 --emotions wonder --alpha-sweep
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


# Aliased: "paths" is already a local here (the geodesic polylines), and a
# module of that name would be shadowed inside main().
# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths as out_paths
from smooth_common import Standardizer, add_common_args, fit_grid, prepare
from geometry.geodesic import (exact_paths, polyline_deviation,
                               polyline_length, rising_pairs)
from geometry.mesh import DEFAULT_ALPHA, edge_graph, height_mesh, largest_component, snap
from geometry.smoothers import gaussian_nw

CMAP = "viridis"


def build_surface(X_raw, y_raw, args):
  """Fit the field, then return it in raw PCA coordinates.

  The smoother is tuned in standardized units, so it is fitted there; the mesh
  is built in raw PCA coordinates because that is the honest metric -- PCA is a
  linear projection, so its distances mean something, and a geodesic is a
  statement about distance.
  """
  xs, ys = Standardizer().fit(X_raw), Standardizer().fit(y_raw)
  gx, gy, Zz, mask, _ = fit_grid(
    gaussian_nw, {"hx": args.hx, "hy": args.hy},
    xs.forward(X_raw), ys.forward(y_raw),
    args.resolution, args.margin, args.density_floor,
  )
  Z = ys.inverse(Zz)
  raw = xs.inverse(np.column_stack([gx.ravel(), gy.ravel()]))
  return raw[:, 0].reshape(gx.shape), raw[:, 1].reshape(gy.shape), Z, mask


def geodesics(GX, GY, Z, mask, X_raw, idx, alpha):
  """Mesh at this alpha, then trace i -> i+1 for each rising pair."""
  vertices, faces, _ = height_mesh(GX, GY, Z, mask, alpha)
  # The exact solver aborts on a disconnected mesh; keep the biggest piece.
  vertices, faces, _ = largest_component(vertices, faces)
  graph = edge_graph(vertices, faces)
  feet, snap_dist = snap(X_raw, vertices)

  pairs = [(int(feet[i]), int(feet[i + 1])) for i in idx]
  paths = exact_paths(vertices, faces, pairs, graph=graph)
  return vertices, faces, paths, snap_dist


def draw(ax, GX, GY, Z, paths, deltas, top, alpha, title):
  ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                  antialiased=True, alpha=0.55, rstride=3, cstride=3)

  # Strongest rises first, and only `top` of them: ~300 arrows is a hairball.
  order = np.argsort(deltas)[::-1][:top]
  for k in order:
    p = paths[k]
    if p is None or len(p) < 2:
      continue
    # Undo the exaggeration for drawing so the path sits on the plotted
    # surface, whose z axis is in score units.
    z = p[:, 2] / alpha if alpha > 0 else np.zeros(len(p))
    ax.plot(p[:, 0], p[:, 1], z, color="#e34948", linewidth=1.0, alpha=0.9)
    # The head marks i+1: the arrow points toward the emotion.
    ax.scatter(p[-1, 0], p[-1, 1], z[-1], color="#e34948", s=8, depthshade=False)

  ax.set_zlim(0.0, 1.0)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_zticklabels([])
  ax.set_title(title, fontsize=10, pad=0)


def report(paths, alpha):
  dev = np.array([polyline_deviation(p) for p in paths if p is not None])
  lens = np.array([polyline_length(p) for p in paths if p is not None])
  dead = sum(p is None for p in paths)
  print(f"  alpha {alpha:>5}: bend median {np.median(dev):.4f} "
        f"p90 {np.percentile(dev, 90):.4f} max {dev.max():.4f} | "
        f"path len median {np.median(lens):.3f} | failed {dead}")
  return dev


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.add_argument("--hx", default=0.2, type=float)
  parser.add_argument("--hy", default=0.2, type=float)
  parser.add_argument("--alpha", default=DEFAULT_ALPHA, type=float,
                      help="Vertical exaggeration of the landscape.")
  parser.add_argument("--alpha-sweep", action="store_true",
                      help="Draw a row of alphas: 1, 3 and 10.")
  parser.add_argument("--top", default=40, type=int,
                      help="How many of the strongest rises to draw.")
  parser.add_argument("--min-delta", default=0.0, type=float,
                      help="Ignore rises no larger than this (the scores are "
                           "quantized, so tiny rises may be rounding).")
  args = parser.parse_args()

  ctx = prepare(args)
  out_dir = out_paths.out_dir(args.book, args.model, out_paths.GEODESICS)
  out_paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
                  estimator="gaussian_nw (anisotropic hx, hy; standardized coords)",
                  emotions=ctx["selected"])

  alphas = [1.0, 3.0, 10.0] if args.alpha_sweep else [args.alpha]

  for emotion in ctx["selected"]:
    y_raw = ctx["matrix"][:, ctx["emotions"].index(emotion)]
    X_raw = ctx["X_raw"]
    print(f"\n{emotion}: gaussian_nw hx={args.hx} hy={args.hy}")

    GX, GY, Z, mask = build_surface(X_raw, y_raw, args)
    print(f"  surface: {int(mask.sum())}/{mask.size} cells | "
          f"relief {np.nanmin(Z):.2f} .. {np.nanmax(Z):.2f}")

    idx, deltas = rising_pairs(y_raw, args.min_delta)
    print(f"  rising pairs: {len(idx)} of {len(y_raw) - 1}")

    fig, axes = plt.subplots(1, len(alphas), figsize=(6.0 * len(alphas), 5.6),
                             subplot_kw={"projection": "3d"}, squeeze=False)

    for c, alpha in enumerate(alphas):
      vertices, faces, paths, snap_dist = geodesics(GX, GY, Z, mask, X_raw, idx, alpha)
      if c == 0:
        far = int((snap_dist > np.diff(GX[0, :2])[0] * 2).sum())
        print(f"  mesh: {len(vertices)} vertices, {len(faces)} faces | "
              f"{far} paragraphs snapped from outside the masked region")
      report(paths, alpha)
      draw(axes[0][c], GX, GY, Z, paths, deltas, args.top, alpha,
           f"alpha = {alpha:g}")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.0)
    tag = "sweep" if args.alpha_sweep else f"a{args.alpha:g}"
    path = os.path.join(out_dir, f"geodesic_{emotion}_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")

  print(f"\nDone.\n{out_dir}")


if __name__ == "__main__":
  main()
