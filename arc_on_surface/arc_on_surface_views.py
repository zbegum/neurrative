"""
The arc on the surface, as grids: every emotion at one angle, one emotion at
every angle -- with the geodesics drawn beside the route.

Two figures, answering two different questions.

  --mode emotions   2x3, one panel per emotion, one camera. Where does the
                    story's route sit on each of the six terrains?
  --mode angles     2x3, one emotion, six cameras. A 3-D scatter read from a
                    single fixed angle is a tangle -- the arc doubles back
                    constantly -- and rotating is the only way to tell a curve
                    that climbs from one that merely passes in front.

`--curves` picks what is drawn on each terrain:

  geodesic (default)  the exact shortest path along the surface between
                      consecutive windows, one leg each, from the MMP solver.
                      Green. This is the geometry the surface itself imposes,
                      uncluttered by anything else.
  route               the lift of the 2-D narrative arc: straight in (x, y)
                      between consecutive windows, raised onto the surface. Red,
                      and it is NOT a geodesic. Stretches where kernel support
                      falls below the density floor are grey -- the height there
                      is extrapolated from too few paragraphs and there is no
                      drawn terrain beneath it.
  both                the two together, which is the comparison: where green
                      departs from red, the cheapest way between two windows was
                      to bend around a rise the story went straight over. Across
                      Alice the two stay within 1-12% in length, so the
                      departures are small -- which is itself the finding.

Example:

python arc_on_surface/arc_on_surface_views.py --model bge-m3 --mode emotions
python arc_on_surface/arc_on_surface_views.py --model bge-m3 --mode angles --emotion wonder
python arc_on_surface/arc_on_surface_views.py --model bge-m3 --mode emotions --curves both
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
from smooth_common import load_pca
from arc_on_surface import (build_surface, resample, support_at, surface_at,
                            window_bounds, window_points)
from geometry import mesh as gmesh
from windows import DEFAULT_SIZE, DEFAULT_STRIDE
from geometry.geodesic import exact_paths

CMAP = "magma"
ROUTE = "#d1495b"
GEO = "#1baf7a"
UNSUPPORTED = "0.55"

# Six cameras for --mode angles. Two elevations so the relief is read both
# obliquely and from near-overhead, where a curve that hugs a slope separates
# from one that cuts across it.
VIEWS = [(28, -60), (28, 0), (28, 60), (28, 130), (58, -60), (10, -35)]


def geodesic_legs(GX, GY, Z, waypoints, alpha, snap_tol):
  """Exact geodesic between each consecutive pair of windows, in score units.

  Returns (paths, kept) -- paths is a list with None where a leg was dropped,
  either because a waypoint does not sit on the meshed surface or because the
  two ends fell in different components of the masked mesh.
  """
  vertices, faces, _ = gmesh.height_mesh(GX, GY, Z, np.isfinite(Z), alpha)
  vertices, faces, _ = gmesh.largest_component(vertices, faces)
  graph = gmesh.edge_graph(vertices, faces)

  feet, dist = gmesh.snap(waypoints, vertices)
  ok = dist <= snap_tol
  idx = [i for i in range(len(feet) - 1) if ok[i] and ok[i + 1]]
  solved = exact_paths(vertices, faces,
                           [(int(feet[i]), int(feet[i + 1])) for i in idx],
                           graph=graph)

  out = [None] * (len(feet) - 1)
  for i, path in zip(idx, solved):
    if path is not None and alpha > 0:
      # Back to score units so it shares the route's z axis.
      out[i] = np.column_stack([path[:, 0], path[:, 1], path[:, 2] / alpha])
  return out, int(sum(p is not None for p in out))


def draw_panel(ax, GX, GY, Z, dense_xy, z, supported, legs, emotion,
               curves="geodesic"):
  ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                  antialiased=True, alpha=0.45, rstride=3, cstride=3)

  if curves in ("geodesic", "both") and legs:
    seg = [np.asarray(p) for p in legs if p is not None]
    if seg:
      # autolim=False because the legs are ragged -- each geodesic has its own
      # point count, and the auto-scaler tries to stack them into one array.
      # The limits are already set by the surface and the explicit zlim.
      ax.add_collection3d(Line3DCollection(seg, colors=GEO, linewidths=1.6,
                                           alpha=0.95, zorder=6),
                          autolim=False)

  pts = np.column_stack([dense_xy[:, 0], dense_xy[:, 1], z])
  if curves in ("route", "both"):
    # The route, split so unsupported stretches are visibly not a measurement.
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    good = supported[:-1] & supported[1:]
    if good.any():
      ax.add_collection3d(Line3DCollection(seg[good], colors=ROUTE,
                                           linewidths=2.0, zorder=5))
    if (~good).any():
      ax.add_collection3d(Line3DCollection(seg[~good], colors=UNSUPPORTED,
                                           linewidths=1.4, zorder=4))

  ax.scatter(*pts[0], color="black", s=32, marker="o", depthshade=False, zorder=7)
  ax.scatter(*pts[-1], color="black", s=40, marker="X", depthshade=False, zorder=7)
  ax.set_zlim(0.0, 1.0)
  ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
  ax.set_title(emotion, fontsize=10, pad=0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--mode", default="emotions", choices=["emotions", "angles"])
  ap.add_argument("--emotion", default="wonder",
                  help="Which emotion --mode angles rotates around.")
  ap.add_argument("--size", default=DEFAULT_SIZE, type=int)
  ap.add_argument("--stride", default=DEFAULT_STRIDE, type=int)
  ap.add_argument("--hx", default=0.2, type=float)
  ap.add_argument("--hy", default=0.2, type=float)
  ap.add_argument("--alpha", default=gmesh.DEFAULT_ALPHA, type=float)
  ap.add_argument("--oversample", default=4.0, type=float)
  ap.add_argument("--snap-tol", default=1.0, type=float)
  ap.add_argument("--curves", default="geodesic",
                  choices=["geodesic", "route", "both"],
                  help="What to draw on each terrain. Default: the geodesic "
                       "legs alone. 'both' adds the story's own route, for the "
                       "side-by-side comparison.")
  ap.add_argument("--window-point", default="mean", choices=["mean", "medoid"],
                  help="What stands for a window: the centroid of its "
                       "paragraphs (mean), or the most central real paragraph "
                       "in it (medoid). The medoid is a place in the book; the "
                       "mean is not.")
  ap.add_argument("--resolution", default=120, type=int)
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  if len(paragraphs) != len(X_raw):
    raise ValueError(f"pca ({len(X_raw)}) != paragraphs ({len(paragraphs)})")
  if args.mode == "angles" and args.emotion not in emotions:
    raise ValueError(f"Unknown emotion {args.emotion!r}. Available: {emotions}")

  out_dir = paths.out_dir(
    args.book, args.model,
    os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_FROM_TERRAIN),
    {"w": args.size, "s": args.stride, "h": args.hx,
     "hy": args.hy if args.hy != args.hx else None, "a": args.alpha,
     "pt": None if args.window_point == "mean" else args.window_point,
     "df": None if args.density_floor == 5.0 else args.density_floor})

  windows = window_bounds(len(X_raw), args.size, args.stride)
  arc_xy, arc_idx = window_points(X_raw, windows, args.window_point)

  probe = build_surface(X_raw, matrix[:, 0], args.hx, args.hy,
                        args.resolution, args.margin, args.density_floor)
  cell = float(abs(probe[0][0, 1] - probe[0][0, 0]))
  dense_xy, _ = resample(arc_xy, cell / args.oversample)
  print(f"=== {args.mode} grid: {args.book} / {args.model} ===")
  print(f"  {len(windows)} windows -> {len(dense_xy)} curve points\n")

  def prepare(e):
    y_raw = matrix[:, emotions.index(e)]
    GX, GY, Z, xs, ys = build_surface(X_raw, y_raw, args.hx, args.hy,
                                      args.resolution, args.margin,
                                      args.density_floor)
    z = surface_at(xs, ys, X_raw, y_raw, dense_xy, args.hx, args.hy)
    sup, floor = support_at(xs, ys, X_raw, y_raw, dense_xy, args.hx, args.hy,
                            args.density_floor)
    supported = sup >= floor
    legs, kept = ([], 0) if args.curves == "route" else geodesic_legs(
      GX, GY, Z, arc_xy, args.alpha, args.snap_tol * cell)
    print(f"  {e:>10}: {100 * supported.mean():.0f}% of the route is on "
          f"supported terrain; {kept}/{len(windows) - 1} geodesic legs")
    return GX, GY, Z, z, supported, legs

  if args.mode == "emotions":
    fig = plt.figure(figsize=(17, 10))
    for i, e in enumerate(emotions):
      GX, GY, Z, z, supported, legs = prepare(e)
      ax = fig.add_subplot(2, 3, i + 1, projection="3d")
      draw_panel(ax, GX, GY, Z, dense_xy, z, supported, legs, e,
                 args.curves)
    what = "The geodesics" if args.curves == "geodesic" else "The story's route"
    title = (f"{what} on every emotion surface -- {args.book} / "
             f"{args.model}\n")
    name = paths.named("arc_grid", f"emotions_{args.curves}")
  else:
    GX, GY, Z, z, supported, legs = prepare(args.emotion)
    fig = plt.figure(figsize=(17, 10))
    for i, (elev, azim) in enumerate(VIEWS):
      ax = fig.add_subplot(2, 3, i + 1, projection="3d")
      draw_panel(ax, GX, GY, Z, dense_xy, z, supported, legs, args.emotion,
                 args.curves)
      ax.view_init(elev=elev, azim=azim)
      ax.set_title(f"elev {elev}°  azim {azim}°", fontsize=10, pad=0)
    what = "geodesics" if args.curves == "geodesic" else "route"
    title = (f"{args.emotion}: the same {what} from six angles -- {args.book} / "
             f"{args.model}\n")
    name = paths.named("arc_grid", f"angles_{args.emotion}_{args.curves}")

  sub = {
    "geodesic": "green = exact geodesics between consecutive windows -- the "
                "shortest path the surface allows, one per leg",
    "route": "red = the story's route (the lifted narrative arc, NOT a "
             "geodesic); grey = unsupported terrain",
    "both": "green = exact geodesics between consecutive windows; red = the "
            "story's own route (NOT a geodesic); grey = unsupported terrain",
  }[args.curves]
  fig.suptitle(title + sub, fontsize=12)
  fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                      wspace=0.04, hspace=0.10)
  p = os.path.join(out_dir, name)
  fig.savefig(p, dpi=170); plt.close(fig)
  print(f"\n  wrote {p}")

  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="gaussian_nw (anisotropic hx, hy; standardized coords)",
              route="lift of the 2-D arc (not a geodesic)",
              geodesic="geometry.geodesic.exact_paths (exact MMP), per leg")
  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
