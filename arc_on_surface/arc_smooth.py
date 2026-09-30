"""
Smooth the narrative arc with distance-based curve smoothing (Pawellek et al.
2024). This replaces the earlier control-point geodesic fit (the retired
arc_geodesic.py): closeness to the arc is now one continuous parameter tau with a
provable bound, not a hand-tuned count of control points.

The narrative arc is a noisy polyline in the PCA (semantic) plane -- exactly the
"initial curve" the paper smooths. geometry/curve_smoothing.py implements their
method for our flat-plane case (see that module's docstring). Here we:

  * build the initial arc from the windowed embeddings,
  * smooth it across a sweep of the tolerance parameter tau (their Figure 5),
  * draw the sweep over the mood/emotion landscape (top view), and
  * lift one chosen smoothing onto the emotion terrain in 3-D.

tau = 0 gives the plain geodesic of the plane (a straight line between the book's
opening and closing); larger tau holds the smoothed curve closer to the original
arc. Unlike the old control-point fit, closeness is one continuous knob, not a
count of hand-placed points, and the result provably stays near the arc.

Example:

  python arc_on_surface/arc_smooth.py --book alice_wonderland --model bge-m3
  python arc_on_surface/arc_smooth.py --emotion wonder --taus 0 0.1 0.3 0.8
"""

import argparse
import os
import sys
from types import SimpleNamespace

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
from arc_on_surface import build_surface
from arc_emotion_axis import sample_surface, DEFAULT_ORDER
from geometry.mesh import DEFAULT_ALPHA
from surface.mood import mood as mood_mod, surface as mood_surface
from windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds
from geometry.curve_smoothing import smooth_curve


def _medoid(pts):
  """The most central actual point of a window: the one minimising total
  distance to the others. Unlike the mean it is a real paragraph, so it lies
  where the data (and hence the surface) actually is."""
  if len(pts) == 1:
    return pts[0]
  d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2).sum(axis=1)
  return pts[int(d.argmin())]


def build_emotion_surface(X_raw, y, args):
  """A single emotion's landscape (score as height) over the PCA grid."""
  GX, GY, Z, _xs, _ys = build_surface(X_raw, y, args.hx, args.hy, args.resolution,
                                      args.margin, args.density_floor)
  return GX, GY, Z


def build_mood_surface(X_raw, matrix, emotions, args):
  """The repository's mood surface (surface/mood) over the PCA grid."""
  m, *_ = mood_mod.mood(matrix, emotions, order=args.order or DEFAULT_ORDER)
  GX, GY, Z, _ = mood_surface.fit(X_raw, m, args.mood_h, resolution=args.resolution)
  return GX, GY, Z


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--emotion", default=None,
                  help="Draw over one emotion's landscape; default combined mood.")
  ap.add_argument("--size", default=DEFAULT_SIZE, type=int)
  ap.add_argument("--stride", default=DEFAULT_STRIDE, type=int)
  ap.add_argument("--taus", nargs="+", type=float,
                  default=[0.0, 0.3, 0.6, 1.0, 1.6],
                  help="Tolerance sweep: 0 = straight geodesic, large = hugs arc.")
  ap.add_argument("--lift-tau", default=0.6, type=float,
                  help="Which tau to lift onto the terrain in the 3-D view.")
  ap.add_argument("--representative", default="mean", choices=["mean", "medoid"],
                  help="Per-window point: 'mean' (as everywhere else) or 'medoid', "
                       "the most central real paragraph.")
  # penalty / solver knobs (sensible defaults; rarely need changing)
  ap.add_argument("--resolution", default=260, type=int)
  ap.add_argument("--blur", default=1.2, type=float)
  # surface (drawing) knobs -- same protocol as arc_on_surface
  ap.add_argument("--hx", default=0.2, type=float)
  ap.add_argument("--hy", default=0.2, type=float)
  ap.add_argument("--mood-h", default=0.2, type=float,
                  help="Bandwidth of the mood surface (surface/mood).")
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  ap.add_argument("--alpha", default=DEFAULT_ALPHA, type=float,
                  help="Vertical exaggeration of the terrain in the 3-D view.")
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  # The initial curve: one representative point per window, in the PCA plane.
  windows = window_bounds(n, args.size, args.stride)
  if args.representative == "mean":
    arc = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])
  else:
    arc = np.array([_medoid(X_raw[s:e]) for s, e in windows])

  label = args.emotion or "mood"
  print(f"=== distance-based arc smoothing on the {label} surface: "
        f"{args.book} / {args.model} ===")
  print(f"  {n} paragraphs -> {len(windows)} windows (size {args.size} "
        f"stride {args.stride}); tau sweep {args.taus}")

  # The landscape, only for drawing. sfields carries what build_*_surface needs.
  sargs = SimpleNamespace(hx=args.hx, hy=args.hy, mood_h=args.mood_h, order=args.order,
                          resolution=args.resolution, margin=args.margin,
                          density_floor=args.density_floor)
  if args.emotion:
    GX, GY, Zsurf = build_emotion_surface(
      X_raw, matrix[:, emotions.index(args.emotion)], sargs)
    zlabel = args.emotion
  else:
    GX, GY, Zsurf = build_mood_surface(X_raw, matrix, emotions, sargs)
    zlabel = "mood"

  # Smooth the arc for each tau.
  results = {}
  for tau in args.taus:
    r = smooth_curve(arc, tau, resolution=args.resolution, blur=args.blur)
    results[tau] = r
    print(f"  tau={tau:<5}: length {r['length']:.3f} (arc {r['length0']:.3f}), "
          f"max deviation {r['max_dev']:.4f}")

  # The terrain the arc is smoothed over depends on the window, the stride and
  # the bandwidth, and on which emotion (or the mood blend) supplies the height
  # -- none of which the filenames beyond the tag record.
  tag = args.emotion or "mood"
  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_CURVE_SMOOTHING),
                          {"w": args.size, "s": args.stride,
                           "h": args.hx if args.emotion else args.mood_h,
                           "z": tag})
  paths.stamp(out_dir, __file__, args, stack="geometry.curve_smoothing",
              estimator="smooth_curve (distance-based, Pawellek 2024)",
              surface="gaussian_nw (geometry.smoothers) for the drawn terrain")

  # --- figure 1: the tau sweep over the landscape (top view) -------------------
  fig, ax = plt.subplots(figsize=(9, 8.2))
  cf = ax.contourf(GX, GY, Zsurf, levels=18, cmap="magma", alpha=0.85)
  fig.colorbar(cf, ax=ax, shrink=0.7, label=zlabel)
  ax.plot(arc[:, 0], arc[:, 1], color="0.25", lw=1.0, alpha=0.7,
          label="narrative arc (initial)", zorder=3)
  ramp = plt.get_cmap("winter")
  for i, tau in enumerate(args.taus):
    c = results[tau]["curve"]
    col = ramp(i / max(1, len(args.taus) - 1))
    ax.plot(c[:, 0], c[:, 1], color=col, lw=2.4, zorder=4,
            label=f"tau = {tau:g}")
  ax.scatter(*arc[0], color="black", s=90, marker="o", zorder=6)
  ax.scatter(*arc[-1], color="black", s=110, marker="X", zorder=6)
  ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
  ax.set_aspect("equal", adjustable="box")
  ax.legend(loc="best", fontsize=8, framealpha=0.9)
  ax.set_title(f"Distance-based smoothing of the narrative arc (Pawellek 2024)\n"
               f"tau: 0 = straight geodesic -> large = hugs the arc  |  "
               f"{label} landscape", fontsize=11)
  fig.tight_layout()
  p = os.path.join(out_dir, f"arc_smooth_{tag}_sweep.png")
  fig.savefig(p, dpi=170); plt.close(fig); print(f"  wrote {p}")

  # --- figure 2: one smoothing lifted onto the terrain -------------------------
  lift_tau = min(args.taus, key=lambda t: abs(t - args.lift_tau))
  smooth = results[lift_tau]["curve"]

  # Unproject onto the surface: read the surface height at each point. Points
  # over a masked (low-density) hole get no height from interpolation, so we
  # snap those to the nearest supported surface cell -- every point lands ON the
  # surface rather than floating.
  from scipy.spatial import cKDTree
  smask = np.isfinite(Zsurf)
  svalid_xy = np.column_stack([GX[smask], GY[smask]])
  svalid_z = Zsurf[smask]
  stree = cKDTree(svalid_xy)

  def lift(xy):
    z = sample_surface(GX, GY, Zsurf, xy[:, 0], xy[:, 1])
    gap = ~np.isfinite(z)
    if gap.any():
      z[gap] = svalid_z[stree.query(xy[gap])[1]]
    return np.column_stack([xy[:, 0], xy[:, 1], args.alpha * z])

  arc3d, smooth3d = lift(arc), lift(smooth)
  lz = 0.03 * args.alpha                          # small lift so lines clear the surface
  fig = plt.figure(figsize=(12, 9))
  ax3 = fig.add_subplot(111, projection="3d")
  # muted grey terrain so the progression-coloured curve is the focus. Draw at
  # full resolution (stride 1) so the drawn surface matches where the dots are
  # sampled from -- a coarser stride would leave the dots hovering over quads
  # that don't reach them.
  ax3.plot_surface(GX, GY, args.alpha * Zsurf, color="0.7", linewidth=0,
                   antialiased=True, alpha=0.25, rstride=1, cstride=1,
                   shade=True)
  # the timeseries points we smooth over -- small dots, no connecting edges,
  # coloured by reading progression (dark = start, bright = end).
  prog = np.linspace(0, 1, len(arc3d))
  ax3.scatter(arc3d[:, 0], arc3d[:, 1], arc3d[:, 2] + lz, c=prog, cmap="plasma",
              s=9, depthshade=False, edgecolors="none", zorder=6)
  # the smoothed curve -- thin, coloured by the same progression.
  seg = np.stack([smooth3d[:-1], smooth3d[1:]], axis=1)
  lc = Line3DCollection(seg + [0, 0, lz], cmap="plasma", linewidth=1.6, zorder=7)
  lc.set_array(np.linspace(0, 1, len(seg)))
  ax3.add_collection3d(lc)
  ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel(zlabel)
  ax3.set_xticklabels([]); ax3.set_yticklabels([])
  ax3.view_init(elev=32, azim=-52)
  ax3.set_title(f"Smoothed narrative arc on the {label} terrain "
                f"(tau = {lift_tau:g})", fontsize=12)
  fig.tight_layout()
  p = os.path.join(out_dir, f"arc_smooth_{tag}_3d.png")
  fig.savefig(p, dpi=170); plt.close(fig); print(f"  wrote {p}")

  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
