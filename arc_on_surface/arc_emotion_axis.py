"""
One interpretable emotion axis: the six emotions collapsed into a single height,
so following the story arc *is* watching the emotion change.

arc_on_surface.py lifts the arc onto six separate emotion surfaces. Here the six
are combined into one "mood" coordinate, laid out as a fixed spectrum:

    sadness -> danger -> confusion -> curiosity -> wonder -> humor
     (heavy / negative)                              (light / positive)

Each emotion sits at a fixed position on that line. A point's height is the
score-weighted position of its emotional blend:

    z = sum_e position(e) * score(e) / sum_e score(e)

so a window dominated by wonder sits high, one dominated by sadness sits low, and
a mix lands in between. The z-axis is ticked with the emotion names at their
positions, so the emotion is read straight off the height -- and as the arc rises
and falls, you are watching the story's mood move. The marker color carries the
*dominant* emotion, so height (blend) and color (winner) are legible together.

The height comes from the emotion *surface* at the arc's location (the arc-on-
surface premise: the arc lives in the surface's own PCA plane), so this is the
mood the landscape assigns to the story's route. Only PCA -- the plane must be
metric for a point to have one height.

The spectrum order is a deliberate, adjustable choice (--order); it sets what
"up" means. Everything lands in arc_on_surface/output/<book>/<model>/arc_on_surface/<variant>/,
the variant naming the window, bandwidth, temperature and normalization used.

Example:

  python arc_on_surface/arc_emotion_axis.py --book alice_wonderland --model bge-m3
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
from geometry.smoothers import nadaraya_watson
from arc_on_surface import window_bounds, build_surface, surface_at
# One definition of mood for the whole repository: surface/mood/mood.py.
from surface.mood.mood import EMOTION_COLOR, SPECTRUM, normalizer, softmax_position as mood

# Valence order, heavy/negative (bottom) to light/positive (top). Adjustable via
# --order; anything not named is appended so the run never silently drops one.
DEFAULT_ORDER = SPECTRUM



def nan_blur(a, sigma):
  """Gaussian blur of a 2-D field that ignores (and preserves) NaN cells.

  Blur the values and the support mask separately and divide, so NaNs neither
  bleed in as zeros nor spread; cells with no nearby support stay NaN.
  """
  if sigma <= 0:
    return a
  from scipy.ndimage import gaussian_filter
  m = np.isfinite(a).astype(np.float64)
  num = gaussian_filter(np.where(m > 0, a, 0.0), sigma)
  den = gaussian_filter(m, sigma)
  out = num / np.where(den > 0, den, np.nan)
  return np.where(den > 0.25, out, np.nan)


def sample_surface(GX, GY, Z, qx, qy):
  """Height of the drawn surface Z at query points (qx, qy).

  The arc must lie *on* the surface it is drawn over, so its height is read from
  the same (blurred, masked) grid the surface is rendered from -- not recomputed
  separately, which leaves it floating. Points over a masked (NaN) cell come back
  NaN so the caller can fall back. GX, GY are an xy-meshgrid; Z[i, j] is the value
  at (GX[0, j], GY[i, 0]).
  """
  from scipy.interpolate import RegularGridInterpolator
  xvals, yvals = GX[0, :], GY[:, 0]
  xi, yi = np.argsort(xvals), np.argsort(yvals)
  f = RegularGridInterpolator((yvals[yi], xvals[xi]), Z[np.ix_(yi, xi)],
                              bounds_error=False, fill_value=np.nan)
  return f(np.column_stack([qy, qx]))


def smooth1d(y, sigma):
  """Gaussian smoothing over the sequence index -- a readable emotional line
  instead of window-to-window jitter. sigma is in windows; 0 leaves y untouched.

  This is the same Nadaraya-Watson average the emotion surface is built from,
  in one dimension over the index rather than two over the plane, so it calls
  the same estimator instead of spelling the kernel out again.
  """
  y = np.asarray(y, dtype=np.float64)
  if sigma <= 0:
    return y
  idx = np.arange(len(y), dtype=np.float64).reshape(-1, 1)
  return nadaraya_watson(idx, y, idx, sigma)[0]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  # 0.15, not the geodesic work's 0.2: below the CV-tuned 0.4, in the "noise ->
  # structure" band the bandwidth grid maps out. 0.2 is a gentle swell, 0.1 is
  # spiky annotation noise; 0.15 buys visible relief so the mood is not flat.
  ap.add_argument("--hx", default=0.15, type=float)
  ap.add_argument("--hy", default=0.15, type=float)
  ap.add_argument("--resolution", default=120, type=int)
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  ap.add_argument("--order", nargs="+", default=None,
                  help="Emotion order, bottom (negative) to top (positive).")
  ap.add_argument("--temp", default=0.3, type=float,
                  help="Softmax temperature: small snaps to the dominant emotion "
                       "(big mood swings), large averages (flatter).")
  ap.add_argument("--norm", default="rank", choices=["rank", "zscore", "minmax"],
                  help="Per-emotion normalization before blending (rank spreads most).")
  ap.add_argument("--spread", default="fill", choices=["absolute", "fill"],
                  help="absolute: y spans the full sadness..humor line. fill: zoom "
                       "y to the mood's actual range so the changes fill the plot.")
  ap.add_argument("--smooth", default=1.5, type=float,
                  help="Gaussian smoothing (in windows) of the mood line; 0 = raw.")
  ap.add_argument("--surface-smooth", default=1.5, type=float,
                  help="NaN-aware blur (grid cells) of the 3-D mood surface; 0 = raw.")
  ap.add_argument("--arc-smooth", default=3.0, type=float,
                  help="Smoothing (in windows) of the 3-D arc path so it reads as a "
                       "clean curve over the landscape instead of a hairball.")
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)

  order = args.order or [e for e in DEFAULT_ORDER if e in emotions]
  order += [e for e in emotions if e not in order]          # never drop one
  pos = np.linspace(0.0, 1.0, len(order))
  pos_of = {e: p for e, p in zip(order, pos)}
  print(f"=== emotion axis: {args.book} / {args.model} ===")
  print("  spectrum (bottom->top): " +
        " -> ".join(f"{e}({p:.2f})" for e, p in zip(order, pos)))

  # The mood is a blend, so the temperature and the normalization change the
  # height as much as the window and the bandwidth do; all five belong in the
  # directory name, since emotion_axis_3d.png records none of them.
  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          {"w": args.size, "s": args.stride, "h": args.hx,
                           "t": args.temp, "norm": args.norm})
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="gaussian_nw (anisotropic hx, hy; standardized coords)",
              spectrum=order)

  windows = window_bounds(n, args.size, args.stride)
  centers = np.array([(s + e - 1) / 2.0 for s, e in windows])
  arc_xy = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])

  # Per-emotion terrain (grid) and arc height, in the fixed spectrum order.
  grid_Z, arc_h, own_h, GX, GY = [], [], [], None, None
  for e in order:
    y = matrix[:, emotions.index(e)]
    GX, GY, Z, xs, ys = build_surface(X_raw, y, args.hx, args.hy,
                                      args.resolution, args.margin, args.density_floor)
    grid_Z.append(Z)
    arc_h.append(surface_at(xs, ys, X_raw, y, arc_xy, args.hx, args.hy))
    own_h.append(np.array([y[s:e2].mean() for s, e2 in windows]))

  positions = np.array([pos_of[e] for e in order])
  # Put every emotion on a common footing first (see normalizer), then blend, so
  # the mood reflects which emotion is *relatively* elevated, not its raw level.
  tf = {e: normalizer(matrix[:, emotions.index(e)], args.norm) for e in order}
  grid_n = np.stack([np.where(np.isnan(Z), np.nan, tf[e](Z))
                     for e, Z in zip(order, grid_Z)])
  arc_n = np.stack([tf[e](h) for e, h in zip(order, arc_h)])
  own_n = np.stack([tf[e](h) for e, h in zip(order, own_h)])

  # Only draw the mood surface where every emotion is supported; a cell backed by
  # one or two emotions blends erratically and shows up as boundary spikes.
  all_supported = np.all(np.isfinite(grid_n), axis=0)
  mood_grid = np.where(all_supported, mood(grid_n, positions, args.temp), np.nan)
  # The combined surface is a competition of six fields, so it comes out spiky;
  # a NaN-aware blur turns it into a readable landscape without going flat.
  mood_grid = nan_blur(mood_grid, args.surface_smooth)
  mood_arc = mood(arc_n, positions, args.temp)
  mood_own = mood(own_n, positions, args.temp)
  # Dominant = the emotion most elevated (in normalized terms) here; matches the
  # mood movement, where raw argmax would always read curiosity/wonder.
  dominant = np.array(order)[np.nanargmax(np.where(np.isfinite(arc_n), arc_n, -np.inf), axis=0)]
  dom_colors = [EMOTION_COLOR.get(e, "#555555") for e in dominant]
  # Dominant for the reading chart, whose primary line is the windows' own emotion.
  dom_own = np.array(order)[np.argmax(own_n, axis=0)]
  dom_own_colors = [EMOTION_COLOR.get(e, "#555555") for e in dom_own]
  print(f"  norm={args.norm} temp={args.temp} spread={args.spread} smooth={args.smooth}")

  print(f"  arc mood range {mood_arc.min():.2f}..{mood_arc.max():.2f} "
        f"(0=neg pole, 1=pos pole)")

  # --- figure 1: the mood surface with the arc riding it (3-D) ------------------
  # Smooth the arc PATH (x, y and height together) so it reads as one clean curve
  # flowing over the landscape rather than a window-to-window hairball, and lift
  # it a hair above the surface so it is never buried inside it.
  ax_s = smooth1d(arc_xy[:, 0], args.arc_smooth)
  ay_s = smooth1d(arc_xy[:, 1], args.arc_smooth)
  # Read the height off the drawn surface at the arc's plotted (x, y) so it rides
  # on the terrain; fall back to the arc's own mood only where the surface is masked.
  az_on = sample_surface(GX, GY, mood_grid, ax_s, ay_s)
  az_s = np.where(np.isfinite(az_on), az_on, smooth1d(mood_arc, args.arc_smooth))
  pts = np.column_stack([ax_s, ay_s, az_s + 0.01])

  fig = plt.figure(figsize=(12, 9))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot_surface(GX, GY, mood_grid, cmap="magma", vmin=0, vmax=1,
                  linewidth=0, antialiased=True, alpha=0.4, rstride=2, cstride=2)
  seg = np.stack([pts[:-1], pts[1:]], axis=1)
  lc = Line3DCollection(seg, cmap="plasma", linewidth=4.0, zorder=5)
  lc.set_array(np.linspace(0, 1, len(pts) - 1))
  ax.add_collection3d(lc)
  # Sparse markers (every 4th) carry the dominant emotion without clutter.
  s = slice(None, None, 4)
  ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], c=[dom_colors[i] for i in range(0, len(pts), 4)],
             s=24, depthshade=False, zorder=6, edgecolors="white", linewidths=0.4)
  ax.scatter(*pts[0], color="black", s=90, marker="o", depthshade=False, zorder=7)
  ax.scatter(*pts[-1], color="black", s=110, marker="X", depthshade=False, zorder=7)
  ax.set_zlim(0, 1)
  ax.set_zticks(positions)
  ax.set_zticklabels(order, fontsize=9)
  ax.set_xticklabels([]); ax.set_yticklabels([])
  ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
  ax.view_init(elev=32, azim=-52)
  ax.set_title("The story arc over one emotion axis\n"
               "height = mood (read off the labelled z-axis); "
               "marker color = dominant emotion", fontsize=12)
  handles = [plt.Line2D([], [], marker="o", ls="", color=EMOTION_COLOR[e], label=e)
             for e in order if e in EMOTION_COLOR]
  ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
  fig.tight_layout()
  p = os.path.join(out_dir, "emotion_axis_3d.png")
  fig.savefig(p, dpi=180); plt.close(fig); print(f"  wrote {p}")

  # --- figure 2: mood over reading order -- follow the story, watch it move -----
  # Primary line is the windows' OWN emotion mood: the surface mood is a heavy
  # regression and comes out nearly flat, so the felt emotion is what actually
  # moves. Both are smoothed for readability; the surface mood rides behind as a
  # faint reference.
  own_s, surf_s = smooth1d(mood_own, args.smooth), smooth1d(mood_arc, args.smooth)
  fig, ax = plt.subplots(figsize=(13, 5.6))
  for e, p in zip(order, positions):
    ax.axhline(p, color="0.85", lw=0.8, zorder=0)
  seg = np.stack([np.column_stack([centers[:-1], own_s[:-1]]),
                  np.column_stack([centers[1:], own_s[1:]])], axis=1)
  lc = LineCollection(seg, cmap="plasma", linewidth=2.6, zorder=2)
  lc.set_array(np.linspace(0, 1, len(centers) - 1))
  ax.add_collection(lc)
  ax.scatter(centers, own_s, c=dom_own_colors, s=22, zorder=3,
             edgecolors="white", linewidths=0.3)
  ax.plot(centers, surf_s, color="0.6", lw=1.0, ls="--", zorder=1,
          label="surface mood (from location)")
  ax.set_xlim(centers.min(), centers.max())
  # Ticks first: set_yticks/set_yticklabels re-autoscale y, so the limits must be
  # applied *after* them or the zoom is silently discarded.
  ax.set_yticks(positions)
  ax.set_yticklabels(order)
  if args.spread == "fill":
    # Robust to single-window spikes: zoom to the 1st-99th percentile band.
    both = np.concatenate([own_s, surf_s])
    lo, hi = np.nanpercentile(both, 1), np.nanpercentile(both, 99)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
  else:
    ax.set_ylim(-0.03, 1.03)
  ax.set_xlabel("reading position (paragraph)")
  ax.set_ylabel("mood  (dominant emotion = point color)")
  ax.set_title(f"{args.book}: the story's mood over reading order\n"
               "line height rises toward the positive pole, falls toward the "
               "negative; color = dominant emotion", fontsize=12)
  ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
  fig.tight_layout()
  p = os.path.join(out_dir, "emotion_axis_over_reading.png")
  fig.savefig(p, dpi=180); plt.close(fig); print(f"  wrote {p}")

  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
