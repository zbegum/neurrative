"""
One interpretable emotion axis: the six emotions collapsed into a single height,
so following the story arc *is* watching the emotion change.

arc_on_surface.py lifts the arc onto six separate emotion surfaces. Here the six
are combined into one "mood", the repository's single definition in
surface/mood/mood.py: each paragraph's emotions are rank-normalised over the book
and blended to one value on the spectrum

    sadness -> danger -> confusion -> curiosity -> wonder -> humor
     (heavy / negative)                              (light / positive)

and those per-paragraph moods are smoothed into one surface over the PCA plane
(surface/mood/surface.py). The z-axis is ticked with the emotion names, so the
emotion is read straight off the height.

The arc is the windows' mean points in the same plane, lightly smoothed, sampled
densely and lifted onto that surface, so it lies on it. The reading-order figure
shows both the mood the landscape assigns to the route (dashed) and the windows'
own mood (the mean of their paragraphs' moods).

Everything lands in arc_on_surface/output/<book>/<model>/narrative_arc_3d/mood_axis/<variant>/,
the variant naming the window, bandwidth and blend.

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
from windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds
# One definition of mood for the whole repository: surface/mood.
from surface.mood import mood as mood_mod, surface as mood_surface
from surface.mood.mood import SPECTRUM

# Valence order, heavy/negative (bottom) to light/positive (top). Adjustable via
# --order; anything not named is appended so the run never silently drops one.
DEFAULT_ORDER = SPECTRUM



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


def lifted_arc(arc_xy, height, arc_smooth, per=20, lift=0.01):
  """The windows' path, smoothed over window index, sampled densely and lifted.

  `height` evaluates the surface at (n, 2) plane points. Lifting every dense
  sample (not just the windows) keeps the drawn path on the surface; straight 3-D
  chords between lifted windows would cut through the hills between them.
  """
  wx = smooth1d(arc_xy[:, 0], arc_smooth)
  wy = smooth1d(arc_xy[:, 1], arc_smooth)
  t = np.linspace(0, len(wx) - 1, per * len(wx))
  xy = np.column_stack([np.interp(t, np.arange(len(wx)), wx),
                        np.interp(t, np.arange(len(wy)), wy)])
  return np.column_stack([xy, height(xy) + lift])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--size", default=DEFAULT_SIZE, type=int)
  ap.add_argument("--stride", default=DEFAULT_STRIDE, type=int)
  ap.add_argument("--blend", default="banded", choices=mood_mod.BLENDS,
                  help="How six emotions become one mood (surface/mood/mood.py).")
  ap.add_argument("--h", default=0.2, type=float,
                  help="Bandwidth of the mood surface, in standardized coordinates.")
  ap.add_argument("--resolution", default=120, type=int)
  ap.add_argument("--order", nargs="+", default=None,
                  help="Emotion order, bottom (negative) to top (positive).")
  ap.add_argument("--spread", default="fill", choices=["absolute", "fill"],
                  help="absolute: y spans the full sadness..humor line. fill: zoom "
                       "y to the mood's actual range so the changes fill the plot.")
  ap.add_argument("--smooth", default=1.5, type=float,
                  help="Gaussian smoothing (in windows) of the mood line; 0 = raw.")
  ap.add_argument("--arc-smooth", default=1.0, type=float,
                  help="Smoothing (in windows) of the 3-D arc path so it reads as a "
                       "clean curve over the landscape instead of a hairball.")
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)

  # The repository's mood: per paragraph, then one surface over the PCA plane.
  m, order, positions, _ = mood_mod.mood(matrix, emotions, order=args.order or DEFAULT_ORDER,
                                         blend=args.blend)
  print(f"=== emotion axis: {args.book} / {args.model} ===")
  print("  spectrum (bottom->top): " +
        " -> ".join(f"{e}({p:.2f})" for e, p in zip(order, positions)))

  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          {"w": args.size, "s": args.stride, "h": args.h,
                           "blend": args.blend})
  paths.stamp(out_dir, __file__, args, stack="surface.mood",
              estimator="mood per paragraph, one gaussian_nw surface (surface/mood)",
              spectrum=order)

  windows = window_bounds(n, args.size, args.stride)
  centers = np.array([(s + e - 1) / 2.0 for s, e in windows])
  arc_xy = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])

  GX, GY, mood_grid, height_at = mood_surface.fit(X_raw, m, args.h,
                                                  resolution=args.resolution)
  mood_arc = height_at(arc_xy)                                   # the surface under each window
  mood_own = np.array([m[s:e].mean() for s, e in windows])       # the window's own mood
  print(f"  blend={args.blend} h={args.h} spread={args.spread} smooth={args.smooth}")

  print(f"  arc mood range {mood_arc.min():.2f}..{mood_arc.max():.2f} "
        f"(0=neg pole, 1=pos pole)")

  # --- figure 1: the mood surface with the arc riding it (3-D) ------------------
  # Smooth the arc PATH (x, y and height together) so it reads as one clean curve
  # flowing over the landscape rather than a window-to-window hairball, and lift
  # it a hair above the surface so it is never buried inside it.
  pts = lifted_arc(arc_xy, height_at, args.arc_smooth)

  fig = plt.figure(figsize=(12, 9))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot_surface(GX, GY, mood_grid, cmap="magma", vmin=0, vmax=1,
                  linewidth=0, antialiased=True, alpha=0.4, rstride=2, cstride=2)
  seg = np.stack([pts[:-1], pts[1:]], axis=1)
  lc = Line3DCollection(seg, cmap="plasma", linewidth=4.0, zorder=5)
  lc.set_array(np.linspace(0, 1, len(pts) - 1))
  ax.add_collection3d(lc)
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
               "color = reading order", fontsize=12)
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
  ax.set_ylabel("mood")
  ax.set_title(f"{args.book}: the story's mood over reading order\n"
               "line height rises toward the positive pole, falls toward the "
               "negative; color = reading order", fontsize=12)
  ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
  fig.tight_layout()
  p = os.path.join(out_dir, "emotion_axis_over_reading.png")
  fig.savefig(p, dpi=180); plt.close(fig); print(f"  wrote {p}")

  print(f"  -> {out_dir}")


if __name__ == "__main__":
  main()
