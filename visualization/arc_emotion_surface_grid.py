"""
A grid of the 3-D emotion-axis surface (arc riding the mood landscape), sweeping
two parameters -- the bandwidth-grid idea applied to emotion_axis_3d.png.

Each panel is the same object as arc_emotion_axis.py's 3-D figure: the six
emotions collapsed onto one mood surface over the PCA plane, with the story arc
riding it and the z-axis labelled by emotion. Sweeping two knobs side by side
shows how the landscape and the route respond:

  rows    --bandwidths   kernel bandwidth h of the emotion surfaces. Small =
          detailed, bumpy terrain; large = a flat swell. (0.1 spiky, 0.4 a dome.)
  cols    --temps        softmax temperature of the mood blend: small drives the
          arc height toward the poles, large keeps it near the middle.

Per-emotion surfaces depend only on the bandwidth, so each row is fitted once and
reused across the temperature columns.

Lands in output/<book>/<model>/arc_on_surface/<variant>/emotion_axis_surface_grid.png,
the variant naming the swept spans and the knobs held fixed.

Example:

  python visualization/arc_emotion_surface_grid.py --book alice_wonderland --model bge-m3
  python visualization/arc_emotion_surface_grid.py --bandwidths 0.1 0.15 0.25 --temps 0.2 0.4
"""

import argparse
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca
from arc_on_surface import window_bounds, build_surface, surface_at
from arc_emotion_axis import (normalizer, mood, smooth1d, nan_blur,
                              sample_surface, EMOTION_COLOR, DEFAULT_ORDER)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--bandwidths", nargs="+", type=float, default=[0.1, 0.15, 0.25],
                  help="Surface bandwidths -> grid rows.")
  ap.add_argument("--temps", nargs="+", type=float, default=[0.2, 0.35, 0.6],
                  help="Softmax temperatures -> grid columns.")
  ap.add_argument("--size", default=10, type=int)
  ap.add_argument("--stride", default=5, type=int)
  ap.add_argument("--norm", default="rank", choices=["rank", "zscore", "minmax"])
  ap.add_argument("--arc-smooth", default=3.0, type=float)
  ap.add_argument("--surface-smooth", default=1.5, type=float)
  ap.add_argument("--resolution", default=90, type=int,
                  help="Grid resolution per panel (lower than the single figure "
                       "since the panels are small).")
  ap.add_argument("--margin", default=0.05, type=float)
  ap.add_argument("--density-floor", default=5.0, type=float)
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  order = args.order or [e for e in DEFAULT_ORDER if e in emotions]
  order += [e for e in emotions if e not in order]
  positions = np.linspace(0.0, 1.0, len(order))

  windows = window_bounds(n, args.size, args.stride)
  arc_xy = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])
  tf = {e: normalizer(matrix[:, emotions.index(e)], args.norm) for e in order}

  # Swept knobs contribute their span (h0.1-0.25, t0.2-0.6); the window and the
  # normalization are fixed per run and pin down the rest of the figure.
  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          {"w": args.size, "s": args.stride,
                           "h": (min(args.bandwidths), max(args.bandwidths)),
                           "t": (min(args.temps), max(args.temps)),
                           "norm": args.norm})
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="gaussian_nw (isotropic h per row; standardized coords)",
              spectrum=order)
  nr, nc = len(args.bandwidths), len(args.temps)
  print(f"=== emotion-axis surface grid: {args.book} / {args.model} ===")
  print(f"  {nr} bandwidths x {nc} temps (each row fitted once), res {args.resolution}")

  fig = plt.figure(figsize=(5.0 * nc, 4.4 * nr))
  for r, h in enumerate(args.bandwidths):
    # Fit the six surfaces once per bandwidth row; temperature columns reuse them.
    GX = GY = None
    grid_Z, arc_h = [], []
    for e in order:
      y = matrix[:, emotions.index(e)]
      GX, GY, Z, xs, ys = build_surface(X_raw, y, h, h, args.resolution,
                                        args.margin, args.density_floor)
      grid_Z.append(Z)
      arc_h.append(surface_at(xs, ys, X_raw, y, arc_xy, h, h))
    grid_n = np.stack([np.where(np.isnan(Z), np.nan, tf[e](Z))
                       for e, Z in zip(order, grid_Z)])
    arc_n = np.stack([tf[e](hh) for e, hh in zip(order, arc_h)])
    all_supported = np.all(np.isfinite(grid_n), axis=0)
    dom = np.array(order)[np.nanargmax(np.where(np.isfinite(arc_n), arc_n, -np.inf), axis=0)]
    dom_colors = [EMOTION_COLOR.get(e, "#555") for e in dom]
    print(f"  bandwidth {h}: surfaces fitted")

    for c, temp in enumerate(args.temps):
      ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
      mood_grid = np.where(all_supported, mood(grid_n, positions, temp), np.nan)
      mood_grid = nan_blur(mood_grid, args.surface_smooth)
      mood_arc = mood(arc_n, positions, temp)

      ax.plot_surface(GX, GY, mood_grid, cmap="magma", vmin=0, vmax=1,
                      linewidth=0, antialiased=True, alpha=0.4, rstride=2, cstride=2)
      ax_s = smooth1d(arc_xy[:, 0], args.arc_smooth)
      ay_s = smooth1d(arc_xy[:, 1], args.arc_smooth)
      az_on = sample_surface(GX, GY, mood_grid, ax_s, ay_s)
      az_s = np.where(np.isfinite(az_on), az_on, smooth1d(mood_arc, args.arc_smooth))
      pts = np.column_stack([ax_s, ay_s, az_s + 0.01])
      seg = np.stack([pts[:-1], pts[1:]], axis=1)
      lc = Line3DCollection(seg, cmap="plasma", linewidth=2.6, zorder=5)
      lc.set_array(np.linspace(0, 1, len(pts) - 1))
      ax.add_collection3d(lc)
      sl = slice(None, None, 5)
      ax.scatter(pts[sl, 0], pts[sl, 1], pts[sl, 2],
                 c=[dom_colors[i] for i in range(0, len(pts), 5)],
                 s=12, depthshade=False, zorder=6)
      ax.scatter(*pts[0], color="black", s=45, marker="o", depthshade=False, zorder=7)
      ax.scatter(*pts[-1], color="black", s=55, marker="X", depthshade=False, zorder=7)

      ax.set_zlim(0, 1)
      ax.set_zticks(positions)
      ax.set_zticklabels(order if c == 0 else [], fontsize=7)
      ax.set_xticklabels([]); ax.set_yticklabels([])
      ax.view_init(elev=30, azim=-52)
      if r == 0:
        ax.set_title(f"temp = {temp:g}", fontsize=11, pad=0)
      if c == nc - 1:
        ax.text2D(1.02, 0.5, f"h = {h:g}", transform=ax.transAxes,
                  rotation=270, va="center", fontsize=11)

  fig.suptitle(f"Emotion-axis mood surface -- {args.book} / {args.model}\n"
               "rows: surface bandwidth (bumpy vs flat) | columns: temperature "
               "(arc swing) | height = mood, color = dominant emotion", fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  p = os.path.join(out_dir, "emotion_axis_surface_grid.png")
  fig.savefig(p, dpi=155); plt.close(fig)
  print(f"  wrote {p}")


if __name__ == "__main__":
  main()
