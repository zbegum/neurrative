"""
A grid of the 3-D emotion-axis surface (arc riding the mood landscape), sweeping
two parameters -- the bandwidth-grid idea applied to emotion_axis_3d.png.

Each panel is the same object as arc_emotion_axis.py's 3-D figure: the
repository's mood surface (surface/mood) over the PCA plane, with the story arc
lifted onto it and the z-axis labelled by emotion. Sweeping two knobs side by
side shows how the landscape and the route respond:

  rows    --bandwidths   bandwidth h of the mood surface. Small = detailed,
          bumpy terrain; large = a flat swell.
  cols    --blends       how six emotions become one mood (surface/mood/mood.py).

Lands in arc_on_surface/output/<book>/<model>/arc_on_surface/<variant>/emotion_axis_surface_grid.png,
the variant naming the swept spans and the knobs held fixed.

Example:

  python arc_on_surface/arc_emotion_surface_grid.py --book alice_wonderland --model bge-m3
  python arc_on_surface/arc_emotion_surface_grid.py --bandwidths 0.1 0.2 0.4 --blends banded softmax
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
from arc_emotion_axis import lifted_arc, DEFAULT_ORDER
from surface.mood import mood as mood_mod, surface as mood_surface
from windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--bandwidths", nargs="+", type=float, default=[0.1, 0.2, 0.4],
                  help="Mood-surface bandwidths -> grid rows.")
  ap.add_argument("--blends", nargs="+", default=list(mood_mod.BLENDS),
                  choices=mood_mod.BLENDS, help="Mood blends -> grid columns.")
  ap.add_argument("--size", default=DEFAULT_SIZE, type=int)
  ap.add_argument("--stride", default=DEFAULT_STRIDE, type=int)
  ap.add_argument("--resolution", default=90, type=int,
                  help="Grid resolution per panel (lower than the single figure "
                       "since the panels are small).")
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = np.asarray(load_pca(args.book, args.model), dtype=np.float64)[:, :2]
  n = len(X_raw)
  moods = {}
  for b in args.blends:
    m, order, positions, _ = mood_mod.mood(matrix, emotions,
                                           order=args.order or DEFAULT_ORDER, blend=b)
    moods[b] = m

  windows = window_bounds(n, args.size, args.stride)
  arc_xy = np.array([X_raw[s:e].mean(axis=0) for s, e in windows])
  centers = np.array([(s + e - 1) / 2.0 for s, e in windows])

  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          {"w": args.size, "s": args.stride,
                           "h": (min(args.bandwidths), max(args.bandwidths))})
  paths.stamp(out_dir, __file__, args, stack="surface.mood",
              estimator="mood per paragraph, one gaussian_nw surface per panel",
              spectrum=order)
  nr, nc = len(args.bandwidths), len(args.blends)
  print(f"=== emotion-axis surface grid: {args.book} / {args.model} ===")
  print(f"  {nr} bandwidths x {nc} blends, res {args.resolution}")

  fig = plt.figure(figsize=(5.0 * nc, 4.4 * nr))
  for r, h in enumerate(args.bandwidths):
    for c, blend in enumerate(args.blends):
      ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
      GX, GY, Z, height_at = mood_surface.fit(X_raw, moods[blend], h,
                                              resolution=args.resolution)
      ax.plot_surface(GX, GY, Z, cmap="magma", vmin=0, vmax=1,
                      linewidth=0, antialiased=True, alpha=0.4, rstride=2, cstride=2)
      pts = np.vstack(lifted_arc(GX, GY, Z, arc_xy, centers))
      seg = np.stack([pts[:-1], pts[1:]], axis=1)
      lc = Line3DCollection(seg, cmap="plasma", linewidth=2.6, zorder=5)
      lc.set_array(np.linspace(0, 1, len(pts) - 1))
      ax.add_collection3d(lc)
      ax.scatter(*pts[0], color="black", s=45, marker="o", depthshade=False, zorder=7)
      ax.scatter(*pts[-1], color="black", s=55, marker="X", depthshade=False, zorder=7)

      ax.set_zlim(0, 1)
      ax.set_zticks(positions)
      ax.set_zticklabels(order if c == 0 else [], fontsize=7)
      ax.set_xticklabels([]); ax.set_yticklabels([])
      ax.view_init(elev=30, azim=-52)
      if r == 0:
        ax.set_title(f"blend {blend}", fontsize=11, pad=0)
      if c == nc - 1:
        ax.text2D(1.02, 0.5, f"h = {h:g}", transform=ax.transAxes,
                  rotation=270, va="center", fontsize=11)
    print(f"  bandwidth {h}: done")

  fig.suptitle(f"Emotion-axis mood surface -- {args.book} / {args.model}\n"
               "rows: surface bandwidth (bumpy vs flat) | columns: mood blend "
               "| height = mood, color = reading order", fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  p = os.path.join(out_dir, "emotion_axis_surface_grid.png")
  fig.savefig(p, dpi=155); plt.close(fig)
  print(f"  wrote {p}")


if __name__ == "__main__":
  main()
