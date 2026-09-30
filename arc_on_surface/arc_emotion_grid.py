"""
A parameter grid for the emotion-axis mood line, in the spirit of the bandwidth
grid: rows and columns sweep two knobs so their effect is legible at a glance.

The panel is the reading-order mood (arc_emotion_axis.py's second figure): the
story's mood, the repository's one definition (surface/mood/mood.py), averaged
over each window and followed left to right. It needs no surface, so a whole grid
is cheap.

  rows    --sizes    window length: how much reading is pooled per point. Small =
          detailed and jumpy; large = a few broad movements.
  cols    --blends   how six emotions become one mood (banded, softmax, project,
          pc1; see surface/mood/mood.py). For project and pc1 the height is a
          direction, so read the emotion ticks only as low and high.

Line color is reading progression. Line smoothing (--smooth) is held fixed so
only the two swept knobs vary.

Lands in arc_on_surface/output/<book>/<model>/arc_on_surface/<variant>/emotion_axis_grid.png,
the variant naming the swept spans and the two knobs held fixed.

Example:

  python arc_on_surface/arc_emotion_grid.py --book alice_wonderland --model bge-m3
  python arc_on_surface/arc_emotion_grid.py --sizes 20 40 80 --blends banded softmax
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
from arc_emotion_axis import smooth1d, DEFAULT_ORDER
from surface.mood import mood as mood_mod
from windows import window_bounds


def mood_line(m, size, smooth):
  """The windows' mean mood over reading order, for one window size."""
  windows = window_bounds(len(m), size, max(1, size // 2))
  centers = np.array([(s + e - 1) / 2.0 for s, e in windows])
  return centers, smooth1d(np.array([m[s:e].mean() for s, e in windows]), smooth)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--sizes", nargs="+", type=int, default=[20, 40, 80],
                  help="Window lengths -> grid rows.")
  ap.add_argument("--blends", nargs="+", default=list(mood_mod.BLENDS),
                  choices=mood_mod.BLENDS, help="Mood blends -> grid columns.")
  ap.add_argument("--smooth", default=1.5, type=float)
  ap.add_argument("--order", nargs="+", default=None)
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  moods = {}
  for b in args.blends:
    m, order, positions, _ = mood_mod.mood(matrix, emotions,
                                           order=args.order or DEFAULT_ORDER, blend=b)
    moods[b] = m

  # The swept window sizes are a span, so the variant carries it.
  out_dir = paths.out_dir(args.book, args.model, os.path.join(paths.NARRATIVE_ARC_3D, paths.ARC_MOOD_AXIS),
                          {"w": (min(args.sizes), max(args.sizes)), "sm": args.smooth})
  paths.stamp(out_dir, __file__, args, estimator="none (windows' own mood)",
              spectrum=order)
  nr, nc = len(args.sizes), len(args.blends)
  print(f"=== emotion-axis grid: {args.book} / {args.model} ===")
  print(f"  {nr} sizes x {nc} blends, smooth={args.smooth}")

  fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.0 * nr),
                           sharex=True, squeeze=False)
  for r, size in enumerate(args.sizes):
    for c, blend in enumerate(args.blends):
      ax = axes[r][c]
      centers, m = mood_line(moods[blend], size, args.smooth)
      for p in positions:
        ax.axhline(p, color="0.9", lw=0.7, zorder=0)
      seg = np.stack([np.column_stack([centers[:-1], m[:-1]]),
                      np.column_stack([centers[1:], m[1:]])], axis=1)
      lc = LineCollection(seg, cmap="plasma", linewidth=2.0, zorder=2)
      lc.set_array(np.linspace(0, 1, len(centers) - 1))
      ax.add_collection(lc)
      ax.set_xlim(centers.min(), centers.max())
      # Robust zoom to the mood's actual range so movement fills each panel.
      lo, hi = np.nanpercentile(m, 1), np.nanpercentile(m, 99)
      pad = 0.12 * (hi - lo) + 1e-6
      ax.set_yticks(positions)
      ax.set_yticklabels(order if c == 0 else [], fontsize=8)
      ax.set_ylim(lo - pad, hi + pad)
      if r == 0:
        ax.set_title(f"blend {blend}", fontsize=11)
      if c == nc - 1:
        ax.text(1.02, 0.5, f"window = {size}", transform=ax.transAxes,
                rotation=270, va="center", ha="left", fontsize=11)
      if r == nr - 1:
        ax.set_xlabel("reading position", fontsize=9)

  fig.suptitle(f"Emotion-axis mood over reading order -- {args.book} / {args.model}\n"
               "rows: window size (detail vs breadth) | columns: mood blend "
               "| color = reading order", fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  p = os.path.join(out_dir, "emotion_axis_grid.png")
  fig.savefig(p, dpi=160); plt.close(fig)
  print(f"  wrote {p}")


if __name__ == "__main__":
  main()
