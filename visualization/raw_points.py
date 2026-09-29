"""
The unsmoothed point cloud every surface in `surface/` is fitted to.

Each estimator subfolder shows an "after": a continuous landscape over the PCA
plane. This is the "before" -- the 789 paragraphs themselves, at their own
positions, carrying their own annotated scores, with nothing fitted. Lands in
`surface/raw/` so the comparison is one directory apart.

Three panels per emotion, because the raw data has three things worth seeing:

  3-D    the same view as surface_<emotion>.png, at the same z-limits and in the
         same colormap, so flipping between them shows exactly what smoothing
         did. This is the scatter those surfaces are drawn through.
  2-D    the plane from above, colored by score: where in semantic space the
         emotion lives, without a surface interpolating over the gaps.
  hist   the score distribution. The annotations are heavily quantized -- a
         handful of distinct values, mostly multiples of 0.1 -- which is the
         reason the pipeline regresses rather than interpolates, and the reason
         no surface can pass through every sample.

Example:

python visualization/raw_points.py --book alice_wonderland --model bge-m3
python visualization/raw_points.py --model bge-m3 --emotions wonder danger
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca

# Matched to smooth_common so a raw panel and a fitted surface are directly
# comparable: same ramp, and the same 0..1 z-limits.
CMAP = "viridis"
SUBDIR = os.path.join(paths.SURFACE, paths.SURFACE_RAW)


def raw_plot(X, y, emotion, n_distinct, output_path):
  fig = plt.figure(figsize=(16, 5.5))

  # Left: the scatter the surfaces are fitted through, at surface z-limits.
  ax = fig.add_subplot(131, projection="3d")
  ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap=CMAP, vmin=0.0, vmax=1.0,
             s=9, alpha=0.55, depthshade=False, linewidths=0)
  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_zlabel(emotion)
  ax.set_zlim(0.0, 1.0)
  ax.set_title("raw scores as height\n(what the surface is fitted to)", fontsize=10)

  # Middle: the plane from above. No surface, so the empty regions stay empty --
  # the gaps a smoother has to invent a height for are visible here.
  ax2 = fig.add_subplot(132)
  sc = ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP, vmin=0.0, vmax=1.0,
                   s=16, alpha=0.85, linewidths=0.2, edgecolors="#33322e")
  ax2.set_xlabel("PC1")
  ax2.set_ylabel("PC2")
  ax2.set_title("where the emotion sits in the plane", fontsize=10)
  fig.colorbar(sc, ax=ax2, label=emotion)

  # Right: why this is a regression problem and not an interpolation one.
  ax3 = fig.add_subplot(133)
  values, counts = np.unique(y, return_counts=True)
  ax3.bar(values, counts, width=0.035, color="#2a78d6", alpha=0.85)
  ax3.set_xlabel(emotion)
  ax3.set_ylabel("paragraphs")
  ax3.set_xlim(-0.05, 1.0)
  ax3.set_title(f"{n_distinct} distinct values over {len(y)} paragraphs\n"
                f"mean {y.mean():.3f}   sd {y.std():.3f}", fontsize=10)

  fig.suptitle(f"{emotion} - raw annotations, unsmoothed")
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--book", default="alice_wonderland")
  ap.add_argument("--model", required=True)
  ap.add_argument("--emotions", nargs="*", default=None,
                  help="Subset of emotions (default: all).")
  args = ap.parse_args()

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X = load_pca(args.book, args.model)
  if len(X) != len(paragraphs):
    raise ValueError(f"pca ({len(X)}) and paragraphs ({len(paragraphs)}) mismatch.")

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

  out_dir = paths.out_dir(args.book, args.model, SUBDIR)
  print(f"=== raw points | {args.book} / {args.model}")
  print(f"{len(X)} paragraphs, {len(selected)} emotion(s)\n")

  distinct = {}
  for emotion in selected:
    y = matrix[:, emotions.index(emotion)]
    n_distinct = int(len(np.unique(y)))
    distinct[emotion] = n_distinct
    print(f"{emotion}: range {y.min():.2f}..{y.max():.2f}  mean {y.mean():.3f}  "
          f"sd {y.std():.3f}  {n_distinct} distinct values")
    raw_plot(X, y, emotion, n_distinct,
             os.path.join(out_dir, paths.named("raw", emotion)))

  paths.stamp(out_dir, __file__, args, estimator="none (raw annotations)",
              emotions=selected, n_paragraphs=int(len(X)),
              distinct_values=distinct)

  print(f"\nDone.\n{out_dir}")


if __name__ == "__main__":
  main()
