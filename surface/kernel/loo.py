"""
Fit a surface to the score-z point cloud: the emotion landscape.

The score-z plots are a height field -- z is the emotion score over the 2-D PCA
plane -- so a surface z = f(x, y) genuinely exists, and kernel smoothing is how
we recover it. (The proj3d plots are not a height field: z there is a third
component, and two paragraphs can share an (x, y) at different heights, so no
such surface exists for them.)

The bandwidth h decides everything about the terrain: too small and it is spikes
of annotation noise, too large and it is a featureless dome. By default we pick
it by leave-one-out cross-validation; --sweep draws a ladder of bandwidths so
you can see the tradeoff rather than trust one number.

The fitted field is saved as .npz for the geodesic work, which walks on exactly
this surface.

Example:

python surface/kernel/loo.py --book alice_wonderland --model bge-m3
python surface/kernel/loo.py --book alice_wonderland --model bge-m3 \
  --emotions wonder --sweep
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np


# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
from geometry.scalar_field import bandwidth_candidates, field, loo_bandwidth
# The PCA plane is the one thing this script shares with the smoothers stack, so
# it reads it through the same loader rather than a second copy that could drift.
from smooth_common import load_pca

CMAP = "viridis"


def surface_plot(X, y, gx, gy, Z, emotion, h, title, output_path):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")

  surf = ax.plot_surface(
    gx, gy, Z, cmap=CMAP, vmin=0.0, vmax=1.0,
    linewidth=0, antialiased=True, alpha=0.85, rstride=2, cstride=2,
  )

  # The samples the surface was fitted to, so its honesty is visible.
  ax.scatter(X[:, 0], X[:, 1], y, c="#33322e", s=3, alpha=0.35, depthshade=False)

  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_zlabel(emotion)
  ax.set_zlim(0.0, 1.0)
  ax.set_title(title)
  fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=emotion)

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def sweep_plot(X, y, emotion, hs, best_h, args, output_path):
  cols = len(hs)
  fig = plt.figure(figsize=(5 * cols, 5))

  for i, h in enumerate(hs):
    gx, gy, Z, _, _ = field(X, y, h, args.resolution, density_floor_pct=args.density_floor)
    ax = fig.add_subplot(1, cols, i + 1, projection="3d")
    ax.plot_surface(gx, gy, Z, cmap=CMAP, vmin=0.0, vmax=1.0,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.set_zlim(0.0, 1.0)
    ax.set_title(f"h = {h:.3f}" + ("  (LOO-CV)" if np.isclose(h, best_h) else ""),
                 fontsize=11)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_zlabel(emotion, fontsize=8)

  fig.suptitle(f"{emotion} landscape vs kernel bandwidth", fontsize=14)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--model", required=True, type=str)
  parser.add_argument("--emotions", nargs="*", default=None,
                      help="Subset of emotions (default: all).")
  parser.add_argument("--bandwidth", type=float, default=None,
                      help="Kernel bandwidth. Default: leave-one-out CV.")
  parser.add_argument("--sweep", action="store_true",
                      help="Also draw the landscape across a ladder of bandwidths.")
  parser.add_argument("--resolution", default=200, type=int)
  parser.add_argument("--density-floor", default=5.0, type=float,
                      help="Mask the surface below this percentile of the kernel "
                           "density seen at the paragraphs themselves.")
  args = parser.parse_args()

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

  output_dir = paths.out_dir(args.book, args.model,
                             os.path.join(paths.SURFACE, paths.SURFACE_ISOTROPIC))

  candidates = bandwidth_candidates(X)
  print(f"PCA span {np.linalg.norm(X.max(0) - X.min(0)):.3f} | "
        f"bandwidths {candidates[0]:.3f} .. {candidates[-1]:.3f}")

  chosen_h = {}
  for emotion in selected:
    y = matrix[:, emotions.index(emotion)]
    print(f"\n{emotion}:")

    if args.bandwidth:
      h, mses = args.bandwidth, None
      print(f"  bandwidth {h:.3f} (given)")
    else:
      h, mses = loo_bandwidth(X, y, candidates)
      # A field that cannot beat the flat mean is not a landscape, it is noise.
      baseline = np.var(y)
      print(f"  bandwidth {h:.3f} (LOO-CV)  mse {mses.min():.4f} vs "
            f"{baseline:.4f} for the flat mean "
            f"-> explains {1 - mses.min() / baseline:.1%} of the variance")

    chosen_h[emotion] = float(h)
    gx, gy, Z, mask, floor = field(X, y, h, args.resolution,
                                   density_floor_pct=args.density_floor)
    print(f"  surface: {mask.sum()} of {mask.size} grid cells above the density "
          f"floor ({floor:.3f}); relief {np.nanmin(Z):.2f} .. {np.nanmax(Z):.2f}")

    np.savez(
      os.path.join(output_dir, paths.named("field", emotion, "npz")),
      gx=gx, gy=gy, Z=Z, mask=mask, h=h, X=X, y=y, emotion=emotion,
    )

    surface_plot(
      X, y, gx, gy, Z, emotion, h,
      f"{emotion} landscape over the PCA map (kernel smoothed, h={h:.3f})",
      os.path.join(output_dir, paths.named("surface", emotion)),
    )

    if args.sweep:
      ladder = [h / 3, h, h * 3]
      sweep_plot(X, y, emotion, ladder, h, args,
                 os.path.join(output_dir, paths.named("sweep", emotion)))

  # surface/kernel/gaussian.py also writes field_<emotion>.npz, from a different
  # estimator over differently scaled coordinates. Naming the stack and the
  # per-emotion bandwidth is what lets a reader tell the two apart later. Written
  # last because the bandwidth is only known once the CV has run.
  paths.stamp(output_dir, __file__, args, stack="geometry.scalar_field",
              estimator="nadaraya_watson (isotropic h, LOO-CV)",
              emotions=selected, bandwidth=chosen_h)

  print(f"\nDone.\n{output_dir}")


if __name__ == "__main__":
  main()
