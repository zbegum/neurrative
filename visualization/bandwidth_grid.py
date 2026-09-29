"""
Bandwidth grid: rows are smoothers, columns are bandwidth. Surfaces only.

The ladder runs *up to* each method's cross-validated bandwidth rather than
around it. Sweeping above the tuned value only ever produced a featureless dome
-- the same picture for every method and every emotion -- so the interesting
range is below it, where the terrain goes from noise to structure.

Each method's tuned bandwidth is found first (same protocol as smooth_common:
train/validation split, k-fold CV inside the training split), then the columns
scale it by --factors. The rightmost column is therefore the CV choice.

The panels carry nothing but the surface and its bandwidth. Metrics go to
metrics.csv beside the images rather than onto them.

Example:

python visualization/bandwidth_grid.py --model bge-m3
python visualization/bandwidth_grid.py --model bge-m3 --emotions wonder \
  --factors 0.125 0.35 1.0
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import paths
from smooth_common import (Standardizer, add_common_args, fit_grid, mae,
                           prepare, rmse, tune)
from geometry.smoothers import SMOOTHERS

import smooth_epanechnikov_nw
import smooth_gaussian_nw
import smooth_local_linear
import smooth_loess

CMAP = "viridis"

# method -> (tuning grid, which parameters the columns scale)
METHODS = [
  ("gaussian_nw", smooth_gaussian_nw.GRID, ("hx", "hy")),
  ("epanechnikov_nw", smooth_epanechnikov_nw.GRID, ("hx", "hy")),
  ("local_linear", smooth_local_linear.GRID, ("hx", "hy")),
  ("loess", dict(smooth_loess.GRID, iterations=[2]), ("frac",)),
]


def scaled(params, keys, factor):
  out = dict(params)
  for k in keys:
    out[k] = params[k] * factor
  # LOESS's neighbourhood is a fraction of the samples, so it cannot exceed 1.
  if "frac" in out:
    out["frac"] = float(np.clip(out["frac"], 1e-3, 1.0))
  return out


def label(params, keys):
  return "  ".join(f"{k}={params[k]:.3g}" for k in keys)


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.add_argument("--factors", nargs=3, type=float, default=[0.25, 0.5, 1.0],
                      help="Multiples of each method's tuned bandwidth. The "
                           "default ladder ends at the CV choice.")
  args = parser.parse_args()

  ctx = prepare(args)
  # The ladder is the whole point of the figure, so it names the directory:
  # smooth_grid.py sweeps a different one into the same parent, and two ladders
  # sharing a folder would overwrite each other's metrics.
  out_dir = paths.out_dir(args.book, args.model,
                          os.path.join(paths.SURFACE, paths.SURFACE_SWEEP),
                          variant_params={"x": args.factors})
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="swept: " + ", ".join(name for name, *_ in METHODS),
              emotions=ctx["selected"])

  X_raw, tr_idx, val_idx = ctx["X_raw"], ctx["tr_idx"], ctx["val_idx"]
  print(f"=== bandwidth grid | {args.book} / {args.model}")
  print(f"train {len(tr_idx)} | held-out {len(val_idx)} | factors {args.factors}")

  rows = []
  for emotion in ctx["selected"]:
    y_raw = ctx["matrix"][:, ctx["emotions"].index(emotion)]
    print(f"\n{emotion}:")

    xs = Standardizer().fit(X_raw[tr_idx])
    ys = Standardizer().fit(y_raw[tr_idx])
    Xtr, ytr = xs.forward(X_raw[tr_idx]), ys.forward(y_raw[tr_idx])
    Xval, yval = xs.forward(X_raw[val_idx]), y_raw[val_idx]

    xs_all, ys_all = Standardizer().fit(X_raw), Standardizer().fit(y_raw)
    X_all, y_all = xs_all.forward(X_raw), ys_all.forward(y_raw)

    fig, axes = plt.subplots(
      len(METHODS), len(args.factors),
      figsize=(4.2 * len(args.factors), 3.9 * len(METHODS)),
      subplot_kw={"projection": "3d"},
    )

    for r, (method, grid, keys) in enumerate(METHODS):
      predict, _ = SMOOTHERS[method]
      best, _ = tune(predict, grid, Xtr, ytr, args.folds, args.seed)
      print(f"  {method:16s} tuned {label(best, keys)}")

      for c, factor in enumerate(args.factors):
        params = scaled(best, keys, factor)

        pred, _ = predict(Xtr, ytr, Xval, **params)
        pred = ys.inverse(np.where(np.isfinite(pred), pred, 0.0))

        gx, gy, Zz, mask, _ = fit_grid(
          predict, params, X_all, y_all, args.resolution, args.margin,
          args.density_floor,
        )
        Z = ys_all.inverse(Zz)
        raw = xs_all.inverse(np.column_stack([gx.ravel(), gy.ravel()]))
        GX, GY = raw[:, 0].reshape(gx.shape), raw[:, 1].reshape(gy.shape)

        ax = axes[r][c]
        ax.plot_surface(GX, GY, Z, cmap=CMAP, vmin=0.0, vmax=1.0,
                        linewidth=0, antialiased=True, rstride=2, cstride=2)
        # One z-scale for every panel: without it, a flat surface would be
        # stretched to look as tall as a real hill.
        ax.set_zlim(0.0, 1.0)
        # Surfaces only -- the shape is the message, and the numbers are in the csv.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_title(label(params, keys), fontsize=9, pad=0)

        rows.append({"emotion": emotion, "method": method, "factor": factor,
                     "params": params, "val_rmse": rmse(yval, pred),
                     "val_mae": mae(yval, pred)})

      # Name the row once, on the left, instead of on all three panels.
      pos = axes[r][0].get_position()
      fig.text(0.012, (pos.y0 + pos.y1) / 2, method, rotation=90,
               va="center", ha="left", fontsize=11)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.01,
                        wspace=0.0, hspace=0.12)
    path = os.path.join(out_dir, paths.named("bandwidth", emotion))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")

  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(rows, f, indent=2)

  with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
    f.write("emotion,method,factor,params,val_rmse,val_mae\n")
    for r in rows:
      f.write(f'{r["emotion"]},{r["method"]},{r["factor"]:.4g},'
              f'"{r["params"]}",{r["val_rmse"]:.5f},{r["val_mae"]:.5f}\n')

  print(f"\nDone.\n{out_dir}")


if __name__ == "__main__":
  main()
