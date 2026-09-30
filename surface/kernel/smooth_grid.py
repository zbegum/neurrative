"""
Compare the smoothers head to head: rows are methods, columns are smoothness.

Every cell is fitted and scored under the identical protocol from
smooth_common (same standardization, same split, same grid, same metrics), so
the only thing varying down a column is the method and along a row is the
smoothness parameter. Each panel is titled with its held-out RMSE, so the
comparison is numeric rather than a matter of taste.

The middle column is each method's cross-validated choice; the outer columns are
that choice divided and multiplied by --factor, i.e. deliberately too sharp and
too smooth. Reading across a row shows the bias-variance tradeoff; reading down
a column shows how much the method itself matters at a comparable smoothness.

Example:

python surface/kernel/smooth_grid.py --model bge-m3 --emotions wonder
python surface/kernel/smooth_grid.py --model bge-m3
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from smooth_common import (Standardizer, add_common_args, fit_grid, mae,
                           param_combos, prepare, rmse, tune)
from geometry.smoothers import SMOOTHERS


# The same methods, tuning grids and column scaling as the bandwidth grid.
from bandwidth_grid import CMAP, METHODS, scaled


def label(params, keys):
  return ", ".join(f"{k}={params[k]:.3g}" for k in keys)


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.add_argument("--factor", default=3.0, type=float,
                      help="How far the outer columns sit from the tuned middle.")
  args = parser.parse_args()

  ctx = prepare(args)
  # Columns are the tuned bandwidth divided and multiplied by --factor, so the
  # ladder is 1/f, 1, f. Named the same way bandwidth_grid.py names its own.
  out_dir = paths.out_dir(args.book, args.model,
                          os.path.join(paths.SURFACE, paths.SURFACE_SWEEP),
                          variant_params={"x": [round(1.0 / args.factor, 2),
                                                1, args.factor]})
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator="swept: " + ", ".join(name for name, *_ in METHODS),
              emotions=ctx["selected"])

  X_raw, tr_idx, val_idx = ctx["X_raw"], ctx["tr_idx"], ctx["val_idx"]
  print(f"=== smoother grid | {args.book} / {args.model}")
  print(f"train {len(tr_idx)} | held-out {len(val_idx)}")

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
      len(METHODS), 3, figsize=(15, 5 * len(METHODS)),
      subplot_kw={"projection": "3d"},
    )

    for r, (method, grid, keys) in enumerate(METHODS):
      predict, _ = SMOOTHERS[method]
      best, _ = tune(predict, grid, Xtr, ytr, args.folds, args.seed)
      print(f"  {method:16s} tuned {label(best, keys)}")

      for c, factor in enumerate([1 / args.factor, 1.0, args.factor]):
        params = scaled(best, keys, factor)

        pred, _ = predict(Xtr, ytr, Xval, **params)
        pred = ys.inverse(np.where(np.isfinite(pred), pred, 0.0))
        val_rmse, val_mae = rmse(yval, pred), mae(yval, pred)

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
        ax.set_zlim(0.0, 1.0)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.set_zlabel(emotion, fontsize=7)
        tuned_mark = "  (CV)" if factor == 1.0 else ""
        ax.set_title(f"{method}{tuned_mark}\n{label(params, keys)}\n"
                     f"RMSE {val_rmse:.4f}   MAE {val_mae:.4f}", fontsize=9)

        rows.append({"emotion": emotion, "method": method,
                     "factor": factor, "params": params,
                     "val_rmse": val_rmse, "val_mae": val_mae})

    base = np.full_like(yval, y_raw[tr_idx].mean())
    fig.suptitle(
      f"{emotion}: smoother x smoothness  |  held-out RMSE to beat "
      f"(flat mean) = {rmse(yval, base):.4f}",
      fontsize=15,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, paths.named("grid", emotion))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")

  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(rows, f, indent=2)

  # A flat table beats digging through JSON when you just want the ranking.
  csv = os.path.join(out_dir, "metrics.csv")
  with open(csv, "w") as f:
    f.write("emotion,method,factor,params,val_rmse,val_mae\n")
    for r in rows:
      f.write(f'{r["emotion"]},{r["method"]},{r["factor"]:.4g},'
              f'"{r["params"]}",{r["val_rmse"]:.5f},{r["val_mae"]:.5f}\n')

  print(f"\nDone.\n{out_dir}")


if __name__ == "__main__":
  main()
