"""
The protocol every smoother is held to.

So that the four methods are actually comparable, they share all of it:

  standardization    x and y are z-scored on the training split only, so the
                     bandwidths h_x, h_y mean the same thing across methods and
                     no validation information leaks into the scaling.
  splits             a held-out validation set is set aside first; parameters
                     are tuned by k-fold CV *inside* the training split, and the
                     reported score is measured once, on data never used to fit
                     or to tune.
  metrics            RMSE and MAE, reported in score units (0-1), not z-units.
  prediction grid    the same grid and the same density mask for every method.
  plots              the fitted surface and the residuals.

Per-method scripts supply only a name, an estimator and a parameter grid.
"""

import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paths
from data import load_paragraphs, load_scores
from geometry.smoothers import f64

CMAP = "viridis"
RESIDUAL_CMAP = "RdBu_r"      # diverging: residuals have a sign and a zero


def load_pca(book, model):
  # Resolved against the repo root, not the working directory: writes go through
  # paths.out_dir, and a read that resolved differently would have these scripts
  # silently reading one book's coordinates while writing another's.
  structure_dir = paths.out_dir(book, model, paths.STRUCTURE_2D, create=False)
  # structure_2d/pca/pca.npy since the projections were foldered by method, but
  # a directory that has not been migrated still has it flat -- find_coords takes
  # whichever exists so the whole geometry stack is not hostage to the layout.
  path = paths.find_coords(structure_dir, "pca.npy")
  if path is None:
    raise FileNotFoundError(
      f"pca.npy not found under {structure_dir}. Run "
      f"visualization/embedding.py --book {book} --model {model} first."
    )
  return np.load(path)


class Standardizer:
  """z-score, fitted on the training split only."""

  def fit(self, a):
    a = f64(a)
    self.mean = a.mean(axis=0)
    self.std = a.std(axis=0)
    # A constant column would divide by zero; leave it alone instead.
    self.std = np.where(self.std > 1e-12, self.std, 1.0)
    return self

  def forward(self, a):
    return (f64(a) - self.mean) / self.std

  def inverse(self, a):
    return f64(a) * self.std + self.mean


def rmse(a, b):
  return float(np.sqrt(np.mean((f64(a) - f64(b)) ** 2)))


def mae(a, b):
  return float(np.mean(np.abs(f64(a) - f64(b))))


def param_combos(grid):
  """Cartesian product of a {name: [values]} grid."""
  names = list(grid)
  return [dict(zip(names, vals)) for vals in itertools.product(*(grid[n] for n in names))]


def kfold_indices(n, folds, rng):
  idx = rng.permutation(n)
  return np.array_split(idx, folds)


def tune(predict, grid, Xtr, ytr, folds, seed):
  """k-fold CV inside the training split. Returns (best params, all results)."""
  rng = np.random.default_rng(seed)
  parts = kfold_indices(len(Xtr), folds, rng)
  results = []

  for params in param_combos(grid):
    errs = []
    for f in range(folds):
      val = parts[f]
      tr = np.concatenate([parts[g] for g in range(folds) if g != f])
      pred, _ = predict(Xtr[tr], ytr[tr], Xtr[val], **params)
      # A method with compact support can decline to predict; score it against
      # the training mean there rather than letting NaN hide the failure.
      pred = np.where(np.isfinite(pred), pred, ytr[tr].mean())
      errs.append(rmse(ytr[val], pred))
    results.append({"params": params, "cv_rmse": float(np.mean(errs))})

  results.sort(key=lambda r: r["cv_rmse"])
  return results[0]["params"], results


def fit_grid(predict, params, X, y, resolution, margin, density_floor_pct):
  """Evaluate the smoother over the prediction grid, masked by support."""
  lo, hi = X.min(axis=0), X.max(axis=0)
  pad = (hi - lo) * margin
  lo, hi = lo - pad, hi + pad
  gx, gy = np.meshgrid(
    np.linspace(lo[0], hi[0], resolution),
    np.linspace(lo[1], hi[1], resolution),
  )
  Q = np.column_stack([gx.ravel(), gy.ravel()])

  values, support = predict(X, y, Q, **params)
  _, sample_support = predict(X, y, X, **params)
  # A negative percentile switches the mask off. It is not the same as 0: the
  # floor is a percentile of support *at the paragraphs*, so 0 still means "at
  # least as well supported as the emptiest paragraph", which masks the corners
  # of the grid where there is nothing at all. Only -inf covers the whole plane.
  floor = (-np.inf if density_floor_pct < 0
           else np.percentile(sample_support, density_floor_pct))

  keep = (support >= floor) & np.isfinite(values)
  Z = np.where(keep, values, np.nan).reshape(gx.shape)
  return gx, gy, Z, keep.reshape(gx.shape), float(floor)


def surface_plot(gx, gy, Z, X, y, emotion, title, output_path):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")
  surf = ax.plot_surface(gx, gy, Z, cmap=CMAP, vmin=0.0, vmax=1.0,
                         linewidth=0, antialiased=True, alpha=0.85,
                         rstride=2, cstride=2)
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


def residual_plot(X, resid, emotion, title, output_path):
  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                gridspec_kw={"width_ratios": [1.15, 1]})

  # Symmetric limits so the diverging ramp puts zero at its neutral midpoint.
  lim = float(np.abs(resid).max()) or 1.0
  sc = ax.scatter(X[:, 0], X[:, 1], c=resid, cmap=RESIDUAL_CMAP,
                  vmin=-lim, vmax=lim, s=14, linewidths=0.2,
                  edgecolors="#33322e")
  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_title("Residuals in the latent plane\n(structure here = the fit is missing something)",
               fontsize=10)
  fig.colorbar(sc, ax=ax, label=f"{emotion}: actual - fitted")

  ax2.hist(resid, bins=40, color="#2a78d6", alpha=0.85)
  ax2.axvline(0.0, color="#33322e", linewidth=1)
  ax2.set_xlabel(f"{emotion}: actual - fitted")
  ax2.set_ylabel("paragraphs")
  ax2.set_title(f"mean {resid.mean():+.3f}   sd {resid.std():.3f}", fontsize=10)

  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def add_common_args(parser):
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--model", required=True, type=str)
  parser.add_argument("--emotions", nargs="*", default=None,
                      help="Subset of emotions (default: all).")
  parser.add_argument("--val-frac", default=0.2, type=float,
                      help="Held-out fraction, never used to fit or tune.")
  parser.add_argument("--folds", default=5, type=int,
                      help="k-fold CV inside the training split.")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--resolution", default=200, type=int)
  parser.add_argument("--margin", default=0.05, type=float)
  parser.add_argument("--density-floor", default=5.0, type=float,
                      help="Mask the surface below this percentile of the "
                           "support seen at the paragraphs themselves. Pass a "
                           "negative value to mask nothing, giving one "
                           "unbroken surface over the whole PCA rectangle; 0 "
                           "is not the same thing, since it still requires as "
                           "much support as the emptiest paragraph has.")
  return parser


def prepare(args):
  """Load, standardize and split. Returns a dict shared by every method."""
  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X_raw = load_pca(args.book, args.model)
  if len(X_raw) != len(paragraphs):
    raise ValueError(f"pca ({len(X_raw)}) and paragraphs ({len(paragraphs)}) mismatch.")

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

  rng = np.random.default_rng(args.seed)
  order = rng.permutation(len(X_raw))
  n_val = int(round(args.val_frac * len(X_raw)))
  val_idx, tr_idx = order[:n_val], order[n_val:]

  return {
    "emotions": emotions, "selected": selected, "matrix": matrix,
    "X_raw": X_raw, "tr_idx": tr_idx, "val_idx": val_idx,
  }


def run(method, predict, grid, args):
  """Run one smoother over the requested emotions under the shared protocol."""
  ctx = prepare(args)
  # The method has its own directory and its parameters are tuned per emotion,
  # not chosen by the caller, so two runs of the same method on the same data
  # are the same run -- with one exception. `--density-floor` is the caller's,
  # and it decides how much of the plane the surface covers: at the default it
  # is masked back to where the paragraphs are, at 0 it is a full rectangle over
  # the whole PCA plane. Those are different surfaces, so they get different
  # directories rather than one silently replacing the other.
  out_dir = paths.out_dir(
    args.book, args.model, os.path.join(paths.SURFACE, method),
    {"df": None if args.density_floor == 5.0 else args.density_floor})

  X_raw, tr_idx, val_idx = ctx["X_raw"], ctx["tr_idx"], ctx["val_idx"]
  print(f"=== {method} | {args.book} / {args.model}")
  print(f"train {len(tr_idx)} | held-out {len(val_idx)} | "
        f"{args.folds}-fold CV over {len(param_combos(grid))} settings")

  summary, tuned = [], {}
  for emotion in ctx["selected"]:
    y_raw = ctx["matrix"][:, ctx["emotions"].index(emotion)]
    print(f"\n{emotion}:")

    # Scalers see the training split only: fitting them on everything would
    # leak the held-out distribution into the model.
    xs = Standardizer().fit(X_raw[tr_idx])
    ys = Standardizer().fit(y_raw[tr_idx])
    Xtr, ytr = xs.forward(X_raw[tr_idx]), ys.forward(y_raw[tr_idx])
    Xval, yval = xs.forward(X_raw[val_idx]), y_raw[val_idx]

    params, results = tune(predict, grid, Xtr, ytr, args.folds, args.seed)
    tuned[emotion] = params
    print(f"  tuned {params}  (cv rmse {results[0]['cv_rmse']:.4f} in z-units)")

    pred_val, _ = predict(Xtr, ytr, Xval, **params)
    pred_val = ys.inverse(np.where(np.isfinite(pred_val), pred_val, 0.0))
    val_rmse, val_mae = rmse(yval, pred_val), mae(yval, pred_val)

    # The flat mean is the thing any real surface has to beat.
    base = np.full_like(yval, y_raw[tr_idx].mean())
    print(f"  held-out RMSE {val_rmse:.4f}  MAE {val_mae:.4f}   "
          f"(flat mean: RMSE {rmse(yval, base):.4f}  MAE {mae(yval, base):.4f})")

    # The delivered surface is fitted on everything, tuned as above.
    xs_all, ys_all = Standardizer().fit(X_raw), Standardizer().fit(y_raw)
    X_all, y_all = xs_all.forward(X_raw), ys_all.forward(y_raw)
    gx, gy, Zz, mask, floor = fit_grid(
      predict, params, X_all, y_all, args.resolution, args.margin, args.density_floor
    )
    Z = ys_all.inverse(Zz)
    gx_raw = xs_all.inverse(np.column_stack([gx.ravel(), gy.ravel()]))
    GX = gx_raw[:, 0].reshape(gx.shape)
    GY = gx_raw[:, 1].reshape(gy.shape)

    fitted, _ = predict(X_all, y_all, X_all, **params)
    resid = y_raw - ys_all.inverse(np.where(np.isfinite(fitted), fitted, 0.0))

    print(f"  surface: {int(mask.sum())}/{mask.size} cells above the support "
          f"floor; relief {np.nanmin(Z):.2f} .. {np.nanmax(Z):.2f}")

    np.savez(os.path.join(out_dir, paths.named("field", emotion, "npz")),
             gx=GX, gy=GY, Z=Z, mask=mask, X=X_raw, y=y_raw,
             emotion=emotion, method=method, params=json.dumps(params))

    surface_plot(GX, GY, Z, X_raw, y_raw, emotion,
                 f"{emotion} landscape - {method} {params}",
                 os.path.join(out_dir, paths.named("surface", emotion)))
    residual_plot(X_raw, resid, emotion,
                  f"{emotion} residuals - {method} {params}",
                  os.path.join(out_dir, paths.named("residuals", emotion)))

    summary.append({
      "method": method, "emotion": emotion, "params": params,
      "cv_rmse_z": results[0]["cv_rmse"],
      "val_rmse": val_rmse, "val_mae": val_mae,
      "baseline_rmse": rmse(yval, base), "baseline_mae": mae(yval, base),
    })

  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

  # emotion_surface.py writes field_<emotion>.npz too. Same estimator, but tuned
  # in raw coordinates by leave-one-out rather than here in standardized ones by
  # k-fold. Recording the tuned parameters is what tells the two apart; the
  # tuning is only known once the loop has run.
  paths.stamp(out_dir, __file__, args, stack="geometry.smoothers",
              estimator=method, emotions=ctx["selected"], tuned=tuned,
              search_space=grid)

  print(f"\nDone.\n{out_dir}")
  return summary
