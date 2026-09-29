"""
Parameter sweeps for the narrative arc: how the picture changes with the window,
the curve fit, the projections' own parameters, and the embedding model.

  windows      rows = (size, stride);   cols = PCA, UMAP, t-SNE, PCA 3-D
  fit          PCA 2-D and 3-D at one window; grid of n_control x lambda
  projections  t-SNE perplexity and UMAP n_neighbors ladders, with seed stability
  models       rows = every embedding model the book has; plus how far the models
               agree on the geometry of the series

Each sweep writes a grid figure and a CSV of metrics to
<output-dir>/<book>/<model>/sweeps/<sweep>/ (models: <book>/_models/sweeps/).

Every panel is colored by reading order (dark -> light), O = opening, X = ending.

Metrics, and what they say:

  closure       |first window - last window| / median pairwise distance, in the
                full embedding space (no projection involved). Near 0: the story
                ends where it began. Above 1: it ends further from its opening than
                a typical pair of windows are from each other.
  drift         Spearman correlation between reading gap |i - j| and distance
                |x_i - x_j|, in the full space. High: the book moves steadily away
                over time. Near 0: it circles one region.
  trust         sklearn trustworthiness (k = 5): whether each window's neighbours in
                the picture are neighbours in the full space. 1 is perfect.
  seed_disp     Procrustes disparity between the seed-0 and seed-1 layouts. 0: the
                layout does not depend on the seed.
  fit_resid     mean distance from each window to the fitted curve / RMS radius.
  out_of_order  fraction of consecutive windows whose curve positions go backwards.
  end_gap       mean distance between the curve's ends and O / X, / RMS radius.

Examples:

python sweeps/sweep.py                                     # all four, sample data
python sweeps/sweep.py --sweeps windows --windows 10:5 25:5 40:20 100:50
python sweeps/sweep.py --book hamlet --sweeps windows models
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness

# arc/ on the path, for the narrative_arc package one level up.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrative_arc import cli, paths
from narrative_arc import windows as W
from narrative_arc.colors import resolve_colors
from narrative_arc.curves import FitOpts, fit_curve
from narrative_arc.data import list_models
from narrative_arc.plot_2d import draw_arc
from narrative_arc.plot_3d import draw_arc_3d
from narrative_arc.projections import METHODS, project

SWEEPS = ("windows", "fit", "projections", "models")

# The arc scripts' default fit, with more iterations so a sweep panel is never a
# half-finished solve.
DEFAULT_FIT = FitOpts(n_control=12, degree=3, lambda_=0.1, solver="dn",
                      max_iter=300)

PANEL = 4.4  # inches per grid cell


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def series_metrics(pooled):
  """closure and drift, measured in the full embedding space."""
  D = squareform(pdist(pooled))
  i, j = np.triu_indices(len(pooled), 1)
  return {
    "closure": float(D[0, -1] / np.median(D[i, j])),
    "drift": float(spearmanr(j - i, D[i, j]).statistic),
  }


def trust(pooled, coords):
  k = min(5, len(pooled) // 2 - 1)
  return float(trustworthiness(pooled, coords, n_neighbors=k))


def fit_metrics(coords, fit):
  center = coords.mean(axis=0)
  rms = np.sqrt(((coords - center) ** 2).sum(axis=1).mean())
  to_curve = np.linalg.norm(coords[:, None] - fit.curve[None], axis=2).min(axis=1)
  ends = (np.linalg.norm(fit.curve[0] - coords[0])
          + np.linalg.norm(fit.curve[-1] - coords[-1])) / 2
  return {
    "fit_resid": float(to_curve.mean() / rms),
    "out_of_order": float((np.diff(fit.u) < 0).mean()),
    "end_gap": float(ends / rms),
    "converged": fit.converged,
  }


def adaptive_perplexity(n):
  """t-SNE perplexity for n windows: 30 when there are plenty, never above n/4.

  A fixed 30 over ~20 windows means every point treats every other as a
  neighbour, and the layout is noise; tying it to n keeps window settings
  comparable. The projections sweep is where perplexity itself is varied.
  """
  return float(min(30, max(5, n // 4)))


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

def panel(fig, rows, cols, index, proj, spec, title, fit=None,
          show_control=False):
  """One arc on a grid cell, 2-D or 3-D depending on the projection."""
  if proj.coords.shape[1] == 3:
    ax = fig.add_subplot(rows, cols, index, projection="3d")
    draw_arc_3d(fig, ax, proj, spec, title, fit, show_control=show_control,
                key=False)
    ax.set_zlabel(proj.labels[2], fontsize=7)
  else:
    ax = fig.add_subplot(rows, cols, index)
    draw_arc(fig, ax, proj, spec, title, fit, show_control=show_control,
             key=False)
  ax.title.set_fontsize(9)
  ax.xaxis.label.set_fontsize(7)
  ax.yaxis.label.set_fontsize(7)
  ax.tick_params(labelsize=6)
  return ax


def save_figure(fig, path, suptitle):
  fig.suptitle(suptitle + "\ncolor = reading order (dark -> light), "
               "O = opening, X = ending", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(path, dpi=130)
  plt.close(fig)
  print(f"  wrote {path}")


def write_csv(rows, path):
  keys = []
  for row in rows:
    keys.extend(k for k in row if k not in keys)
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v)
                       for k, v in row.items()})
  print(f"  wrote {path}")


def display_name(proj):
  return proj.name + (" 3-D" if proj.coords.shape[1] == 3 else "")


def four_projections(pooled, seed=0):
  """PCA, UMAP, t-SNE in 2-D, then PCA in 3-D -- the columns of most sweeps."""
  n = len(pooled)
  projs = project(pooled, METHODS, n_components=2, neighbors=15, min_dist=0.1,
                  perplexity=adaptive_perplexity(n), seed=seed)
  projs += project(pooled, ["pca"], n_components=3)
  return projs


def pca_variance(proj):
  ratios = proj.info.get("explained_variance_ratio")
  return float(sum(ratios)) if ratios else None


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def sweep_windows(args, out):
  settings = [tuple(int(x) for x in w.split(":")) for w in args.windows]
  cols = 4
  fig = plt.figure(figsize=(PANEL * cols, PANEL * len(settings)))
  rows = []

  for r, (size, stride) in enumerate(settings):
    print(f"\n  -- window {size}, stride {stride}")
    series = W.build(args.data_dir, args.book, args.model, size, stride)
    spec = resolve_colors(["progression"], series)[0]
    base = {"size": size, "stride": stride, "n_windows": len(series),
            **series_metrics(series.pooled)}

    for c, proj in enumerate(four_projections(series.pooled)):
      fit = fit_curve(proj.coords, DEFAULT_FIT)
      fm = fit_metrics(proj.coords, fit)
      t = trust(series.pooled, proj.coords)
      title = (f"w{size} s{stride} ({len(series)} windows) - {display_name(proj)}\n"
               f"trust {t:.2f} - fit resid {fm['fit_resid']:.2f}")
      panel(fig, len(settings), cols, r * cols + c + 1, proj, spec, title, fit)
      rows.append({**base, "projection": display_name(proj),
                   "pca_var": pca_variance(proj), "trust": t,
                   "tsne_perplexity": proj.info.get("perplexity"), **fm})

  save_figure(fig, os.path.join(out, "windows.png"),
              f"Window sweep -- {args.book} / {args.model} -- fit c12 lambda 0.1, "
              f"t-SNE perplexity = min(30, n/4)")
  write_csv(rows, os.path.join(out, "windows.csv"))
  return rows


def sweep_fit(args, out):
  series = W.build(args.data_dir, args.book, args.model, args.size, args.stride)
  spec = resolve_colors(["progression"], series)[0]
  n_controls = [6, 10, 16, 24]
  lambdas = [0.01, 0.1, 1.0, 10.0]
  rows = []

  for dim in (2, 3):
    proj = project(series.pooled, ["pca"], n_components=dim)[0]
    fig = plt.figure(figsize=(PANEL * len(n_controls), PANEL * len(lambdas)))
    for r, lam in enumerate(lambdas):
      for c, nc in enumerate(n_controls):
        opts = DEFAULT_FIT._replace(n_control=nc, lambda_=lam)
        fit = fit_curve(proj.coords, opts)
        fm = fit_metrics(proj.coords, fit)
        title = (f"c{nc}  lambda {lam:g}\n"
                 f"resid {fm['fit_resid']:.2f} - order {fm['out_of_order']:.0%} - "
                 f"ends {fm['end_gap']:.2f}"
                 + ("" if fm["converged"] else " - NOT CONVERGED"))
        panel(fig, len(lambdas), len(n_controls), r * len(n_controls) + c + 1,
              proj, spec, title, fit, show_control=True)
        rows.append({"dim": dim, "n_control": nc, "lambda": lam, "degree": 3,
                     "solver": "dn", **fm})
    save_figure(fig, os.path.join(out, f"fit_pca{dim}d.png"),
                f"Fit sweep -- PCA {dim}-D -- {args.book} / {args.model} -- "
                f"window {args.size}, stride {args.stride}")

    # Degree and solver only go in the table: at a fixed n_control and lambda
    # their curves are too alike to be worth a figure.
    for degree in (2, 3, 4, 5):
      for solver in ("dn", "lm"):
        opts = DEFAULT_FIT._replace(degree=degree, solver=solver)
        rows.append({"dim": dim, "n_control": opts.n_control,
                     "lambda": opts.lambda_, "degree": degree, "solver": solver,
                     **fit_metrics(proj.coords, fit_curve(proj.coords, opts))})

  write_csv(rows, os.path.join(out, "fit.csv"))
  return rows


def sweep_projections(args, out):
  size, stride = args.proj_size, args.proj_stride
  series = W.build(args.data_dir, args.book, args.model, size, stride)
  spec = resolve_colors(["progression"], series)[0]
  n = len(series)
  ladder = [5, 10, 20, 40]
  rows = []
  fig = plt.figure(figsize=(PANEL * len(ladder), PANEL * 2))

  for r, method in enumerate(("tsne", "umap")):
    for c, value in enumerate(ladder):
      kw = ({"perplexity": float(value)} if method == "tsne"
            else {"neighbors": value})
      runs = [project(series.pooled, [method], n_components=2, seed=seed,
                      **kw)[0] for seed in (0, 1)]
      proj = runs[0]
      _, _, disparity = procrustes(runs[0].coords, runs[1].coords)
      t = trust(series.pooled, proj.coords)
      fit = fit_curve(proj.coords, DEFAULT_FIT)
      fm = fit_metrics(proj.coords, fit)
      label = (f"perplexity {proj.info['perplexity']:g}" if method == "tsne"
               else f"n_neighbors {proj.info['n_neighbors']}")
      title = f"{proj.name} {label}\ntrust {t:.2f} - seed disparity {disparity:.2f}"
      panel(fig, 2, len(ladder), r * len(ladder) + c + 1, proj, spec, title, fit)
      rows.append({"projection": proj.name, "param": label, "n_windows": n,
                   "trust": t, "seed_disp": float(disparity), **fm})

  pca = project(series.pooled, ["pca"], n_components=2)[0]
  rows.append({"projection": "PCA", "param": "-", "n_windows": n,
               "trust": trust(series.pooled, pca.coords), "seed_disp": 0.0})

  save_figure(fig, os.path.join(out, "projections.png"),
              f"Projection sweep -- {args.book} / {args.model} -- "
              f"window {size}, stride {stride} ({n} windows)")
  write_csv(rows, os.path.join(out, "projections.csv"))
  return rows


def sweep_models(args, out):
  models = list_models(args.data_dir, args.book)
  if len(models) < 2:
    print(f"  skipping models sweep: {args.book} has only {models}")
    return []

  cols = 4
  fig = plt.figure(figsize=(PANEL * cols, PANEL * len(models)))
  rows, distances = [], {}

  for r, model in enumerate(models):
    print(f"\n  -- {model}")
    series = W.build(args.data_dir, args.book, model, args.size, args.stride)
    spec = resolve_colors(["progression"], series)[0]
    distances[model] = pdist(series.pooled)
    base = {"model": model, "n_windows": len(series),
            **series_metrics(series.pooled)}
    for c, proj in enumerate(four_projections(series.pooled)):
      fit = fit_curve(proj.coords, DEFAULT_FIT)
      t = trust(series.pooled, proj.coords)
      title = (f"{model} - {display_name(proj)}\n"
               f"closure {base['closure']:.2f} - drift {base['drift']:.2f} - "
               f"trust {t:.2f}")
      panel(fig, len(models), cols, r * cols + c + 1, proj, spec, title, fit)
      rows.append({**base, "projection": display_name(proj),
                   "pca_var": pca_variance(proj), "trust": t,
                   **fit_metrics(proj.coords, fit)})

  # Do the models agree on the shape of the series? Spearman correlation of the
  # window-to-window distances, model against model.
  agreement = []
  for a in range(len(models)):
    for b in range(a + 1, len(models)):
      rho = spearmanr(distances[models[a]], distances[models[b]]).statistic
      agreement.append({"model_a": models[a], "model_b": models[b],
                        "distance_spearman": float(rho)})
      print(f"  {models[a]} vs {models[b]}: distance correlation {rho:.2f}")

  save_figure(fig, os.path.join(out, "models.png"),
              f"Model sweep -- {args.book} -- window {args.size}, "
              f"stride {args.stride}")
  write_csv(rows, os.path.join(out, "models.csv"))
  write_csv(agreement, os.path.join(out, "models_agreement.csv"))
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  cli.add_target_args(parser)
  parser.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=SWEEPS)
  parser.add_argument("--windows", nargs="+",
                      default=["10:5", "20:10", "40:20", "80:40", "40:5"],
                      help="size:stride pairs for the windows sweep.")
  parser.add_argument("--size", default=W.DEFAULT_SIZE, type=int,
                      help="Window for the fit and models sweeps.")
  parser.add_argument("--stride", default=W.DEFAULT_STRIDE, type=int)
  parser.add_argument("--proj-size", default=20, type=int,
                      help="Window for the projections sweep; smaller than the "
                           "default so perplexity/neighbors have room to vary.")
  parser.add_argument("--proj-stride", default=10, type=int)
  args = parser.parse_args()

  # The models sweep already compares every model of a book, so it runs once
  # per book rather than once per (book, model).
  models_done = set()
  for book, model in cli.targets(args, parser):
    args.book, args.model = book, model
    for sweep in args.sweeps:
      if sweep == "models":
        if book in models_done:
          continue
        models_done.add(book)
      print(f"\n=== {sweep} sweep: {book} / {model} ===")
      owner = "_models" if sweep == "models" else model
      out = paths.out_dir(args.output_dir, book, owner, "sweeps", sweep)
      globals()[f"sweep_{sweep}"](args, out)
      paths.stamp(out, __file__, args, sweep=sweep)

  print("\nDone.")


if __name__ == "__main__":
  main()
