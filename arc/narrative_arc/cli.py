"""Arguments and setup shared by arc_2d.py and arc_3d.py."""

import os

import numpy as np

from . import paths
from . import windows as W
from .colors import COLOR_HELP
from .curves import SOLVERS, FitOpts, fit_curve
from .data import list_books, list_models
from .projections import METHODS


def add_target_args(parser):
  """Which books and models to run, and where data comes from / goes to."""
  parser.add_argument("--data-dir", default=paths.DEFAULT_DATA_DIR,
                      help="Directory holding <book>/processed.json and "
                           "<book>/embeddings/<model>/embeddings.npy.")
  parser.add_argument("--output-dir", default=paths.DEFAULT_OUTPUT_DIR)
  parser.add_argument("--book", default="alice_wonderland")
  parser.add_argument("--all-books", action="store_true")
  parser.add_argument("--model", default="bge-m3")
  parser.add_argument("--all-models", action="store_true")


def add_window_args(parser):
  parser.add_argument("--size", default=W.DEFAULT_SIZE, type=int,
                      help="Window length in paragraphs. Larger smooths more but "
                           "blurs fast turns in the story.")
  parser.add_argument("--stride", default=W.DEFAULT_STRIDE, type=int,
                      help="Paragraphs between consecutive windows. 1 is maximal "
                           "overlap; = size gives non-overlapping blocks.")
  parser.add_argument("--l2", action="store_true",
                      help="L2-normalize embeddings before pooling (pool by "
                           "direction, matching cosine comparison).")


def add_arc_args(parser):
  add_target_args(parser)
  add_window_args(parser)

  parser.add_argument("--methods", nargs="+", default=list(METHODS),
                      choices=METHODS, help="Which projections to draw.")
  parser.add_argument("--color", nargs="+", default=["progression", "chapter"],
                      help=COLOR_HELP)
  parser.add_argument("--min-score", default=0.2, type=float,
                      help="For --color dominant: below this pooled top score a "
                           "window is 'unclear' rather than colored.")
  parser.add_argument("--no-grid", action="store_true",
                      help="Skip the side-by-side figure of all projections.")

  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--neighbors", default=15, type=int, help="UMAP n_neighbors.")
  parser.add_argument("--min-dist", default=0.1, type=float, help="UMAP min_dist.")
  parser.add_argument("--perplexity", default=30.0, type=float,
                      help="t-SNE perplexity; clamped below the window count.")

  parser.add_argument("--fit", action="store_true",
                      help="Overlay a uniform B-spline fitted through the windows "
                           "with bspline-regression.")
  parser.add_argument("--n-control", default=12, type=int,
                      help="B-spline control points. Fewer => smoother/looser; "
                           "more => hugs the windows.")
  parser.add_argument("--degree", default=3, type=int,
                      help="B-spline degree (3 = cubic).")
  parser.add_argument("--lambda", dest="lambda_", default=0.1, type=float,
                      help="Regularisation pulling adjacent control points "
                           "together. Larger => shorter, straighter curve.")
  parser.add_argument("--solver", default="dn", choices=SOLVERS,
                      help="dn: damped Newton. lm: Levenberg-Marquardt.")
  parser.add_argument("--max-iter", default=100, type=int,
                      help="Solver iteration cap.")
  parser.add_argument("--show-control", action="store_true",
                      help="Draw the fitted control polygon too.")


def targets(args, parser):
  """[(book, model)] for the requested --book/--all-books x --model/--all-models."""
  if not os.path.isdir(args.data_dir):
    parser.error(f"--data-dir {args.data_dir} does not exist")
  books = list_books(args.data_dir) if args.all_books else [args.book]
  pairs = []
  for book in books:
    models = list_models(args.data_dir, book) if args.all_models else [args.model]
    pairs.extend((book, model) for model in models)
  return pairs


def fit_opts(args):
  if not args.fit:
    return None
  return FitOpts(args.n_control, args.degree, args.lambda_, args.solver,
                 args.max_iter)


def fit_projection(proj, opts, path):
  """Fit one projection's arc, report it, and save it to `path` (.npz).

  bspline-regression optimises every window's position u on the curve along with
  the control points, so a window can end up nearer a later stretch of curve
  than an earlier one. How many do is printed: a few is jitter; many means the
  curve is not following reading order and n_control or lambda needs changing.
  """
  if opts is None:
    return None
  fit = fit_curve(proj.coords, opts)
  out_of_order = int((np.diff(fit.u) < 0).sum())
  print(f"  {proj.name} fit: energy {fit.energy:.3g}, "
        f"{'converged' if fit.converged else 'NOT converged (raise --max-iter)'}, "
        f"{out_of_order} of {len(fit.u) - 1} steps out of reading order")
  np.savez(path, curve=fit.curve, control_points=fit.control_points, u=fit.u,
           converged=fit.converged, energy=fit.energy)
  return fit


def grid_suffix(projections):
  """Which projections, and all their parameters, in one filename fragment."""
  return "_" + "-".join(p.tag for p in projections) + "".join(
    p.suffix for p in projections)
