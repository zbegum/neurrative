"""Fit a B-spline surface to (PC1, PC2, emotion score), one per emotion.

The same height field `surface/kernel/loo.py` recovers with a kernel
smoother, fitted instead with a least-squares B-spline. z = f(x, y)
is a genuine function over the PCA plane -- PCA is linear and metric, so each
paragraph has one well-defined (x, y) -- which is what makes the tensor-product
form applicable at all. It is not applicable over UMAP or t-SNE, and there is no
flag here to try.

The knob is the knot count, and it runs the opposite way from a bandwidth: more
knots is a more flexible surface, so it is chosen by k-fold CV, like
`smooth_common.py` does for hx/hy, and reported against the flat-mean baseline
so an emotion whose landscape is not real says so.

`--domain` decides how much plane the answer covers, and the two settings are
different claims, not different renderings:

  support   the default. Knots span the paragraphs, the ridge keeps the system
            solvable, and grid cells with no paragraph near them are left blank.
            The surface is drawn only where the book is.

  unit      PC1 and PC2 are rescaled into [0, 1] and the surface covers the
            whole square, paragraphs or no paragraphs. Filling the empty part
            needs a penalty that ties neighbouring coefficients together rather
            than the reference's ridge, which would drop those cells to zero -- see
            `penalty_matrix`. The support mask is still computed, saved, and
            drawn as the solid part of the figure with the extension faded
            behind it, because the extension is an extension and the figure
            should not pretend otherwise.

Example:

  python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3
  python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3 \
    --domain unit
  python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3 \
    --emotions wonder --sweep
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


import bspline_surface as bs
# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_paragraphs, load_scores
from smooth_common import load_pca

CMAP = "viridis"

# The ladder the knot count is chosen from. It stops at 20 because 20 knots per
# axis is 23x23 = 529 coefficients against ~800 paragraphs -- past that the fit
# has more freedom than the book has text, whatever the CV says.
#
# It starts at 2 -- a single span, i.e. one bicubic polynomial over the whole
# plane -- because under the ridge the CV keeps choosing the coarse end, and a
# ladder whose floor is the answer cannot say whether it is the answer or the
# floor.
KNOT_LADDER = (2, 3, 4, 6, 8, 10, 12, 16, 20)

# Only searched on the penalized path. Under the ridge `lam` is not a smoothing
# parameter -- it is there to keep the matrix invertible -- so it stays at
# whatever `--lam` says and only the knot count is tuned.
LAM_LADDER = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def breaks_for(coords, n_knots, args):
  """The breakpoints per axis: the data's own range, or the whole unit square.

  Under `--domain unit` these do not depend on the data at all, which is what
  lets a CV fold and the full fit share one domain -- and what makes the corners
  of the square part of the fit rather than off the end of it.
  """
  if args.domain == "unit":
    breaks = np.linspace(0.0, 1.0, n_knots)
    return breaks, breaks
  return (bs.uniform_breaks(coords[:, 0], n_knots, args.pad),
          bs.uniform_breaks(coords[:, 1], n_knots, args.pad))


def cv_rmse(X, z, n_knots, lam, args):
  """k-fold CV error for one (knot count, lam), or inf if it cannot be fitted.

  A fold can leave a coefficient with no data behind it, which under the ridge
  is a singular system. That is not an error to abort on -- it is this knot
  count being too fine for this data, i.e. exactly what the CV is measuring --
  so it scores as inf and the ladder moves on.
  """
  rng = np.random.default_rng(args.seed)
  order = rng.permutation(len(X))
  errors = []

  for k in range(args.folds):
    test = order[k::args.folds]
    train = np.setdiff1d(order, test)
    bx, by = breaks_for(X[train], n_knots, args)
    try:
      surface = bs.fit(X[train, 0], X[train, 1], z[train], args.degree, bx, by,
                       lam, penalty_order=args.penalty_order)
    except (ValueError, np.linalg.LinAlgError):
      return np.inf
    # Points outside the training fold's knot rectangle evaluate to 0 through
    # the basis, which would be scored as a confident prediction of zero. They
    # are held out of the error instead: what is being compared is the knot
    # counts, and every one of them has the same handful of edge paragraphs.
    # (Under `--domain unit` the rectangle is the square and nothing is outside.)
    inside = ((X[test, 0] >= bx[0]) & (X[test, 0] <= bx[-1]) &
              (X[test, 1] >= by[0]) & (X[test, 1] <= by[-1]))
    if not inside.any():
      return np.inf
    pred = surface(X[test, 0][inside], X[test, 1][inside])
    errors.append(np.mean((pred - z[test][inside]) ** 2))

  return float(np.sqrt(np.mean(errors)))


def extension_range(X, z, n_knots, lam, args):
  """How far outside [0, 1] the surface goes over the whole domain.

  Cross-validation cannot see this. It scores predictions at held-out
  *paragraphs*, all of which are inside the cloud, so it is measuring fidelity
  where there is data and is silent about the corners of the square -- which is
  precisely where a weakly-penalized fit goes to a score of -6 or 5. A candidate
  that is excellent by CV and absurd in the corners has to be rejected by
  something, and this is that something.
  """
  bx, by = breaks_for(X, n_knots, args)
  try:
    surface = bs.fit(X[:, 0], X[:, 1], z, args.degree, bx, by, lam,
                     penalty_order=args.penalty_order)
  except (ValueError, np.linalg.LinAlgError):
    return np.inf, -np.inf
  gx, gy = np.meshgrid(np.linspace(bx[0], bx[-1], 60),
                       np.linspace(by[0], by[-1], 60))
  Z = surface.grid(gx, gy)
  return float(Z.min()), float(Z.max())


def choose_parameters(X, z, args):
  """The best (knots, lam) by CV among the candidates that stay in bounds.

  Returns (n_knots, lam, report) where report is the whole search, so the run
  log and `params.json` can show what was rejected rather than only what won.
  """
  lams = LAM_LADDER if args.penalty_order else (args.lam,)
  # `--control` fixes the coefficient count, which fixes the knot count, so the
  # ladder collapses to one rung and only `lam` is still being chosen.
  ladder = (args.knots,) if args.knots else KNOT_LADDER
  report = {}
  best, best_score = None, np.inf

  for n_knots in ladder:
    for lam in lams:
      score = cv_rmse(X, z, n_knots, lam, args)
      entry = {"cv_rmse": None if not np.isfinite(score) else round(score, 5)}

      if np.isfinite(score) and args.envelope is not None:
        lo, hi = extension_range(X, z, n_knots, lam, args)
        entry["range"] = [round(lo, 3), round(hi, 3)]
        if lo < -args.envelope or hi > 1.0 + args.envelope:
          entry["rejected"] = "leaves the envelope"
          report[f"k{n_knots}_lam{lam:g}"] = entry
          continue

      report[f"k{n_knots}_lam{lam:g}"] = entry
      if score < best_score:
        best, best_score = (n_knots, lam), score

  if best is None:
    raise ValueError(
      "No candidate was both fittable and inside the envelope. Widen "
      "--envelope, or pass --knots and --lam to force one."
    )
  return best[0], best[1], report


def evaluate(X, z, n_knots, lam, args, radius=None):
  """Fit and sample onto the plotting grid. Returns gx, gy, Z, mask, surface.

  Z is the surface everywhere on the grid; `mask` is where a paragraph is close
  enough for it to be a fit rather than an extension. Under `--domain support`
  the extension is thrown away here (Z is NaN outside the mask) because that
  domain's claim is that there is nothing to say out there. Under `--domain
  unit` both are kept and the drawing decides how to show the difference.
  """
  bx, by = breaks_for(X, n_knots, args)
  surface = bs.fit(X[:, 0], X[:, 1], z, args.degree, bx, by, lam,
                   penalty_order=args.penalty_order)

  gx, gy = np.meshgrid(np.linspace(bx[0], bx[-1], args.resolution),
                       np.linspace(by[0], by[-1], args.resolution))
  Z = surface.grid(gx, gy)

  if radius is None:
    radius = bs.nn_radius(X[:, 0], X[:, 1], args.support_percentile)
  mask = bs.support_mask(gx, gy, X[:, 0], X[:, 1], radius)
  if args.domain != "unit":
    Z = np.where(mask, Z, np.nan)
  return gx, gy, Z, mask, surface


def zlimits(Z):
  """(0, 1) unless the surface leaves it, in which case show where it went."""
  lo = min(0.0, float(np.floor(np.nanmin(Z) * 10) / 10))
  hi = max(1.0, float(np.ceil(np.nanmax(Z) * 10) / 10))
  return lo, hi


def draw_control_net(ax, surface):
  """The control points and the net joining them, over the surface.

  Worth drawing only when there are few enough of them to read, which is the
  case `--control` exists for. The points sit at the Greville abscissae, not at
  the reference's Cx/Cy least-squares solve -- see `Surface.control_grid`.
  """
  cx, cy, cz = surface.control_grid()
  for i in range(cx.shape[0]):
    ax.plot(cx[i, :], cy[i, :], cz[i, :], color="#e34948", lw=0.8, alpha=0.8)
  for j in range(cx.shape[1]):
    ax.plot(cx[:, j], cy[:, j], cz[:, j], color="#e34948", lw=0.8, alpha=0.8)
  ax.scatter(cx.ravel(), cy.ravel(), cz.ravel(), c="#e34948", s=18,
             depthshade=False)


def draw(ax, X, z, gx, gy, Z, mask, emotion, args, points=True, surface=None):
  """The surface, with the extension faded behind the fitted part."""
  if args.domain == "unit":
    # Drawn whole and then again masked to the support, rather than as two
    # complementary patches: complementary patches leave a one-cell seam along
    # a boundary this ragged, and the seam reads as a crack in the surface.
    ax.plot_surface(gx, gy, Z, cmap=CMAP, vmin=0.0, vmax=1.0, linewidth=0,
                    antialiased=True, alpha=0.25, rstride=3, cstride=3)
    drawn = np.where(mask, Z, np.nan)
  else:
    drawn = Z

  surf = ax.plot_surface(gx, gy, drawn, cmap=CMAP, vmin=0.0, vmax=1.0,
                         linewidth=0, antialiased=True, alpha=0.9,
                         rstride=2, cstride=2)
  if points:
    # The scores the surface was fitted to, so the reader can see what it left out.
    ax.scatter(X[:, 0], X[:, 1], z, c="#33322e", s=3, alpha=0.35,
               depthshade=False)

  lo, hi = zlimits(Z)
  if surface is not None and args.control_net:
    draw_control_net(ax, surface)
    # A control point does not lie on the surface -- the fit only approaches its
    # net -- so it can sit outside the surface's own range and would be clipped
    # away by limits taken from Z alone.
    lo = min(lo, float(np.floor(surface.coeff.min() * 10) / 10))
    hi = max(hi, float(np.ceil(surface.coeff.max() * 10) / 10))
  ax.set_zlim(lo, hi)
  if lo < 0.0 or hi > 1.0:
    # A score cannot be here; the surface is, and the plane says where the
    # legal range ended.
    ax.plot_surface(gx, gy, np.zeros_like(Z), color="#b0afa8", alpha=0.12,
                    linewidth=0, rstride=gx.shape[0] - 1, cstride=gx.shape[1] - 1)

  axis = "PC1' " if args.domain == "unit" else "PC1"
  ax.set_xlabel(axis)
  ax.set_ylabel(axis.replace("1", "2"))
  ax.set_zlabel(emotion)
  return surf


def surface_plot(X, z, gx, gy, Z, mask, emotion, subtitle, args, output_path,
                 surface=None):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection="3d")
  surf = draw(ax, X, z, gx, gy, Z, mask, emotion, args, surface=surface)
  ax.set_title(f"{emotion} landscape over the PCA map\n{subtitle}", fontsize=11)
  fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=emotion)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def grid_plot(panels, args, subtitle, output_path):
  """Every emotion in one figure, drawn identically.

  The per-emotion figures answer "what does this emotion's landscape look like";
  this answers "how do they differ", which is a comparison and therefore needs
  every panel on the same footing -- same colour scale (0 to 1, as everywhere
  else in the repo), same viewing angle, same z range across all six. Without
  the shared z range, six autoscaled panels make a flat emotion look as
  mountainous as a varied one.
  """
  cols = 3
  rows = int(np.ceil(len(panels) / cols))
  fig = plt.figure(figsize=(5.5 * cols, 4.8 * rows))

  lo = min(zlimits(p["Z"])[0] for p in panels)
  hi = max(zlimits(p["Z"])[1] for p in panels)

  surf = None
  for i, p in enumerate(panels):
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    surf = draw(ax, p["X"], p["z"], p["gx"], p["gy"], p["Z"], p["mask"],
                p["emotion"], args, points=args.grid_points,
                surface=p["surface"])
    ax.set_zlim(lo, hi)
    ax.set_title(f"{p['emotion']}  ({p['knots']} knots, lam {p['lam']:g})",
                 fontsize=11)
    ax.tick_params(labelsize=6)
    ax.set_xlabel(ax.get_xlabel(), fontsize=8)
    ax.set_ylabel(ax.get_ylabel(), fontsize=8)
    ax.set_zlabel("score", fontsize=8)

  fig.suptitle(subtitle, fontsize=14)
  fig.colorbar(surf, ax=fig.axes, shrink=0.5, pad=0.02, label="score")
  fig.savefig(output_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"\nwrote {output_path}")


def sweep_plot(X, z, emotion, ladder, chosen, lam, args, radius, output_path):
  """One panel per knot count: under-fitted dome to over-fitted ripple."""
  fig = plt.figure(figsize=(5 * len(ladder), 5))

  for i, n_knots in enumerate(ladder):
    ax = fig.add_subplot(1, len(ladder), i + 1, projection="3d")
    try:
      gx, gy, Z, mask, panel = evaluate(X, z, n_knots, lam, args, radius)
      draw(ax, X, z, gx, gy, Z, mask, emotion, args, points=False,
           surface=panel)
    except (ValueError, np.linalg.LinAlgError) as exc:
      ax.text2D(0.5, 0.5, f"not fittable\n{type(exc).__name__}",
                ha="center", transform=ax.transAxes, fontsize=8)
    ax.set_title(f"{n_knots} knots" + ("  (CV)" if n_knots == chosen else ""),
                 fontsize=11)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("PC1'" if args.domain == "unit" else "PC1", fontsize=8)
    ax.set_ylabel("PC2'" if args.domain == "unit" else "PC2", fontsize=8)
    ax.set_zlabel(emotion, fontsize=8)

  fig.suptitle(f"{emotion} landscape vs knot count", fontsize=14)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--model", default="bge-m3", type=str)
  parser.add_argument("--emotions", nargs="*", default=None,
                      help="Subset of emotions (default: all).")
  parser.add_argument("--domain", default="support", choices=("support", "unit"),
                      help="support: cover the paragraphs only. unit: rescale "
                           "PC1/PC2 into [0, 1] and cover the whole square.")
  parser.add_argument("--degree", default=None, type=int,
                      help="Spline degree per axis. Default 3 (cubic); the "
                           "reference's own surface example uses 4. Lowered "
                           "automatically if --control asks for fewer "
                           "coefficients than the degree can carry.")
  parser.add_argument("--knots", default=None, type=int,
                      help="Breakpoints per axis. Default: chosen by k-fold CV.")
  parser.add_argument("--control", default=None, type=int,
                      help="Control points per axis, the other way of saying "
                           "the same thing: coefficients = knots + degree - 1. "
                           "--control 3 is a single biquadratic patch, 9 "
                           "control points for the whole plane.")
  parser.add_argument("--lam", default=None, type=float,
                      help="Penalty weight. Default: 1e-6 under --domain "
                           "support (a ridge, only there to keep the system "
                           "solvable), chosen by CV under --domain unit (a "
                           "roughness penalty, which is the smoothing knob).")
  parser.add_argument("--penalty-order", default=None, type=int,
                      choices=(0, 1, 2),
                      help="0: the reference's ridge. 1: first differences, so the "
                           "surface flattens off the data. 2: second "
                           "differences, so it continues the edge slope. "
                           "Default: 0 under --domain support, 1 under unit.")
  parser.add_argument("--margin", default=0.08, type=float,
                      help="--domain unit: fraction of the unit square left "
                           "outside the paragraphs, on the long axis.")
  parser.add_argument("--envelope", default=0.5, type=float,
                      help="--domain unit: reject a candidate whose surface "
                           "leaves [-envelope, 1+envelope] anywhere on the "
                           "square. Pass a negative number to disable.")
  parser.add_argument("--pad", default=0.02, type=float,
                      help="--domain support: widen the knot span by this "
                           "fraction of the data range on each side.")
  parser.add_argument("--folds", default=5, type=int)
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--resolution", default=200, type=int)
  parser.add_argument("--support-percentile", default=99.0, type=float,
                      help="Percentile of the nearest-neighbour distances used "
                           "as the radius a grid cell must have a paragraph "
                           "within to count as supported.")
  parser.add_argument("--sweep", action="store_true",
                      help="Also draw the landscape across the knot ladder.")
  parser.add_argument("--grid", action="store_true",
                      help="Also draw every emotion in one figure, on a shared "
                           "z range.")
  parser.add_argument("--grid-points", action="store_true",
                      help="Draw the paragraph scatter in the grid panels too. "
                           "Off by default: at panel size 789 points hide the "
                           "surface they are there to justify.")
  parser.add_argument("--control-net", action="store_true",
                      help="Draw the control points and the net joining them. "
                           "Readable only when there are few, which is what "
                           "--control is for.")
  args = parser.parse_args()

  # The defaults that depend on another flag. Spelled out here rather than in
  # the argparse defaults so that `params.json` records the value actually used.
  #
  # --control is a second spelling of --knots: a tensor-product B-spline has
  # knots + degree - 1 coefficients per axis, and "control points" is what those
  # coefficients are called when you are thinking about the net rather than the
  # basis. Asking for 3 x 3 therefore also decides the degree -- 3 control
  # points on an axis is a single quadratic span, and a cubic cannot be carried
  # by fewer than 4 -- so an unset --degree is lowered to fit rather than
  # failing on an arithmetic the caller had no reason to have done.
  if args.control is not None:
    if args.knots is not None:
      raise ValueError("--control and --knots say the same thing; pass one.")
    if args.control < 2:
      raise ValueError("--control must be at least 2.")
    if args.degree is None and args.control - 1 < 3:
      args.degree = args.control - 1
      print(f"--control {args.control} cannot carry a cubic; using degree "
            f"{args.degree}, i.e. a single "
            f"{'quadratic' if args.degree == 2 else 'linear'} span per axis.")
    args.degree = 3 if args.degree is None else args.degree
    args.knots = args.control - args.degree + 1
    if args.knots < 2:
      raise ValueError(
        f"{args.control} control points per axis cannot carry degree "
        f"{args.degree}: that needs {args.degree + 1} or more. Lower --degree."
      )
  args.degree = 3 if args.degree is None else args.degree

  if args.penalty_order is None:
    args.penalty_order = 1 if args.domain == "unit" else 0
  if args.lam is None and args.domain != "unit":
    args.lam = 1e-6
  if args.envelope is not None and args.envelope < 0:
    args.envelope = None
  if args.domain != "unit":
    args.envelope = None
  if args.penalty_order == 0 and args.domain == "unit":
    raise ValueError(
      "--domain unit with --penalty-order 0 fills the empty part of the square "
      "with zeros: the ridge pulls unsupported coefficients to 0 rather than "
      "toward their neighbours. Use --penalty-order 1 or 2."
    )

  paragraphs = load_paragraphs(args.book)
  emotions, matrix = load_scores(args.book, paragraphs)
  X = load_pca(args.book, args.model)
  if len(X) != len(paragraphs):
    raise ValueError(f"pca ({len(X)}) and paragraphs ({len(paragraphs)}) "
                     f"mismatch.")

  transform = None
  print(f"{len(X)} paragraphs | PC1 {X[:, 0].min():.3f}..{X[:, 0].max():.3f} "
        f"| PC2 {X[:, 1].min():.3f}..{X[:, 1].max():.3f}")
  if args.domain == "unit":
    X, transform = bs.to_unit_square(X, args.margin)
    print(f"rescaled into the unit square by one shared scale "
          f"{transform['scale']:.4f}: PC1' {X[:, 0].min():.3f}.."
          f"{X[:, 0].max():.3f} | PC2' {X[:, 1].min():.3f}..{X[:, 1].max():.3f}")

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

  figure = paths.SURFACE_BSPLINE_UNIT if args.domain == "unit" \
      else paths.SURFACE_BSPLINE
  projection = "pca_unit" if args.domain == "unit" else paths.BASE_PROJECTION
  output_dir = paths.out_dir(
    args.book, args.model, os.path.join(paths.SURFACE, figure),
    {"d": args.degree, "p": args.penalty_order, "k": args.knots,
     "lam": args.lam},
  )

  # One radius for every emotion, since the mask is about where the paragraphs
  # are and that does not change between emotions.
  radius = bs.nn_radius(X[:, 0], X[:, 1], args.support_percentile)
  print(f"support radius {radius:.4f} ({args.support_percentile:g}th pct "
        f"nearest-neighbour distance)")

  chosen, searches, panels = {}, {}, []
  for emotion in selected:
    z = matrix[:, emotions.index(emotion)]
    print(f"\n{emotion}:")

    if args.knots and (args.lam is not None):
      n_knots, lam = args.knots, args.lam
      print(f"  {n_knots} knots per axis, lam {lam:g} (given)")
    else:
      n_knots, lam, report = choose_parameters(X, z, args)
      searches[emotion] = report
      baseline = float(np.std(z))
      score = report[f"k{n_knots}_lam{lam:g}"]["cv_rmse"]
      n_rejected = sum("rejected" in e for e in report.values())
      print(f"  searched {len(report)} candidates, {n_rejected} rejected for "
            f"leaving the envelope")
      print(f"  {n_knots} knots per axis, lam {lam:g} (CV rmse {score:.4f} vs "
            f"{baseline:.4f} for the flat mean -> explains "
            f"{1 - (score / baseline) ** 2:.1%} of the variance)")

    chosen[emotion] = {"knots": n_knots, "lam": lam}
    gx, gy, Z, mask, surface = evaluate(X, z, n_knots, lam, args, radius)

    empty = bs.unsupported(X[:, 0], X[:, 1], args.degree,
                           surface.breaks_x, surface.breaks_y)
    residual = surface(X[:, 0], X[:, 1]) - z
    print(f"  {surface.shape[0]}x{surface.shape[1]} = {surface.coeff.size} "
          f"coefficients, {empty} with no paragraph behind them")
    print(f"  in-sample rmse {np.sqrt(np.mean(residual ** 2)):.4f} | "
          f"relief {np.nanmin(Z):.2f}..{np.nanmax(Z):.2f}")
    if args.domain == "unit":
      inside = Z[mask]
      print(f"  supported {mask.mean():.0%} of the square "
            f"({np.nanmin(inside):.2f}..{np.nanmax(inside):.2f}), "
            f"the other {1 - mask.mean():.0%} is extension")
    else:
      print(f"  drawn over {np.isfinite(Z).mean():.0%} of the knot rectangle")

    # Same keys as the kernel surfaces' field_<emotion>_pca.npz, so the geodesic
    # stack can walk on this terrain without knowing which fit produced it. The
    # spline-specific fields trail. `transform` is None under --domain support
    # and the (scale, offset) back to pca.npy units under --domain unit; `mask`
    # means "supported" in both, but only under support is it also "drawn".
    np.savez(
      os.path.join(output_dir, paths.named("field", emotion, "npz", projection)),
      gx=gx, gy=gy, Z=Z, mask=mask, X=X, y=z, emotion=emotion,
      coeff=surface.coeff, breaks_x=surface.breaks_x,
      breaks_y=surface.breaks_y, degree=args.degree, lam=lam,
      penalty_order=args.penalty_order, n_knots=n_knots,
      support_radius=radius, domain=args.domain,
      transform=json.dumps(
        None if transform is None
        else {"scale": transform["scale"], "offset": transform["offset"].tolist()}
      ),
    )

    panels.append({"emotion": emotion, "X": X, "z": z, "gx": gx, "gy": gy,
                   "Z": Z, "mask": mask, "surface": surface, "knots": n_knots,
                   "lam": lam})

    where = ("the unit square, extension faded" if args.domain == "unit"
             else "the paragraphs only")
    surface_plot(
      X, z, gx, gy, Z, mask, emotion,
      f"least-squares B-spline, degree {args.degree}, {n_knots} knots per axis,"
      f"\nlam {lam:g}, penalty order {args.penalty_order}, over {where}",
      args, os.path.join(output_dir, paths.named("surface", emotion, "png",
                                                 projection)),
      surface=surface,
    )

    if args.sweep:
      # Sorted and deduplicated so the panels read left to right from smooth to
      # ragged whatever the CV chose, and the CV's choice is not drawn twice.
      ladder = sorted({2, 6, n_knots, 12})
      sweep_plot(X, z, emotion, ladder, n_knots, lam, args, radius,
                 os.path.join(output_dir,
                              paths.named("sweep", emotion, "png", projection)))

  if args.grid:
    # The net size goes in the title only when every panel shares one. Left to
    # the CV the emotions choose different knot counts, and a single number in
    # the title would then be a claim about panels it does not describe.
    nets = {p["surface"].shape for p in panels}
    if len(nets) == 1:
      nx, ny = nets.pop()
      net = f"{nx}x{ny} control points"
    else:
      net = "control points chosen per emotion by CV"
    grid_plot(
      panels, args,
      f"{args.book} / {args.model}: least-squares B-spline landscapes, {net}"
      + (", over the unit square" if args.domain == "unit" else ""),
      os.path.join(output_dir, paths.named("grid", "emotions", "png",
                                           projection)),
    )

  paths.stamp(output_dir, __file__, args,
              stack="surface/bspline/bspline_surface",
              estimator="least-squares tensor-product B-spline "
                        "(after LorenzoPratesi/B-spline-Curves-and-Surfaces)",
              emotions=selected, chosen=chosen, search=searches,
              support_radius=radius,
              transform=None if transform is None else {
                "scale": transform["scale"],
                "offset": transform["offset"].tolist(),
              })

  print(f"\nDone.\n{output_dir}")


if __name__ == "__main__":
  main()
