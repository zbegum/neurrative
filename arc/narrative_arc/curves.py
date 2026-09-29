"""Smooth curves through the windows, fitted with bspline-regression.

vendor/bspline_regression is Richard Stebbing's uniform B-spline regression
(https://github.com/rstebbing/bspline-regression). It fits a uniform B-spline of
any degree to points in any dimension -- so the 2-D and 3-D arcs use the same
fitter -- by jointly optimising the control points X and every point's
correspondence u on the curve, minimising

    0.5 * ( sum |Y - M(u, X)|^2  +  lambda * sum |X[j+1] - X[j]|^2 )

with damped Newton or Levenberg-Marquardt. The numerics are entirely theirs;
this module only sets the problem up for an arc and reads the result back.

Two arc-specific choices in the setup:

  initialised by reading order. The library is built for unstructured points,
  whose correspondences have to be guessed. Here the order is known, so u starts
  at each window's reading position spread evenly along the curve, and the
  control points start on the path itself, sampled at evenly spaced reading
  positions. The optimiser refines both, but starting from reading order keeps a
  path that crosses itself (common in t-SNE) from collapsing its two passes onto
  one branch.

  normalised coordinates. A PCA layout spans ~0.1, a 3-D t-SNE one ~100. Points
  are centred and scaled to unit RMS radius before fitting and mapped back
  after, so the solver's trust-region defaults behave the same for every
  projection. (lambda is scale-free regardless: both terms are squared
  distances.)

Knobs: n_control (fewer => smoother, looser), degree (3 = cubic), lambda (larger
pulls adjacent control points together => a shorter, straighter curve).

A uniform B-spline is not clamped, so the curve begins near -- not exactly at --
the opening window, and likewise ends near the last one.
"""

from collections import namedtuple

import numpy as np

from vendor.bspline_regression.fit_uniform_bspline import (
  UniformBSplineLeastSquaresOptimiser)
from vendor.bspline_regression.uniform_bspline import UniformBSpline

SOLVERS = ("dn", "lm")

FitOpts = namedtuple("FitOpts", "n_control degree lambda_ solver max_iter")

# curve: (n_eval, dim) samples in reading order. t_norm: [0, 1] along it, for
# coloring by progression. control_points: the fitted X. u: each window's
# fitted position on the curve, in segment units.
Fit = namedtuple("Fit", "curve t_norm control_points u converged energy")


def fit_curve(coords, opts, n_eval=400):
  """Fit a uniform B-spline through the windows. Returns a Fit."""
  n, dim = coords.shape
  if opts.n_control <= opts.degree:
    raise ValueError(f"--n-control ({opts.n_control}) must exceed --degree "
                     f"({opts.degree}).")
  if opts.lambda_ <= 0:
    raise ValueError(f"--lambda must be > 0 (got {opts.lambda_}).")

  center = coords.mean(axis=0)
  scale = np.sqrt(((coords - center) ** 2).sum(axis=1).mean()) or 1.0
  Y = (coords - center) / scale

  spline = UniformBSpline(opts.degree, opts.n_control, dim)
  sample_at = np.linspace(0, n - 1, opts.n_control)
  X0 = np.column_stack(
    [np.interp(sample_at, np.arange(n), Y[:, k]) for k in range(dim)])
  u0 = spline.uniform_parameterisation(n)

  u, X, converged, states, _, _ = UniformBSplineLeastSquaresOptimiser(
    spline, opts.solver,
  ).minimise(Y, np.ones_like(Y), opts.lambda_, u0, X0, return_all=True,
             max_num_iterations=opts.max_iter)

  curve = spline.M(spline.uniform_parameterisation(n_eval), X)
  return Fit(curve * scale + center, np.linspace(0, 1, n_eval),
             X * scale + center, u, bool(converged), float(states[-1][2]))


def fit_tag(opts):
  """The filename fragment that says which fit, if any, was overlaid."""
  if opts is None:
    return ""
  return (f"_fit-d{opts.degree}-c{opts.n_control}-l{opts.lambda_:g}"
          + ("-lm" if opts.solver == "lm" else ""))
