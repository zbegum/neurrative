"""Hold the port to the vendor's own surface example.

`vendor/.../example/ex_15_bspline_surf_ls.m` fits log(4x^2 + y^2) on a 100x100
grid over [-3, 3] x [-10, 10], with uniform noise of amplitude 0.5, degree 4 and
24 knots per axis, and reports a relative RMSE against the noiseless truth.
MATLAB is not installed here, so this rebuilds that setup in numpy and checks
the port reaches the same accuracy -- agreement with the example, not with a
MATLAB run. See PROVENANCE.md.

The other three checks are properties, not comparisons, and they are the ones
that would actually catch a wrong index in the recurrence:

  partition of unity   the basis functions sum to 1 everywhere inside the knot
                       span. An off-by-one in the clamping breaks this at the
                       ends while leaving the middle looking fine.
  polynomial exactness a degree-d spline reproduces any degree-d polynomial
                       exactly. This is the strongest single statement about the
                       basis, and it fails for any misplaced knot.
  solver agreement     the vendor's QR path and numpy's lstsq give the same
                       coefficients, so the hand-written back substitution is
                       not quietly wrong.

Run: python surface/bspline/check_port.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bspline_surface as bs


def check_partition_of_unity():
  degree = 3
  breaks = np.linspace(-2.0, 5.0, 9)
  knots = bs.clamped_knots(breaks, degree)
  x = np.linspace(breaks[0], breaks[-1], 501)
  total = bs.basis_matrix(x, knots, degree + 1).sum(axis=1)
  worst = float(np.abs(total - 1.0).max())
  print(f"  partition of unity: max |sum(B) - 1| = {worst:.2e}")
  assert worst < 1e-12, "basis does not sum to 1; the clamping is wrong"


def check_polynomial_exactness():
  degree = 3
  bx = np.linspace(-1.0, 2.0, 7)
  by = np.linspace(0.0, 3.0, 6)
  rng = np.random.default_rng(0)
  x = rng.uniform(bx[0], bx[-1], 4000)
  y = rng.uniform(by[0], by[-1], 4000)
  # A polynomial of total degree <= 3 in each variable.
  z = 1.0 - 2.0 * x + 0.5 * y ** 2 + x ** 3 - 0.25 * x * y + y ** 3

  surface = bs.fit(x, y, z, degree, bx, by, lam=0.0, solver="lstsq")
  worst = float(np.abs(surface(x, y) - z).max())
  print(f"  cubic exactness:    max |fit - truth| = {worst:.2e}")
  assert worst < 1e-8, "a cubic spline must reproduce a cubic exactly"


def check_solvers_agree():
  degree = 3
  bx = np.linspace(0.0, 1.0, 6)
  by = np.linspace(0.0, 1.0, 5)
  rng = np.random.default_rng(1)
  x, y = rng.uniform(0, 1, 3000), rng.uniform(0, 1, 3000)
  z = np.sin(6 * x) * np.cos(4 * y)

  a = bs.fit(x, y, z, degree, bx, by, lam=1e-6, solver="qr")
  b = bs.fit(x, y, z, degree, bx, by, lam=1e-6, solver="lstsq")
  worst = float(np.abs(a.coeff - b.coeff).max())
  print(f"  qr vs lstsq:        max |dC| = {worst:.2e}")
  assert worst < 1e-6, "the hand-written back substitution disagrees with lstsq"


def check_ex_15():
  """The vendor's own example, rebuilt."""
  degree, nx, ny, nknots = 4, 100, 100, 24
  rng = np.random.default_rng(15)

  x = np.linspace(-3.0, 3.0, nx)
  y = np.linspace(-10.0, 10.0, ny)
  gx, gy = np.meshgrid(x, y)
  gx, gy = gx.ravel(), gy.ravel()

  truth = np.log(4 * gx ** 2 + gy ** 2)
  noisy = truth + rng.uniform(-0.25, 0.25, truth.shape)

  bx = np.linspace(-3.0, 3.0, nknots)
  by = np.linspace(-10.0, 10.0, nknots)
  surface = bs.fit(gx, gy, noisy, degree, bx, by, lam=0.0)
  fitted = surface(gx, gy)

  # ex_15's "rrmse": RMSE against the noiseless surface, as a percentage of its
  # mean. The mean of a log is near 1.5 here, so this is a loose figure -- it is
  # what the example prints, so it is what we compare.
  rrmse = np.sqrt(np.mean((truth - fitted) ** 2)) / np.mean(truth) * 100
  noise_rrmse = np.sqrt(np.mean((truth - noisy) ** 2)) / np.mean(truth) * 100
  print(f"  ex_15 surface:      rrmse {rrmse:.2f}% "
        f"(the noise alone is {noise_rrmse:.2f}%)")
  # The fit must be a real improvement on the noise it was given, and must not
  # be so far off that the basis is misplaced. ex_15 lands a few percent.
  assert rrmse < noise_rrmse, "the fit is no better than the raw noisy samples"
  assert rrmse < 10.0, f"rrmse {rrmse:.2f}% is far from the example's few percent"


def check_unit_square():
  """The rescaling is a similarity, and it lands inside the square."""
  rng = np.random.default_rng(2)
  X = rng.normal(size=(400, 2)) * [0.3, 0.1] + [-1.0, 4.0]
  U, tr = bs.to_unit_square(X, margin=0.1)

  assert U.min() >= 0.1 - 1e-12 and U.max() <= 0.9 + 1e-12, \
    "the rescaled cloud leaves the margin"
  # Shape preserved: every pairwise distance scales by the same factor, which is
  # what "one shared scale" has to mean and what an axis-wise rescale would break.
  d0 = np.linalg.norm(X[:50, None, :] - X[None, :50, :], axis=2)
  d1 = np.linalg.norm(U[:50, None, :] - U[None, :50, :], axis=2)
  ratio = d1[d0 > 0] / d0[d0 > 0]
  print(f"  unit square:        distance ratio spread "
        f"{ratio.max() - ratio.min():.2e}, scale {tr['scale']:.4f}")
  assert ratio.max() - ratio.min() < 1e-12, "the rescale is not a similarity"


def check_extension_flattens():
  """Order 1 continues flat off the data; order 2 continues the slope.

  This is the property the `--domain unit` fit rests on, so it is checked rather
  than asserted in a comment: fit a tilted plane on the left half of the square
  only, and ask what each penalty does with the empty right half.
  """
  degree = 3
  breaks = np.linspace(0.0, 1.0, 9)
  rng = np.random.default_rng(3)
  x = rng.uniform(0.0, 0.45, 3000)
  y = rng.uniform(0.0, 1.0, 3000)
  z = 2.0 * x                                    # slope 2, so z spans 0 .. 0.9

  far = (np.full(200, 0.95), np.linspace(0.05, 0.95, 200))
  flat = bs.fit(x, y, z, degree, breaks, breaks, 1e-2, solver="lstsq",
                penalty_order=1)(*far)
  sloped = bs.fit(x, y, z, degree, breaks, breaks, 1e-2, solver="lstsq",
                  penalty_order=2)(*far)
  # Order 1 does not come out exactly at 0.90. What it flattens is the
  # *continuation*, and the edge coefficient it continues from is itself fitted,
  # so it carries a little of the boundary overshoot with it. The claim being
  # checked is that it stays near the edge value rather than near 1.9, which is
  # where the slope would have taken it.
  linear = 2.0 * 0.95
  print(f"  extension at x=0.95: order 1 -> {flat.mean():.2f}, "
        f"order 2 -> {sloped.mean():.2f} (data ends at 0.90, slope gives "
        f"{linear:.2f})")
  assert flat.max() < z.max() + 0.25, "order 1 should not continue the slope"
  assert flat.mean() < (z.max() + linear) / 2, \
    "order 1 is closer to the slope's continuation than to the edge value"
  assert sloped.mean() > flat.mean() + 0.3, "order 2 should continue the slope"


def main():
  print("checking the port against vendor/B-spline-Curves-and-Surfaces:")
  check_partition_of_unity()
  check_polynomial_exactness()
  check_solvers_agree()
  check_ex_15()
  print("checking the extension machinery, which the vendor has no counterpart "
        "for:")
  check_unit_square()
  check_extension_flattens()
  print("all checks passed.")


if __name__ == "__main__":
  main()
