"""Least-squares tensor-product B-spline surface, ported from the MATLAB vendor.

This is `vendor/B-spline-Curves-and-Surfaces/bs_least_square_2.m` in numpy: the
same clamped uniform knots, the same Cox-de Boor basis, the same ridge on the
normal equations, the same QR-plus-back-substitution solve. `check_port.py`
holds it to the vendor's own surface example.

Why a port and not a bridge: MATLAB is not installed, the upstream repo carries
no license (see `vendor/.../PROVENANCE.md`), and `bspline_basis.m` is written
one basis function per call with a recursive descent -- for 789 paragraphs and a
few hundred coefficients that is thousands of redundant recurrences. The port
builds the whole order ladder once, so the entire basis matrix costs one pass.

How this differs from the kernel smoothers in `geometry/smoothers.py`, which fit
the same z = f(PC1, PC2):

  A kernel smoother is local and has no parameters -- to predict at (x, y) it
  reweights the paragraphs near (x, y), every time. A B-spline surface is a
  *fit*: `(ncoeff_x, ncoeff_y)` control heights are solved for once and then the
  surface is that basis expansion, everywhere, with no further reference to the
  data. The knot count plays the role the bandwidth plays there, but from the
  other direction -- more knots is *less* smoothing.

  Its weakness on this data is the tensor grid. Paragraphs are a clumped point
  cloud in the PCA plane, not a lattice, so some basis cells contain no
  paragraph at all and their coefficient is unconstrained. `lam` is what keeps
  those solvable, and `unsupported()` is what reports how many there were --
  a fit with many is extrapolating, however good its residuals look.
"""

import numpy as np


def clamped_knots(breaks, degree):
  """The vendor's knot convention: repeat each endpoint `degree` times.

  `breaks` is the breakpoint vector including both ends, as passed to
  `bs_least_square_2` -- the clamping is the fitter's job there, not the
  caller's, and it is kept here so that the two take the same argument.
  """
  breaks = np.asarray(breaks, dtype=np.float64).ravel()
  if breaks.ndim != 1 or breaks.size < 2:
    raise ValueError("breaks must be a vector with at least two entries.")
  if np.any(np.diff(breaks) < 0):
    raise ValueError("Knot vector values should be nondecreasing.")
  return np.concatenate([
    np.repeat(breaks[0], degree), breaks, np.repeat(breaks[-1], degree),
  ])


def basis_matrix(x, knots, order):
  """All B-spline basis functions of `order` at `x`: an (len(x), m) matrix.

  `m = len(knots) - order`, matching `bspline_basis.m`'s bound on its interval
  index j. `order` is degree + 1 (2 for linear, 3 for quadratic), as in the
  vendor.

  The recurrence is the vendor's, read bottom-up instead of top-down: order 1 is
  the interval indicator, and each subsequent order is a two-term combination of
  the one below. Zero-width knot spans -- which the clamped ends are full of --
  give 0/0, and are taken as 0 exactly as `bspline_basis.m` does by guarding the
  denominator.
  """
  x = np.asarray(x, dtype=np.float64).ravel()
  t = np.asarray(knots, dtype=np.float64).ravel()
  m = t.size - order
  if m < 1:
    raise ValueError(f"Knot vector of length {t.size} is too short for order "
                     f"{order}; it supports {m} basis functions.")

  # Order 1: the indicator of [t_i, t_{i+1}). The final span is closed on the
  # right so that x == t[-1] is not silently outside every basis function --
  # `bspline_basis.m` treats it as the same special case.
  left = t[:-1][None, :]
  right = t[1:][None, :]
  xc = x[:, None]
  B = np.where(right < t[-1], (left <= xc) & (xc < right), left <= xc)
  B = B.astype(np.float64)

  for r in range(2, order + 1):
    n_r = t.size - r                     # basis functions at this order
    lo_num = xc - t[:n_r][None, :]
    lo_den = (t[r - 1:r - 1 + n_r] - t[:n_r])[None, :]
    hi_num = t[r:r + n_r][None, :] - xc
    hi_den = (t[r:r + n_r] - t[1:1 + n_r])[None, :]

    lo = np.divide(lo_num, lo_den, out=np.zeros_like(lo_num),
                   where=lo_den != 0)
    hi = np.divide(hi_num, hi_den, out=np.zeros_like(hi_num),
                   where=hi_den != 0)
    B = lo * B[:, :n_r] + hi * B[:, 1:n_r + 1]

  return B[:, :m]


def to_unit_square(X, margin=0.08):
  """Map PCA coordinates into [0, 1]^2. Returns (U, transform).

  One scale for both axes, not one per axis: PCA is metric, so a unit of PC1 and
  a unit of PC2 are the same unit, and stretching the axes independently to fill
  the square would make an isotropic knot grid anisotropic in the data's own
  geometry. The cloud is centred in the square and the shorter axis simply does
  not reach the edges.

  `margin` is the fraction of the square left as border on the long axis, so the
  fit has somewhere to be an extension rather than starting one at the last
  paragraph.

  `transform` is `{"scale", "offset"}` with `U = X * scale + offset`, recorded
  alongside every artifact so a surface fitted here can be put back into
  `pca.npy` units.
  """
  X = np.asarray(X, dtype=np.float64)
  lo, hi = X.min(axis=0), X.max(axis=0)
  span = float((hi - lo).max())
  if span <= 0:
    raise ValueError("Cannot rescale: the point cloud has no extent.")

  scale = (1.0 - 2.0 * margin) / span
  centre = (lo + hi) / 2.0
  offset = 0.5 - centre * scale
  return X * scale + offset, {"scale": float(scale), "offset": offset}


def difference_matrix(n, order):
  """The `order`-th difference operator on `n` coefficients, (n-order, n)."""
  if order < 1:
    raise ValueError("order must be at least 1.")
  if n <= order:
    raise ValueError(f"{n} coefficients cannot carry a difference of order "
                     f"{order}.")
  D = np.eye(n)
  for _ in range(order):
    D = np.diff(D, axis=0)
  return D


def penalty_matrix(nx, ny, order):
  """A P-spline roughness penalty over the coefficient grid, x-major.

  `lam * I` -- the vendor's ridge -- pulls every coefficient toward zero, which
  is a sane way to keep the system solvable and a *terrible* way to fill a cell
  with no data in it: the surface there falls to a score of 0, and does it
  abruptly, because nothing ties that cell to its neighbours.

  This penalizes differences between neighbouring coefficients instead (Eilers
  and Marx's P-splines). Where there is data the penalty is a mild smoother and
  the data decides; where there is none the penalty is the *only* thing acting,
  so the coefficient is whatever continues its neighbours most smoothly. That is
  what makes the surface extendable past the paragraphs at all.

  The order sets what "continues" means, and it is the whole safety argument:

    order 1   flat continuation. Far from the data the surface levels off at the
              nearby scores. Wrong, but wrong *inside* the range a score can
              take, and visibly an extension.
    order 2   linear continuation: the slope at the edge of the cloud is carried
              outward. Over a domain this much larger than the cloud that walks
              straight out of [0, 1] -- `closed-surface/poisson.py` refuses a
              local *linear* extension for exactly this reason. Available, not
              the default.
  """
  Dx = difference_matrix(nx, order)
  Dy = difference_matrix(ny, order)
  return (np.kron(Dx.T @ Dx, np.eye(ny)) +
          np.kron(np.eye(nx), Dy.T @ Dy))


def qr_solve(A, b):
  """`QR_solve.m`: Householder QR, then back substitution.

  numpy would do this in one `solve`, and the answer is the same; it is spelled
  out because the vendor spells it out, and because the back substitution is
  where a rank-deficient A announces itself as a division by a zero pivot rather
  than as a silent least-norm answer.
  """
  Q, R = np.linalg.qr(A)
  y = Q.T @ b
  n = R.shape[0]
  out = np.zeros(n, dtype=np.float64)
  for j in range(n - 1, -1, -1):
    if R[j, j] == 0.0:
      raise np.linalg.LinAlgError(
        f"Zero pivot at row {j} of R: the normal equations are singular. "
        f"Raise lam, or use fewer knots."
      )
    out[j] = (y[j] - R[j, :] @ out) / R[j, j]
  return out


class Surface:
  """A fitted z = f(x, y): the control heights plus the knots they hang on.

  `coeff` is (ncoeff_x, ncoeff_y) -- the vendor's flat x-major column ordering
  reshaped, since a grid is what it means.
  """

  def __init__(self, degree, breaks_x, breaks_y, coeff, control_xy=None):
    self.degree = degree
    self.breaks_x = np.asarray(breaks_x, dtype=np.float64)
    self.breaks_y = np.asarray(breaks_y, dtype=np.float64)
    self.knots_x = clamped_knots(self.breaks_x, degree)
    self.knots_y = clamped_knots(self.breaks_y, degree)
    self.coeff = np.asarray(coeff, dtype=np.float64)
    self.control_xy = control_xy

  @property
  def shape(self):
    return self.coeff.shape

  def __call__(self, x, y):
    """Evaluate at scattered (x, y). Shapes must match; result matches them."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
      raise ValueError(f"x {x.shape} and y {y.shape} must have the same shape.")
    Bx = basis_matrix(x.ravel(), self.knots_x, self.degree + 1)
    By = basis_matrix(y.ravel(), self.knots_y, self.degree + 1)
    # einsum rather than building the (n, ncoeff_x*ncoeff_y) tensor product:
    # on a 300x300 grid that matrix is 90000 x ~200 doubles for no reason.
    return np.einsum("ij,jk,ik->i", Bx, self.coeff, By).reshape(x.shape)

  def grid(self, gx, gy):
    """Evaluate on a meshgrid pair, returning an array of the grid's shape."""
    return self(gx, gy)

  def control_grid(self):
    """The control net as (X, Y, Z) meshes, for drawing it over the surface.

    x and y come from the Greville abscissae of the knot vectors -- where a
    coefficient's basis function is centred -- which is where the control point
    sits when the fit is a height field. `control_xy` from the vendor's Cx/Cy
    solve is *not* used for this: those are the least-squares fits of x and y by
    the same basis, so they reproduce the same net only where the data fills the
    domain, and wander where it does not.
    """
    X, Y = np.meshgrid(greville(self.knots_x, self.degree),
                       greville(self.knots_y, self.degree), indexing="ij")
    return X, Y, self.coeff


def greville(knots, degree):
  """Greville abscissae: the x each coefficient is anchored at."""
  knots = np.asarray(knots, dtype=np.float64)
  m = knots.size - degree - 1
  return np.array([knots[i + 1:i + 1 + degree].mean() if degree else knots[i]
                   for i in range(m)])


def uniform_breaks(v, n, pad=0.0):
  """`n` uniform breakpoints spanning the data, optionally padded outwards.

  `build_knot_vector.m` builds knots on [0, 1] because its examples fit a
  parametrized curve; a surface over the PCA plane needs them on the data's own
  range, which is what `ex_15` does with `linspace(xMin, xMax, nknots)`.

  `pad` widens the span by that fraction on each side. Without it the outermost
  basis functions are supported by the handful of paragraphs at the extreme edge
  of the point cloud, which is exactly where their coefficients are least
  determined.
  """
  lo, hi = float(np.min(v)), float(np.max(v))
  if hi <= lo:
    raise ValueError("Cannot place knots: the coordinate is constant.")
  margin = (hi - lo) * pad
  return np.linspace(lo - margin, hi + margin, n)


def fit(x, y, z, degree, breaks_x, breaks_y, lam=0.0, solver="qr",
        penalty_order=0):
  """`bs_least_square_2`: least-squares tensor-product surface through (x, y, z).

  Returns a `Surface`. `lam` weights the penalty on `B'B`; `solver` is "qr" for
  the vendor's own path or "lstsq" for numpy's SVD least squares, which answers
  the same question but survives a singular system instead of raising.

  `penalty_order` 0 is the vendor's plain ridge, `lam * I`. 1 or 2 substitute
  the difference penalty from `penalty_matrix` -- required if the surface is to
  be evaluated anywhere the paragraphs are not, since a ridge leaves those cells
  at zero rather than continuing their neighbours.

  The vendor also solves the same system for x and for y, giving the control
  net's own coordinates; those are carried on the result as `control_xy` for
  parity, and are not used to evaluate the height.
  """
  x = np.asarray(x, dtype=np.float64).ravel()
  y = np.asarray(y, dtype=np.float64).ravel()
  z = np.asarray(z, dtype=np.float64).ravel()
  if not (x.size == y.size == z.size):
    raise ValueError(f"x ({x.size}), y ({y.size}) and z ({z.size}) must agree.")

  tx = clamped_knots(breaks_x, degree)
  ty = clamped_knots(breaks_y, degree)
  Bx = basis_matrix(x, tx, degree + 1)
  By = basis_matrix(y, ty, degree + 1)
  nx, ny = Bx.shape[1], By.shape[1]
  # Only under the ridge. A difference penalty ties neighbouring coefficients
  # together, so more coefficients than points is not the failure it would be
  # here -- it is the ordinary P-spline setting, where the knots are deliberately
  # generous and `lam` does the work.
  if penalty_order == 0 and x.size < nx * ny:
    raise ValueError(
      f"{x.size} points cannot determine {nx}x{ny} = {nx * ny} coefficients. "
      f"Use fewer knots, or a difference penalty."
    )

  # The row-wise Kronecker product: column (j, k) is Bx[:, j] * By[:, k], in the
  # vendor's x-major order.
  B = (Bx[:, :, None] * By[:, None, :]).reshape(x.size, nx * ny)

  P = (np.eye(nx * ny) if penalty_order == 0
       else penalty_matrix(nx, ny, penalty_order))
  A = B.T @ B + lam * P
  rhs = B.T @ np.column_stack([x, y, z])

  if solver == "qr":
    sol = np.column_stack([qr_solve(A, rhs[:, i]) for i in range(3)])
  elif solver == "lstsq":
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
  else:
    raise ValueError(f"Unknown solver {solver!r}; expected 'qr' or 'lstsq'.")

  return Surface(degree, breaks_x, breaks_y, sol[:, 2].reshape(nx, ny),
                 control_xy=sol[:, :2].reshape(nx, ny, 2))


def support(x, y, degree, breaks_x, breaks_y):
  """How much data each coefficient sees: an (ncoeff_x, ncoeff_y) weight sum.

  A zero here is a coefficient no paragraph constrains. It still gets a value --
  the ridge gives it one -- but that value is not a measurement, and the surface
  over that cell is whatever the penalty preferred. Reported rather than fixed,
  because the fix is a decision (fewer knots, or a mask) that belongs to the
  caller.
  """
  Bx = basis_matrix(x, clamped_knots(breaks_x, degree), degree + 1)
  By = basis_matrix(y, clamped_knots(breaks_y, degree), degree + 1)
  return Bx.T @ By


def unsupported(x, y, degree, breaks_x, breaks_y, tol=1e-9):
  """The count of coefficients with no data behind them."""
  return int((support(x, y, degree, breaks_x, breaks_y) <= tol).sum())


def support_mask(gx, gy, x, y, radius):
  """True where a grid cell has a paragraph within `radius`.

  The fitted surface is defined over the whole knot rectangle, including the
  corners of the PCA plane where the book has no paragraphs at all. Plotting it
  there draws a height for a region no text occupies. This is the same honesty
  the kernel surfaces get from their density floor, done by distance because a
  spline fit has no density to threshold.
  """
  pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
  cells = np.column_stack([gx.ravel(), gy.ravel()])
  # Chunked so a 300x300 grid against 800 paragraphs does not allocate one
  # 90000 x 800 distance matrix.
  keep = np.zeros(cells.shape[0], dtype=bool)
  for start in range(0, cells.shape[0], 4096):
    block = cells[start:start + 4096]
    d = np.linalg.norm(block[:, None, :] - pts[None, :, :], axis=2)
    keep[start:start + 4096] = d.min(axis=1) <= radius
  return keep.reshape(gx.shape)


def nn_radius(x, y, percentile=99.0):
  """A default `radius` for `support_mask`, read off the data's own spacing.

  The distance from a paragraph to its nearest neighbour is how far apart the
  samples actually are here; a grid cell further than that from every paragraph
  is between samples in name only. The percentile is over paragraphs, so it is
  set by the sparse part of the cloud rather than the dense core.

  It defaults near the top of that distribution, not the middle: at the 90th
  percentile the mask punches holes *inside* the point cloud, between paragraphs
  that are merely further apart than typical, and a hole there says "no data"
  about a region that has data on all sides. What the mask is for is the corners
  of the plane that the book never visits.
  """
  pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
  d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
  np.fill_diagonal(d, np.inf)
  return float(np.percentile(d.min(axis=1), percentile))
