"""
Scatterplot smoothers: fit a surface z = f(x, y) to scattered samples.

Every estimator has the same signature

    predict(X, y, Q, **params) -> (values, support)

where X are the sample positions, y their scores, Q the query points, and
support is the summed kernel weight behind each prediction -- the effective
number of samples speaking for it. Where support is tiny the value is a 0/0
fabrication, so callers mask on it rather than trust the height.

The four implementations trade off differently:

  gaussian_nw       Kernel average, infinite support. Smooth everywhere, but
                    biased at the boundary: with samples on one side only, the
                    average is pulled inward, flattening the edges of the hill.
  epanechnikov_nw   Kernel average with compact support -- optimal in the MSE
                    sense, and strictly local, but it can leave holes where no
                    sample falls inside the window.
  local_linear      Fits a plane at each query instead of an average. This is
                    the fix for the boundary bias: a tilted local fit follows a
                    trend off the edge of the data instead of flattening.
  loess             Local polynomial on a k-nearest-neighbour window, with
                    bisquare reweighting so a handful of wild annotations cannot
                    drag the surface. The neighbourhood adapts to local density.

See https://en.wikipedia.org/wiki/Kernel_smoother
"""

import numpy as np

# Query points are processed in blocks: a full (grid x samples) weight matrix is
# 200*200*2084 entries for P&P, which is large enough to hurt.
_BLOCK = 2048

_EPS = 1e-12


def f64(a):
  """Everything here runs in float64.

  The embeddings -- and so the PCA coordinates -- arrive as float32, whose
  smallest normal is ~1e-38. Kernel weights drop below that long before they are
  actually negligible, and distances and weights want the precision anyway.
  """
  return np.asarray(a, dtype=np.float64)


def weighted_sum(w, y):
  """w @ y, tolerating a spurious BLAS warning.

  Once the weights get small, the BLAS behind `@` reports "divide by zero
  encountered in matmul" even though every input and output is finite and no
  subnormal is involved (einsum on the same data is silent). It is the library
  setting FPU flags that numpy then blames on the product, so the warning is
  suppressed here and the result checked instead of trusted.
  """
  with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    out = w @ y
  if not np.isfinite(out).all():
    raise FloatingPointError("kernel weighting produced non-finite values")
  return out


def _scaled_offsets(q, X, hx, hy):
  """Per-axis offsets, and the squared radius in bandwidth units.

  Two-dimensional only: the local polynomial smoothers need dx and dy separately
  to build their design matrix. Kernels that only need the radius should use
  `scaled_radius2`, which works in any dimension.
  """
  dx = q[:, None, 0] - X[None, :, 0]
  dy = q[:, None, 1] - X[None, :, 1]
  u2 = (dx / hx) ** 2 + (dy / hy) ** 2
  return dx, dy, u2


def scaled_radius2(Q, X, h):
  """Squared distance from each query to each sample, in bandwidth units.

  `h` is either one bandwidth for every axis or one per axis. Any dimension:
  the plane for the emotion landscape, but also 1-D, where the samples are
  paragraph positions in reading order.
  """
  Q, X = f64(Q), f64(X)
  h = np.asarray(h, dtype=np.float64)
  if h.ndim == 0:
    h = np.full(Q.shape[1], float(h))
  if h.shape != (Q.shape[1],):
    raise ValueError(
      f"bandwidth of shape {h.shape} for {Q.shape[1]}-d points; pass one value "
      f"or one per axis."
    )
  u = (Q[:, None, :] - X[None, :, :]) / h
  return np.einsum("qxd,qxd->qx", u, u)


def gaussian_weights(Q, X, h):
  """The Gaussian kernel weight of every (query, sample) pair.

  Exposed because bandwidth selection needs the weight matrix itself, not a
  fitted value: leave-one-out CV subtracts each sample's own weight from its own
  estimate, which it cannot do through `nadaraya_watson`.
  """
  # Distant pairs underflow to 0, which is the right answer: no influence.
  return np.exp(-0.5 * scaled_radius2(Q, X, h))


def _blocks(n):
  for start in range(0, n, _BLOCK):
    yield start, min(start + _BLOCK, n)


def nadaraya_watson(X, y, Q, h):
  """Kernel-weighted average of y at the query points Q. Any dimension.

  The single Gaussian Nadaraya-Watson in this repo. `h` is one bandwidth for
  every axis or one per axis; `gaussian_nw` is the two-axis spelling of it, and
  `geometry.scalar_field` the isotropic one.

  Returns (values, support), where support is the summed kernel weight -- the
  effective number of samples speaking for each point. Where support is tiny the
  value is a 0/0 fabrication from one distant sample, so callers mask on it
  rather than trust the height.
  """
  X, y, Q = f64(X), f64(y), f64(Q)
  values, support = np.empty(len(Q)), np.empty(len(Q))

  for lo, hi in _blocks(len(Q)):
    w = gaussian_weights(Q[lo:hi], X, h)
    wsum = w.sum(axis=1)
    # Guard the 0/0: where no sample has any weight the value is meaningless
    # anyway and the caller's density mask will drop it.
    values[lo:hi] = weighted_sum(w, y) / np.where(wsum > 0, wsum, 1.0)
    support[lo:hi] = wsum

  return values, support


def gaussian_nw(X, y, Q, hx, hy):
  """Nadaraya-Watson with a Gaussian kernel and one bandwidth per axis.

  The `predict(X, y, Q, **params)` spelling of `nadaraya_watson`, so the tuning
  protocol can sweep hx and hy as named parameters like any other smoother.
  """
  return nadaraya_watson(X, y, Q, (hx, hy))


def epanechnikov_nw(X, y, Q, hx, hy):
  """Nadaraya-Watson with an Epanechnikov kernel (compact support)."""
  X, y, Q = f64(X), f64(y), f64(Q)
  values, support = np.empty(len(Q)), np.empty(len(Q))

  for lo, hi in _blocks(len(Q)):
    _, _, u2 = _scaled_offsets(Q[lo:hi], X, hx, hy)
    w = np.where(u2 <= 1.0, 0.75 * (1.0 - u2), 0.0)
    wsum = w.sum(axis=1)
    values[lo:hi] = weighted_sum(w, y) / np.where(wsum > 0, wsum, 1.0)
    support[lo:hi] = wsum

  # An empty window is not a zero score, it is no information. Mark it so the
  # caller's density floor drops it rather than plotting a hole at z=0.
  values[support <= _EPS] = np.nan
  return values, support


def _wls(dx, dy, w, y, degree):
  """Weighted least squares of a local polynomial, one fit per query point.

  Returns the intercept -- the fit evaluated at the query itself, since the
  design is built from offsets and so the query sits at the origin.
  """
  cols = [np.ones_like(dx), dx, dy]
  if degree == 2:
    cols += [dx * dx, dx * dy, dy * dy]
  A = np.stack(cols, axis=2)                       # (m, n, p)

  Aw = A * w[:, :, None]
  AtA = np.einsum("mnp,mnq->mpq", Aw, A)           # (m, p, p)
  Aty = np.einsum("mnp,mn->mp", Aw, y[None, :] if y.ndim == 1 else y)

  # Ridge the normal equations: with too few samples in the window (or samples
  # that are collinear -- common at the boundary) AtA is singular, and an exact
  # solve would blow up rather than fail. The nudge is far below the signal.
  p = AtA.shape[1]
  AtA = AtA + _EPS * np.eye(p)[None, :, :]

  beta = np.linalg.solve(AtA, Aty[:, :, None])[:, :, 0]
  return beta[:, 0]


def local_linear(X, y, Q, hx, hy, degree=1):
  """Local linear (or quadratic) regression with a Gaussian kernel."""
  X, y, Q = f64(X), f64(y), f64(Q)
  values, support = np.empty(len(Q)), np.empty(len(Q))

  for lo, hi in _blocks(len(Q)):
    dx, dy, u2 = _scaled_offsets(Q[lo:hi], X, hx, hy)
    w = np.exp(-0.5 * u2)
    support[lo:hi] = w.sum(axis=1)
    values[lo:hi] = _wls(dx, dy, w, y, degree)

  return values, support


def loess(X, y, Q, frac=0.3, degree=1, iterations=2):
  """Robust LOESS: local polynomial on a k-nearest-neighbour window.

  `frac` is the neighbourhood size as a fraction of the samples. Tricube weights
  fall to zero at the window edge; `iterations` rounds of bisquare reweighting
  then discount whichever samples the fit cannot explain, so a few wild
  annotations do not drag the surface toward themselves.
  """
  X, y, Q = f64(X), f64(y), f64(Q)
  n = len(X)
  k = max(int(np.ceil(frac * n)), degree + 2)
  k = min(k, n)

  # Robustness weights are a property of the samples, so they are learned by
  # fitting at the samples themselves, then reused when predicting on the grid.
  delta = np.ones(n)
  for _ in range(max(iterations, 0)):
    fitted = _loess_predict(X, y, X, k, degree, delta)
    resid = y - fitted
    s = np.median(np.abs(resid))
    if s <= _EPS:
      break                                   # a perfect fit: nothing to discount
    u = np.clip(resid / (6.0 * s), -1.0, 1.0)
    delta = (1.0 - u ** 2) ** 2

  values = _loess_predict(X, y, Q, k, degree, delta)
  support = _loess_support(X, Q, k)
  return values, support


def _tricube(dist, radius):
  u = np.clip(dist / np.where(radius > 0, radius, 1.0)[:, None], 0.0, 1.0)
  return (1.0 - u ** 3) ** 3


def _knn_radius(dist, k):
  """Distance to the k-th nearest sample, per query."""
  part = np.partition(dist, k - 1, axis=1)[:, k - 1]
  return part


def _loess_predict(X, y, Q, k, degree, delta):
  values = np.empty(len(Q))

  for lo, hi in _blocks(len(Q)):
    q = Q[lo:hi]
    dx = q[:, None, 0] - X[None, :, 0]
    dy = q[:, None, 1] - X[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    radius = _knn_radius(dist, k)
    w = _tricube(dist, radius) * delta[None, :]
    values[lo:hi] = _wls(dx, dy, w, y, degree)

  return values


def _loess_support(X, Q, k):
  support = np.empty(len(Q))
  for lo, hi in _blocks(len(Q)):
    q = Q[lo:hi]
    dist = np.sqrt(((q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    radius = _knn_radius(dist, k)
    support[lo:hi] = _tricube(dist, radius).sum(axis=1)
  return support


# name -> (function, parameter names tuned by cross-validation)
SMOOTHERS = {
  "gaussian_nw": (gaussian_nw, ("hx", "hy")),
  "epanechnikov_nw": (epanechnikov_nw, ("hx", "hy")),
  "local_linear": (local_linear, ("hx", "hy", "degree")),
  "loess": (loess, ("frac", "degree")),
}
