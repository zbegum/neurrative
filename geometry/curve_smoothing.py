"""
Distance-based smoothing of a curve, after

  M. Pawellek, C. Rossl, K. Lawonn, "Distance-Based Smoothing of Curves on
  Surface Meshes", Computer Graphics Forum 43(5), 2024 (EGSGP 2024).

The paper's algorithm, in four steps:

  1. build a penalty potential  phi : S -> R  that grows with distance from the
     initial curve gamma (so leaving gamma costs height);
  2. lift the surface and the curve into 4D by appending phi as a coordinate,
     chi_phi(x) = (x, phi(x));
  3. find a geodesic in the lifted surface, in gamma's isotopy class -- straight
     (locally shortest) there, hence smooth, yet held near gamma by phi's walls;
  4. project back by dropping the phi coordinate.

The single parameter tau trades smoothness against closeness: tau = 0 gives the
plain geodesic of the base surface (smoothest); large tau pins the result to the
original curve.

--- our specialisation -------------------------------------------------------

Our "surface" is a height field over the 2-D PCA (semantic) plane, and the curve
we smooth -- the narrative arc -- is a polyline in that plane. For a *flat* base
plane the 4-D lift (x, y, 0, phi) is isometric to the 3-D height field
(x, y, phi): the paper's own 2-D illustration (their Figure 2) is exactly this
case. So we implement the method with no 4-D machinery and no mesh solver:

  * phi is built on a regular grid over the plane (steps 1);
  * the lifted geodesic (steps 2-3) is the minimiser of the paper's energy
    E(alpha) = 1/2 integral |alpha'(t)|^2 dt   (their Eq. 1)
    over polylines constrained to the lifted surface z = phi(x, y), with the
    endpoints fixed. Descending this energy from gamma keeps the result in
    gamma's isotopy class (we deform, never reconnect), which is what the paper
    obtains from FlipOut. The Euler-Lagrange form is Laplacian smoothing of the
    plane coordinates coupled to phi's gradient -- Laplacian smoothing held back
    by a distance penalty, matching the paper's own reading of the method.

The emotion terrain is not used here; it is for *drawing* the smoothed arc (see
arc_on_surface/arc_smooth.py). Smoothing on the true curved terrain would need a
custom-metric geodesic (FlipOut) we do not vendor; the penalty phi, not the
terrain's curvature, is what does the smoothing.
"""

import numpy as np


def _resample(curve, spacing):
  """Resample a polyline to roughly uniform arc-length spacing."""
  curve = np.asarray(curve, dtype=np.float64)
  seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
  s = np.concatenate([[0.0], np.cumsum(seg)])
  total = float(s[-1])
  if total <= 0:
    return curve.copy()
  n = max(3, int(round(total / spacing)) + 1)
  su = np.linspace(0.0, total, n)
  x = np.interp(su, s, curve[:, 0])
  y = np.interp(su, s, curve[:, 1])
  return np.column_stack([x, y])


def distance_field(gx, gy, curve, samples_per_unit=None):
  """Euclidean distance from every grid point to the polyline `curve`.

  On a flat domain the geodesic distance to the curve is just this Euclidean
  distance; densely sampling the curve and taking the nearest sample is exact to
  the sampling spacing (the paper likewise samples the curve onto the mesh).
  Returns an array shaped like gx.
  """
  from scipy.spatial import cKDTree
  span = float(max(gx.max() - gx.min(), gy.max() - gy.min()))
  spacing = span / (samples_per_unit or 400.0)
  dense = _resample(curve, max(spacing, 1e-6))
  tree = cKDTree(dense)
  q = np.column_stack([gx.ravel(), gy.ravel()])
  d, _ = tree.query(q)
  return d.reshape(gx.shape)


def penalty_potential(delta, tau, a=10.0):
  """The paper's modifier m_tau applied to a distance field (their Eq. 5-6).

    f(x) = x^2,  g(x) = L f(x/L) = x^2 / L,  m_tau(x) = g(a tau x) = (a tau)^2 x^2 / L
    L = ||delta||_inf     a = 10 (their default)

  tau = 0 flattens phi to zero (-> plain geodesic); larger tau steepens the
  walls that hold the curve near its original.
  """
  L = float(np.nanmax(delta)) or 1.0
  return (a * tau) ** 2 * delta ** 2 / L


class _Field:
  """Bilinear sampler for phi and its gradient on the regular grid."""

  def __init__(self, xs, ys, Z):
    self.xs, self.ys, self.Z = xs, ys, Z
    self.dx = xs[1] - xs[0]
    self.dy = ys[1] - ys[0]

  def _cell(self, x, y):
    xs, ys = self.xs, self.ys
    fx = np.clip((x - xs[0]) / self.dx, 0, len(xs) - 1.001)
    fy = np.clip((y - ys[0]) / self.dy, 0, len(ys) - 1.001)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    Z = self.Z
    return (Z[iy, ix], Z[iy, ix + 1], Z[iy + 1, ix], Z[iy + 1, ix + 1], tx, ty)

  def phi(self, x, y):
    a00, a10, a01, a11, tx, ty = self._cell(x, y)
    return ((a00 * (1 - tx) + a10 * tx) * (1 - ty) +
            (a01 * (1 - tx) + a11 * tx) * ty)

  def grad(self, x, y):
    # Exact gradient of the SAME bilinear interpolant phi returns, so the energy
    # and its gradient stay consistent for the optimiser.
    a00, a10, a01, a11, tx, ty = self._cell(x, y)
    dphidx = ((a10 - a00) * (1 - ty) + (a11 - a01) * ty) / self.dx
    dphidy = ((a01 - a00) * (1 - tx) + (a11 - a10) * tx) / self.dy
    return dphidx, dphidy


def smooth_curve(curve, tau, *, resolution=260, margin=0.25, blur=1.2,
                 a=10.0, iters=800, step=0.5, tol=1e-9):
  """Distance-based smoothing of a planar curve (Pawellek et al. 2024).

  Parameters
    curve       (n, 2) polyline in the plane -- the initial (noisy) curve gamma.
    tau         smoothness/closeness trade-off. 0 = straight geodesic, large =
                hugs the original curve.
    resolution  grid resolution for the penalty potential.
    margin      grid padding around the curve's bounding box, as a fraction of
                its extent (the curve must be able to move outward).
    blur        Gaussian smoothing (in grid cells) of the distance field, the
                stand-in for the paper's enlarged heat-method time step: it makes
                phi's gradient well behaved across the medial axis.
    a           the modifier constant (paper default 10).
    iters/step/tol  gradient-descent budget for the constrained energy minimiser.

  Returns a dict:
    curve       (m, 2) the smoothed polyline (endpoints preserved).
    phi         (m,)  penalty height along the smoothed curve.
    delta       (m,)  distance of each smoothed point from the original curve.
    length, length0   arc length of the smoothed and original curves (in-plane).
    max_dev     largest distance any smoothed point sits from the original curve.
    grid        (xs, ys, Phi) the penalty potential, for plotting.
  """
  curve = np.asarray(curve, dtype=np.float64)

  # --- grid over the plane, padded so the curve can move outward ---------------
  lo = curve.min(axis=0)
  hi = curve.max(axis=0)
  pad = (hi - lo) * margin + 1e-9
  lo, hi = lo - pad, hi + pad
  xs = np.linspace(lo[0], hi[0], resolution)
  ys = np.linspace(lo[1], hi[1], resolution)
  gx, gy = np.meshgrid(xs, ys)

  # --- penalty potential phi = m_tau(distance-to-curve) ------------------------
  delta = distance_field(gx, gy, curve)
  if blur > 0:
    from scipy.ndimage import gaussian_filter
    delta = gaussian_filter(delta, blur)
  Phi = penalty_potential(delta, tau, a=a)
  field = _Field(xs, ys, Phi)

  # --- initial lifted polyline (densified, endpoints fixed) --------------------
  span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
  P = _resample(curve, span / 220.0)
  length0 = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())

  # --- minimise the lifted Dirichlet energy E = 1/2 sum |C_{i+1}-C_i|^2 --------
  # C_i = (x_i, y_i, phi(x_i, y_i)), endpoints fixed. This is a smooth function of
  # the interior (x_i, y_i); its minimiser is a discrete geodesic of the lifted
  # surface in gamma's isotopy class (we descend from gamma, never reconnect).
  # We use L-BFGS with the analytic gradient
  #   dE/dx_i = -( lap(x)_i + lap(phi)_i * dphi/dx_i ),   (lap = x_{i-1}-2x_i+x_{i+1})
  # which handles both the stiff steep-wall regime (large tau) and the slow
  # low-frequency relaxation (small tau) that defeat explicit gradient flow.
  from scipy.optimize import minimize

  x0, y0 = P[:, 0].copy(), P[:, 1].copy()
  M = len(x0) - 2                                  # interior node count
  if M <= 0:
    out = np.column_stack([x0, y0])
  else:
    xa, xb, ya, yb = x0[0], x0[-1], y0[0], y0[-1]  # fixed endpoints

    def unpack(v):
      x = np.empty(M + 2); y = np.empty(M + 2)
      x[0], x[-1], y[0], y[-1] = xa, xb, ya, yb
      x[1:-1] = v[:M]; y[1:-1] = v[M:]
      return x, y

    def fun(v):
      x, y = unpack(v)
      z = field.phi(x, y)
      C = np.column_stack([x, y, z])
      d = np.diff(C, axis=0)
      E = 0.5 * float((d ** 2).sum())
      lap = C[:-2] - 2 * C[1:-1] + C[2:]           # interior discrete Laplacian
      dphidx, dphidy = field.grad(x[1:-1], y[1:-1])
      gx = -(lap[:, 0] + lap[:, 2] * dphidx)       # dE/dx_i
      gy = -(lap[:, 1] + lap[:, 2] * dphidy)       # dE/dy_i
      return E, np.concatenate([gx, gy])

    v0 = np.concatenate([x0[1:-1], y0[1:-1]])
    res = minimize(fun, v0, jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "gtol": tol, "ftol": tol})
    x, y = unpack(res.x)
    out = np.column_stack([x, y])
  # closeness diagnostics against the ORIGINAL curve
  from scipy.spatial import cKDTree
  dev = cKDTree(_resample(curve, span / 600.0)).query(out)[0]
  return {
    "curve": out,
    "phi": field.phi(x, y),
    "delta": dev,
    "length": float(np.linalg.norm(np.diff(out, axis=0), axis=1).sum()),
    "length0": length0,
    "max_dev": float(dev.max()),
    "grid": (xs, ys, Phi),
  }
