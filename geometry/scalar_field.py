"""
Interpolate per-paragraph scores into a scalar field over the latent plane.

The paragraphs give us scattered samples (x_i, y_i): a position in 2-D PCA space
and a score. Kernel smoothing turns those into a function f(x) defined
everywhere, whose graph z = f(x) is the "landscape" the geodesics later walk on.

We regress rather than interpolate. The scores are noisy annotations quantized
to a handful of distinct values, and near-identical positions routinely carry
different scores -- so no surface can pass through every sample, and one that
tried would be spikes of annotation noise. Nadaraya-Watson averages instead:

    f(x) = sum_i K_h(x - x_i) y_i / sum_i K_h(x - x_i)

with a Gaussian kernel. See https://en.wikipedia.org/wiki/Kernel_smoother

This module is the *isotropic* face of that estimator: one bandwidth h for every
axis, chosen by closed-form leave-one-out CV, over a grid built here. The
estimator itself lives in `geometry.smoothers` and is shared with the anisotropic
hx/hy path -- there is one Gaussian Nadaraya-Watson in this repo, not two. What
differs between the two entry points is only how the bandwidth is chosen and in
which units it is expressed, which is a genuine difference and the reason both
exist.
"""

import numpy as np

from geometry.smoothers import f64 as _f64
from geometry.smoothers import gaussian_weights, nadaraya_watson, weighted_sum


def make_grid(X, resolution=200, margin=0.05):
  """Axis-aligned grid over the samples' bounding box, padded by `margin`."""
  X = _f64(X)
  lo, hi = X.min(axis=0), X.max(axis=0)
  pad = (hi - lo) * margin
  lo, hi = lo - pad, hi + pad
  xs = np.linspace(lo[0], hi[0], resolution)
  ys = np.linspace(lo[1], hi[1], resolution)
  gx, gy = np.meshgrid(xs, ys)
  return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def loo_bandwidth(X, y, candidates):
  """Pick h by leave-one-out cross-validation. Returns (best_h, mse per h).

  Leave-one-out is closed-form here: dropping sample i just means removing its
  own weight from its own estimate, so no refitting is needed.

  Every h is scored on every point. A narrow kernel leaves some points with no
  neighbour at all -- there the honest prediction is the global mean, and it is
  charged for that. Skipping those points instead would quietly score narrow
  bandwidths on a smaller, easier subset and hand them an unearned win.
  """
  X, y = _f64(X), _f64(y)
  fallback = y.mean()
  mses = []

  for h in candidates:
    w = gaussian_weights(X, X, h)
    num = weighted_sum(w, y) - np.diag(w) * y     # drop own contribution
    den = w.sum(axis=1) - np.diag(w)

    isolated = den <= 1e-12
    pred = np.where(isolated, fallback, num / np.where(isolated, 1.0, den))
    mses.append(np.mean((y - pred) ** 2))

  mses = np.array(mses)
  return candidates[int(np.argmin(mses))], mses


def bandwidth_candidates(X, n=12):
  """A geometric ladder of plausible bandwidths for this point cloud.

  Anchored to the data's own scale: from a fraction of the typical
  nearest-neighbour gap up to a fraction of the cloud's extent.
  """
  X = _f64(X)
  d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
  np.fill_diagonal(d2, np.inf)
  nn = np.sqrt(d2.min(axis=1))
  lo = max(np.median(nn) * 0.5, 1e-6)
  hi = np.linalg.norm(X.max(axis=0) - X.min(axis=0)) * 0.5
  return np.geomspace(lo, hi, n)


def field(X, y, h, resolution=200, margin=0.05, density_floor_pct=5.0):
  """Smooth y over the plane. Returns (gx, gy, Z, mask, floor).

  Z is masked (NaN) wherever the kernel density falls below the given percentile
  of the density seen at the samples themselves -- i.e. wherever the surface
  would be extrapolating into empty latent space.
  """
  gx, gy, Q = make_grid(X, resolution, margin)
  values, density = nadaraya_watson(X, y, Q, h)

  _, sample_density = nadaraya_watson(X, y, X, h)
  floor = np.percentile(sample_density, density_floor_pct)

  Z = np.where(density >= floor, values, np.nan).reshape(gx.shape)
  mask = (density >= floor).reshape(gx.shape)
  return gx, gy, Z, mask, floor
