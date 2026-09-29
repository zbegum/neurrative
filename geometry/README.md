# geometry

The geometric algorithms the figures are built on. Nothing here plots; every
module takes arrays and returns arrays.

| module | what |
|---|---|
| `smoothers.py` | Scatterplot smoothers `predict(X, y, Q, **params) -> (values, support)`: Gaussian and Epanechnikov Nadaraya-Watson, local linear, LOESS. `support` is the summed kernel weight behind each prediction, so callers can mask where there is no data. |
| `scalar_field.py` | Per-paragraph scores to a height field `z = f(x, y)` on a grid over the PCA plane, with a leave-one-out bandwidth search. |
| `mesh.py` | A masked height field lifted to a triangle mesh `(x, y, alpha * f)`. `alpha` is the exchange rate between score units and plane units, and so decides how much a geodesic bends. |
| `geodesic.py` | Shortest paths along that mesh, plus length and deviation measures and the "rising pairs" used by the geodesic figures. |
| `_geodesic_native.py` | ctypes bridge to the vendored exact solver below. Builds `geodesic_cpp/wrapper.cpp` into a shared library on first import (needs a C++ compiler, nothing else). |
| `geodesic_cpp/` | Danil Kirsanov's exact geodesic library (Mitchell-Mount-Papadimitriou), vendored verbatim; see `UPSTREAM_README.txt`. `wrapper.cpp` is ours. |
| `curve_smoothing.py` | Distance-based smoothing of a curve on a surface, after Pawellek, Rössl and Lawonn, *Distance-Based Smoothing of Curves on Surface Meshes*, Computer Graphics Forum 43(5), 2024. One tolerance `tau` bounds how far the smoothed arc may leave the original. |

Import from the repository root (`from geometry.smoothers import gaussian_nw`).
Every script puts the root on `sys.path` itself (the short bootstrap block at the
top of each file), so they run from any working directory.
