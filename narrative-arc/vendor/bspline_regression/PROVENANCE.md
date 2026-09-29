# Vendored: bspline-regression

These files are copied from

  https://github.com/rstebbing/bspline-regression  (MIT, see LICENSE)
  commit f9e9aabc68d342dd2c21d84cc90cad519c62e06e

so that `narrative_arc/curves.py` fits the narrative-arc curve with that
project's uniform B-spline least-squares solver. Only the files on the fitting
path are vendored:

  uniform_bspline.py       UniformBSpline: basis, evaluation, derivatives
  fit_uniform_bspline.py   UniformBSplineLeastSquaresOptimiser (damped Newton / LM)
  util.py                  shape checks, previous_float

`generate_example.py`, `visualise.py` and the figures are not vendored.

**One change** from upstream: the three sibling imports are made package-relative
(`from uniform_bspline import ...` -> `from .uniform_bspline import ...`, and
likewise `from util`), so a module called `util` elsewhere on the path cannot
shadow this one. The numerics are untouched. `__init__.py` is ours and empty.
