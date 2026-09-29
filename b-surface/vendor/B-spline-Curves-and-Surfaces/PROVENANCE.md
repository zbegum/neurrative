# Vendored: B-spline-Curves-and-Surfaces

Copied verbatim from

  https://github.com/LorenzoPratesi/B-spline-Curves-and-Surfaces
  commit 1b37161291381e5b22ee5297c0dfe2fddf7a6949

Only the files on the surface-fitting path are vendored:

  bs_least_square_2.m              least-squares tensor-product surface fit
  bspline_basis.m                  Cox-de Boor basis, one function at a time
  build_knot_vector.m              clamped uniform knots
  QR_solve.m                       QR + back substitution for the normal equations
  bspline_deboor.m                 curve evaluation (not on the surface path;
                                   kept because it is what defines the knot and
                                   control-point conventions the fit assumes)
  example/ex_15_bspline_surf_ls.m  the surface example, which is the reference
                                   our port is checked against

The curve-fitting files (`bs_least_square.m`), the report and the notebooks are
not vendored: `visualization/narrative_arc.py` already fits curves through the
`spline_curve_fitting` vendor, and nothing here needs a second curve fitter.

**No license.** The upstream repository carries no LICENSE file, so these files
are here as a reference implementation to check our port against, not as code we
have a grant to redistribute. That is also why `../../bspline_surface.py` is a
port rather than a bridge that shells out: the code we actually run is ours, and
the MATLAB sits next to it as the thing it must agree with.

MATLAB is not installed here, so `check_port.py` reproduces `ex_15`'s setup --
the same `f(x, y) = log(4x^2 + y^2)` on the same domain with the same uniform
noise, degree and knot counts -- and checks the port lands at the same RRMSE the
example reports. That is agreement with the example, not with a MATLAB run.

## Conventions worth knowing before reading the port

- `d` is called "order of the B-Spline base" in the docstrings but is used as the
  *degree*: the basis is evaluated at order `d+1`. The port calls it `degree`.
- `knots_x` is the vector of *breakpoints*, endpoints included. `bs_least_square_2`
  clamps it itself by repeating each end `d` times, so the caller passes a plain
  `linspace`. Coefficient count is therefore `numel(knots_x) + d - 1`.
- Control points come back as one flat `(ncoeff_x*ncoeff_y, 3)` array in
  x-major order: index `(j-1)*ncoeff_y + k`.
- `lambda` is a plain ridge on the normal equations (`B'B + lambda*I`), not a
  roughness penalty -- it does not know that neighbouring coefficients should be
  close, it only keeps the system solvable.
