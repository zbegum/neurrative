# surface/bspline — least-squares B-spline surface

A least-squares tensor-product B-spline surface fitted to
`z = emotion score` over `(x, y) = (PC1, PC2)`, one surface per emotion.

    python surface/bspline/check_port.py                              # validate the port
    python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3
    python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3 \
      --domain unit                                            # cover the whole square
    python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3 \
      --domain unit --control 3 --grid --control-net    # 3x3, all six, one figure
    python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3 \
      --emotions wonder --sweep

Outputs go where every other fit of this family goes,
`surface/bspline/output/<book>/<model>/bspline_ls/<variant>/`, next to `isotropic_loo`
and the four kernel smoothers. It is a different estimator of the same object,
so it is a sibling of theirs rather than a directory of its own -- and it writes
`field_<emotion>_pca.npz` with the same keys, so the geodesic stack can walk on
this terrain without knowing which fit produced it.

`--domain unit` writes to `surface/bspline_ls_unit/` instead, and its files are
named `..._pca_unit` rather than `..._pca`, because its coordinates are not
`pca.npy`'s: see **Covering the whole square** below.

## What is here

    vendor/B-spline-Curves-and-Surfaces/  the MATLAB original, verbatim, with
                                          PROVENANCE.md -- read that first
    bspline_surface.py                    the port: basis, fit, evaluation, mask
    check_port.py                         what holds the port honest
    fit_surface.py                        the driver over a book's emotions

Why a port rather than a bridge (as the old JS B-spline curve fitter was): MATLAB is
not installed, and the upstream repo carries no license. So the code that runs
is ours and the MATLAB sits beside it as the specification. `check_port.py`
covers partition of unity, exact reproduction of a cubic, agreement between the
vendor's hand-written QR back substitution and numpy's `lstsq`, and the vendor's
own surface example rebuilt in numpy.

## How it differs from the kernel surfaces

`surface/kernel/loo.py` and the `smooth_*` family fit the same height
field by kernel smoothing: no parameters, just a reweighting of nearby
paragraphs at every query point. This fits *parameters* -- a grid of control
heights -- and the surface afterwards is that basis expansion, with no further
reference to the paragraphs. The knot count plays the role of the bandwidth, and
runs the other way: more knots is less smoothing.

Two consequences worth knowing before reading a figure from here.

**The knot grid is a rectangle; the paragraphs are a clump.** A basis cell with
no paragraph in it still gets a coefficient, from the ridge rather than from any
text. `fit_surface.py` prints how many, and masks grid cells with no paragraph
within a nearest-neighbour radius so the empty corners of the plane are not
drawn at all. The masked fraction is printed too -- for Alice under bge-m3 the
book occupies about 77% of its own bounding rectangle.

**The fit is unconstrained in z.** Scores live in [0, 1]; a least-squares spline
does not know that, and its relief runs slightly outside on every emotion. The
plots keep `zlim` at (0, 1) rather than hiding this, so an overshoot shows as a
spike leaving the box.

## Covering the whole square

`--domain unit` answers a different question: not "what does the landscape look
like where the book is" but "what does it look like over the whole plane". Three
things have to change for that to be answerable rather than merely drawable.

**The coordinates.** PC1 and PC2 are rescaled into `[0, 1]` by *one shared
scale*, not one per axis -- PCA is metric, so stretching the axes independently
would make an isotropic knot grid anisotropic in the data's own geometry. The
cloud is centred and the shorter axis simply does not reach the edges.
`--margin` (default 0.08) is how much square is left outside the paragraphs on
the long axis. The `{scale, offset}` is written into every `.npz` and into
`params.json`, so anything here can be put back into `pca.npy` units.

**The penalty.** The vendor's `lam * I` ridge pulls every coefficient toward
zero, which is fine for keeping the matrix invertible and useless for filling an
empty cell -- the surface would fall to a score of 0 out there, abruptly,
because nothing ties that cell to its neighbours. It is replaced by a P-spline
difference penalty (Eilers & Marx): where there is data the penalty is a mild
smoother and the data decides; where there is none it is the only thing acting,
so a coefficient becomes whatever continues its neighbours. `--penalty-order 1`
(the default) continues *flat*, so the surface levels off at the nearby scores.
`--penalty-order 2` continues the edge *slope*, which over a domain this much
bigger than the cloud walks straight out of `[0, 1]` -- the same reason
`surface/poisson/poisson.py` refuses a local linear extension. Available, not the
default.

**The rejection rule.** Cross-validation cannot see any of this: it scores
held-out *paragraphs*, all of which are inside the cloud, so it is silent about
the corners. A candidate is therefore fitted on all the data first and rejected
if the surface leaves `[-0.5, 1.5]` anywhere on the square (`--envelope`);
`(knots, lam)` is then chosen by CV among the survivors, and `params.json`
records the whole search including what was thrown out. For Alice that rejects
16-19 of the 45 candidates per emotion.

The figure draws the supported part solid and the extension at low alpha, so the
two are never confused. The support mask is saved either way; under `--domain
support` it also decides what is drawn, under `--domain unit` only how.

## Asking for a control-point count

`--control N` is the same knob as `--knots` said the other way round: a
tensor-product B-spline has `knots + degree - 1` coefficients per axis, and
"control points" is what those coefficients are called when you are looking at
the net rather than the basis. Asking for a specific net therefore also
constrains the degree -- 3 control points on an axis is a *single quadratic
span*, and a cubic cannot be carried by fewer than 4 -- so an unset `--degree`
is lowered to fit rather than failing on arithmetic the caller had no reason to
have done. `--control 3` gives degree 2, 2 breakpoints, 9 coefficients for the
whole plane.

That fixes the knot count, so the CV has only `lam` left to choose. It also
makes the net small enough to draw: `--control-net` puts the control points and
the net joining them over the surface. They sit at the Greville abscissae, and
they are *not* on the surface -- a B-spline approaches its net without
interpolating it, which is visible in the figure and is the point of drawing it.

`--grid` puts every emotion in one figure on a shared z range. The shared range
is what makes it a comparison: six autoscaled panels would make a flat emotion
look as mountainous as a varied one.

## What it says about Alice

Under `--domain support`, cross-validated knot counts land at the very bottom of
the ladder -- 2 or 3 breakpoints per axis, i.e. 16 to 25 coefficients for 789
paragraphs -- and the `--sweep` panels show why: by 6 knots the surface is
already throwing spikes out of the score range where a cell is thinly supported,
and by 12 it is mostly spikes. That is the ridge failing, not the data: it has
nothing to say about a coefficient no paragraph constrains, so the fit is free
to do anything there, and does.

Under `--domain unit` the difference penalty removes that failure mode, and the
CV then buys detail it could not afford before -- 4 to 10 knots per axis
depending on the emotion, at the top of the `lam` ladder. Wonder, for instance,
goes from 3 knots explaining 33.2% of the variance to 6 knots explaining 34.9%,
and its relief stays inside `[0.11, 0.68]` across the entire square instead of
overshooting `[0, 1]` at the edges. The price is that a penalty strong enough to
extend safely also flattens what it is extending from: these surfaces are
smoother than their `bspline_ls` counterparts, and the smoothness is a choice
made by the envelope rule, not a finding.

**Nine control points are almost the whole story.** At `--control 3` -- one
biquadratic patch, 9 coefficients for the entire plane -- variance explained
runs 10.1% (sadness) to 33.7% (wonder), against 9% to 34.9% for the fits that
were allowed up to 10 knots per axis. Nearly everything the CV-chosen surfaces
found is a tilt with one bend in it. Two smaller observations from that run: the
CV score barely moves across the whole `lam` ladder (fourth decimal place),
because with 9 coefficients and every cell full of paragraphs the penalty has
almost nothing left to do; and the tightest net is not the safest one -- humor
at `lam` 0.01 still reached 1.20 in a corner and was rejected, since a quadratic
that has to bend for the middle of the cloud keeps bending past the edge of it.

Either way the shape of the answer is the same. The emotion landscape that
survives cross-validation is a broad tilt with one or two rises on it, not
terrain, and about 40% of the unit square is extension rather than fit. Variance
explained under 5-fold CV runs from 9% (sadness) to 33-35% (wonder); whether
that ordering matches the kernel fits' has not been checked here.
