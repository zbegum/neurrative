# surface/poisson — screened Poisson reconstruction

A surface fitted to `(x, y) = (PC1, PC2)`, `z = emotion score`, through the
vendored [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) -- one per
emotion, in either of two shapes.

    surface/poisson/build_vendor.sh                              # once
    python surface/poisson/check_recon.py                        # validate it

    # the closed solid under the landscape, over the book's own outline
    python surface/poisson/fit_surface.py --book alice_wonderland --model bge-m3

    # the landscape alone, open, spanning the whole PCA rectangle
    python surface/poisson/fit_surface.py --book alice_wonderland --model bge-m3 \
      --open

`--open` is the one to reach for if what you want is a *surface* in the sense
the rest of this repo means it. It fits the same solid -- the closure is what
makes the solve well posed -- then keeps only the top of it, so what lands on
disk is an open sheet with a boundary, covering all of (PC1, PC2), plus the same
`field_<emotion>_pca.npz` the kernel fits write. Everything below is about the
closed object it is cut from.

Output goes to `surface/poisson/output/<book>/<model>/poisson_closed/<variant>/`, or
`poisson_open/` with `--open`, beside `bspline_ls` and the kernel smoothers,
because it fits the same cloud.

## What is here

    vendor/PoissonRecon/   the upstream repo, verbatim, MIT -- see PROVENANCE.md
    build_vendor.sh        how it builds on macOS, and why not with `make`
    poisson.py             normals, the closure, PLY I/O, the vendor call
    check_recon.py         what holds the vendor call honest
    fit_surface.py         the driver over a book's emotions

Unlike `surface/bspline/vendor/`, this is code that *runs*. It is MIT licensed and it
is C++ that builds here, so there was nothing to port.

## It is not a height field, and that is the point

Every other fit of this cloud -- the four kernel smoothers and `surface/bspline` --
returns `z = f(x, y)`: one height per point of the PCA plane, an open sheet.
Poisson reconstruction cannot return that. It takes *oriented* points and
returns the boundary of the solid those normals bound: closed, two-sided,
watertight. So what this folder fits is

    {(x, y, z) : (x, y) in the book's support, -base <= z <= f(x, y)}

and what it writes is that solid's skin. `wonder` under Alice comes out as a
30-thousand-triangle object you could 3-D print: a plinth, ragged vertical walls
around the outline of the book's own footprint in the PCA plane, and the emotion
terrain as its lid.

The consequence is that the paragraphs alone are not a valid input -- they
sample the lid and nothing else, and a solve given only a lid closes the solid
wherever the boundary condition sends it. So `poisson.py` builds the rest of the
boundary explicitly, and `fit_surface.py` prints the three parts separately
because only one of them is the book:

    lid     789 points, one per paragraph, normals from a local plane fit
    floor   ~4800 points at z = -base, normals down
    wall    3000-10000 points around the rim, normals out

Roughly 5-7% of the input is text. The rest is scaffolding, and it is not
neutral: it decides the footprint (whatever the support mask says), the base
(whatever `--base` says) and nothing else. The lid is where the scores are.

## What `--open` does differently

Two changes to the cloud, then one cut.

**The footprint becomes the whole rectangle**, so the surface can cover the
plane rather than the book's ragged outline. The lid needs heights out there,
where there are no paragraphs, so `lid_fill` extends the fit with a
nearest-neighbour-widened kernel estimate -- which *flattens* away from the data
rather than extrapolating a slope off the end of it. Nothing out there is a
claim about the book. For Alice it is 25% of the rectangle, it is reported as a
separate count, and the figure draws the support boundary as an orange line on
the surface so a reader can see which side of it a feature is on.

**The extension is held back** from the support by `--fill-buffer` support
radii, because a smooth kernel estimate sampled hard against noisy paragraph
heights is a contradiction the solver resolves by tearing.

**Then the lid is cut off the solid.** `top_sheet` keeps the triangles whose
normal points up -- that alone is the honest object, and it is *not* a height
field: it folds, and once its overhanging faces are removed it has holes where
they were. So what gets written as `Z` is the solid's **upper envelope**, every
triangle rasterized onto the grid with the maximum kept. One height per column,
hole-free by construction, equal to the sheet wherever the sheet does not fold.
`folded_fraction` reports how much of the plane that qualification covers.

The mesh written to `field_<emotion>_pca.npz` is open: it has a boundary at the
rim of the rectangle, and `is_watertight` on it is correctly False.

## Knobs

`--depth` is the smoothing knob and it runs like `surface/bspline`'s knot count rather
than a bandwidth: deeper is *less* smoothing. 8 puts 256 cells across the box.

`--relief` is the height of a score of 1.0 in units of the PCA plane's width.
It is not a display setting. The octree is isotropic, so a flatter landscape
gets fewer cells of vertical resolution and is smoothed more in z than in x.

`--base` is the plinth under `z = 0`. Without it a score of 0 gives the solid
zero thickness, lid and floor land in one octree cell, and the reconstruction
pinches through -- a "closed surface" with a tunnel in it, which is the boundary
of nothing. Over half of Alice's sadness scores are below 0.1, so this is the
common case. At `--base 0` five of six emotions still come out watertight and
sadness does not; at the default 0.08 all six do.

`--point-weight` is the screening weight from Kazhdan-Hoppe 2013. 0 is the
unscreened 2006 solve, which smooths away from the samples; higher holds the
surface to them.

`--lid` and `--bandwidth` are what stop the landscape from being combed, and
the story is worth the paragraph. Screened Poisson *interpolates* its samples,
and neighbouring paragraphs disagree about an emotion by as much as 0.8. Fed the
raw scores (`--lid scores`), the reconstruction does exactly what it is asked:
it plunges a narrow canyon between every disagreeing pair, and a run of those
along a ridge looks like the teeth of a comb. The default `--lid fitted` puts
the samples at the local plane fit's own value instead, which makes `--bandwidth`
the smoothing knob, as it is in every sibling fit in this repo. `--lid scores` is
kept because seeing the comb is the argument for smoothing.

`--merge-radius` collapses paragraphs that land on the same spot. PCA is a
projection and it is not injective: 57 of Alice's paragraph pairs land within
0.004 of each other in the unit box, some at exactly the same point, carrying
different scores. Two heights over one (x, y) is not a height field, and the
solver resolves the contradiction by tunnelling through the solid -- which is
where the Euler characteristic used to lose its 2. 16 paragraphs, 2% of the book,
merge at the default.

`--open` also brings `--fill-buffer` and `--rim-margin`; both are documented
where they are used, and both exist for the same reason -- a seam between two
things that disagree tears, and a gap lets the solver interpolate instead.

## Reading the numbers it prints

**Watertight, and the Euler characteristic.** Every edge shared by exactly two
triangles, and `V - E + F = 2`, is what "this is the boundary of a solid" cashes
out to. Anything else is a defect, so both are printed per emotion and stamped
into `params.json` rather than being assumed. At the defaults all six of Alice's
emotions come out watertight with chi = 2, closed and free of handles, in both
modes. Getting there took the plinth, the merge and the fitted lid; each of
those three was found by a chi that was not 2.

**Dropped components.** A book is one solid, so a second component is spurious
-- a bubble a few cells across thrown off where the wall sampling is dense. The
largest component is kept and the rest reported: for Alice they run 0.0% to
0.1% of the triangles.

**Paragraph-to-nearest-vertex distance.** How far the surface passes from the
lid it was fitted to, in unit-box units, against a cell size of 1/256. For Alice
the median runs 0.0064 (danger) to 0.0074 (humor), under two cells, so the
surface is faithful to the lid. Note what that is *not* measuring: under the
default `--lid fitted` the lid is already the smoothed landscape, so the distance
from the surface to the raw scores is the bandwidth's business, not this number's.
Read it as a residual with a floor, either way -- the nearest *vertex* is up to
half an edge further than the nearest point of the surface.

**With `--open`, three more.** `overhanging cells` is the fraction of the plane
the top sheet passes over twice, i.e. how far from being a height field it is:
0.00% for five of Alice's emotions and 0.03% for sadness. `uncovered` is the
fraction of grid cells no triangle rasterizes onto, filled from a neighbour --
0.02%. And the field's range in score units, with how much of it escapes [0, 1]:
nothing does, at the defaults, on any emotion.

## The two workarounds, so they are not mistaken for magic

**The build.** The vendor's `make poissonrecon` does not reach a binary on
macOS: its default compiler branch passes `-fopenmp`, which Apple clang has no
runtime for, and its link rule first builds a bundled 1990s `libpng` whose
`pngconf.h` includes the classic MacOS `<fp.h>`. Nothing links that build -- the
binary takes `-lpng` from the system -- so `build_vendor.sh` compiles the object
through the vendor's own no-OpenMP branch and links it directly. The Makefile is
not patched, so `git diff` inside `vendor/` stays empty.

**The retry.** The vendor's iso-surface extractor sometimes aborts with `Failed
to close loop` -- the level set passing exactly through an octree corner, where
the marching-cubes case is ambiguous. A cloud with thousands of samples on
exactly coincident planes, which floors and walls sampled from a regular grid
are, invites it. `reconstruct()` retries with the cloud jittered by 2% of a cell.
That is a real change to the input, so it is printed when it happens. At the
defaults none of Alice's six emotions needs it; at `--base 0` three of them do,
which is a second reason to keep the plinth -- a solid that is thin somewhere is
what walks the level set into the degenerate cell in the first place.
