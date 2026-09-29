# surface — step 2: the emotion landscape

A height field `z = emotion(PC1, PC2)`: for each of the six emotions, a surface
over the PCA plane whose height is that emotion's score. PCA only, because UMAP
and t-SNE are non-metric: two paragraphs can land on one spot with different
scores, so a height there is not well defined. Run `projection/embedding.py`
first; every method reads its `pca.npy`.

Each method has its own folder, with its own README and its own `output/`:

| folder | method | idea | knob |
|---|---|---|---|
| [`points/`](points/) | none | the raw paragraph cloud, nothing fitted: what every fit is judged against | — |
| [`kernel/`](kernel/) | kernel smoothing | a local weighted average of nearby paragraphs at every point (Gaussian, Epanechnikov, local linear, LOESS) | bandwidth |
| [`bspline/`](bspline/) | least-squares B-spline | a global smooth surface with a small grid of control heights | number of knots |
| [`poisson/`](poisson/) | screened Poisson reconstruction | the landscape as the lid of a watertight solid; `--open` cuts it back to a sheet | octree depth |
| [`mood/`](mood/) | one mood surface | the six emotions collapsed to one sadness → humor value first, then one surface | blend, bandwidth |

Kernel, B-spline and Poisson all write the same `field_<emotion>_pca.npz`, so
`arc_on_surface/` can walk on any of them.

## What they agree on (Alice, bge-m3)

- **The landscape is a broad tilt with one or two rises, not rugged terrain.**
  The four kernel smoothers agree to within about one percentage point per
  emotion on held-out paragraphs, and nine B-spline control points explain
  almost as much as the best fit.
- **Wonder has the most structure, sadness the least**, in every method: the
  kernel fits improve on a flat average by about 18.5% for wonder and 2–5% for
  sadness, and the B-spline explains about 35% and 9% of the variance.
- **Poisson reconstruction is watertight** (Euler characteristic 2) for all six
  emotions, and its open sheet stays inside the [0, 1] score range.
