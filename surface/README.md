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

## What they agree on (bge-m3)

- **The landscape is a broad tilt with one or two rises, not rugged terrain.**
  The four kernel smoothers agree closely per emotion on held-out paragraphs,
  and on Alice nine B-spline control points explain almost as much as the best
  fit.
- **Which emotion has structure depends on the book.** Held-out improvement
  over a flat average (Gaussian, Epanechnikov and local linear; LOESS is lower
  on Pride and Prejudice, see `kernel/`):

  | | most structure | least structure |
  |---|---|---|
  | Alice | wonder 18–19%, curiosity 15–17% | sadness 2–5%, humor 7% |
  | Pride and Prejudice | sadness 19–22%, danger 6% | wonder about 1%, confusion about 2% |

  On Alice the B-spline agrees (about 35% of the variance for wonder, 9% for
  sadness). On Pride and Prejudice most emotions are close to flat over the
  plane: where a paragraph sits in the semantic plane says little about them.
- **Poisson reconstruction is watertight** (Euler characteristic 2) for all six
  emotions in both books. The open sheet stays inside the [0, 1] score range
  except for Pride and Prejudice's humor, 0.6% of the plane.

Figures: [docs/figures](../docs/figures/README.md#5-the-emotion-landscape).
