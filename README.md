# Narrative Tension Field: A Geometric View of Story Structure

**[Project document](https://docs.google.com/document/d/14Wmcna-hrG1aNa0YlczN_qigWfSBBZtYJDHj2jzXlR0/edit?usp=sharing)**
|
**[SGI website](https://sgi.mit.edu/)**
|
**[Justin's geometry course](https://groups.csail.mit.edu/gdpgroup/68410_spring_2023.html)**
|
**[Upstream repository](https://github.com/FlyingGiraffe/neurrative)**

A book, read as geometry. Every paragraph is a point in an embedding space and
carries six emotion scores (wonder, danger, sadness, humor, confusion,
curiosity). From there the project builds three kinds of object and asks what
their shape says about the story:

- **The semantic plane.** The paragraphs projected to 2-D (PCA, UMAP, t-SNE),
  with a check on which projection can be trusted and how reading order can be
  put into the layout.
- **The emotion landscape.** A surface `z = emotion(x, y)` fitted over that
  plane, four different ways, with geodesics that walk on it.
- **The narrative arc.** The book as an ordered curve: windows of paragraphs,
  mean-pooled, joined in reading order, then fitted, validated, lifted onto the
  landscape and thickened into a tube.

The goal is interpretability and visualization, not a predictive model.

## Repository layout

```
books/            the data: three books, three embedding models, emotion scores
preprocess/       raw text -> paragraphs -> embeddings -> LLM emotion scores
geometry/         smoothers, height fields, meshes, exact geodesics, curve smoothing
visualization/    projections, emotion surfaces, the arc on the surface, geodesics
b-surface/        global fit #1: least-squares tensor-product B-spline surface
closed-surface/   global fit #2: screened Poisson reconstruction (watertight solid)
mood_surface/     six emotions -> one mood height; the arc as geodesics on it
narrative-arc/    the arc itself: 2-D/3-D paths, B-spline fit, tube, explorers
site/             a pipeline reference page generated from the code
docs/figures/     selected projection, sweep and arc (B-spline) figures
```

Each of `geometry/`, `b-surface/`, `closed-surface/`, `mood_surface/` and
`narrative-arc/` has its own README with the method and its findings.

## Setup

```bash
pip install -r requirements.txt               # everything except preprocess/
pip install -r requirements-preprocess.txt    # only to add a new book
```

Two parts build native code on first use, and need only a C++ compiler:
`geometry/` compiles the exact geodesic solver automatically on import, and
`closed-surface/build_vendor.sh` builds PoissonRecon once.

## Data

| book | paragraphs | embeddings | emotion scores |
|---|---|---|---|
| `alice_wonderland` | 789 | bge-m3, e5-large-v2, qwen3-embedding | yes |
| `pride_and_prejudice` | 2084 | bge-m3, e5-large-v2, qwen3-embedding | yes |
| `hamlet` | 892 (acts and speakers) | bge-m3, e5-large-v2, qwen3-embedding | no |

```
books/<book>/
  raw.txt                   Project Gutenberg text
  processed.json            chapters (or acts) and paragraphs
  paragraph_scores.json     six emotion scores per paragraph
  embeddings/<model>/embeddings.npy   (n_paragraphs, dim); row i = paragraph i
```

Paragraphs that were only markup (for example `* * * * *` scene breaks) have
been removed, and every embedding and score file is row-aligned with
`processed.json`. The same files are also on
**[Google Drive](https://drive.google.com/drive/folders/1Ut6MJmPx0LbXRXq15MGAbfBs9oUBT0Zj?usp=sharing)**.

### Adding a book

```
python preprocess/preprocess.py --book <book>        # a novel: chapters, paragraphs
python preprocess/preprocess_play.py --book <book>   # a play: acts, speakers
python preprocess/get_embeddings.py --book <book> --model <model>
python preprocess/get_annotations.py --book <book>   # emotion scores (Qwen2.5-7B-Instruct)
```

`paragraph_scores.json` records no model metadata, so which annotator produced a
score file cannot be recovered from the file itself; `--model` is overridable
and only the default is Qwen2.5-7B-Instruct.

## The semantic plane (`visualization/`)

Every script takes `--book` and `--model` and writes under
`output/<book>/<model>/<figure>/`.

- `embedding.py`: 2-D projections (PCA / UMAP / t-SNE) colored by chapter,
  reading order or any emotion. `embedding_grid.py` sweeps UMAP and t-SNE
  parameters (PCA has none, which is the point).
- `emotion_3d.py`: emotion as the third axis, or as color over a 3-component
  projection.
- `projection_comparison.py`: **which projection?** PCA, UMAP and t-SNE side
  by side across neighbourhood scale, scored by trustworthiness and by chapter
  purity over a shuffled-label control.
- `chain_umap.py`: **reading order as graph structure.** UMAP embeds a fuzzy
  graph, so reading order can be added to that graph before the layout runs:
  `G_semantic` t-conorm `beta * G_chain`. `beta = 0` is plain UMAP; the output
  is the trade-off curve, not one layout.
- `inspect_points.py`: trace a point on a plot back to its paragraph.

The projection trials, with their numbers, are collected in
**[docs/figures](docs/figures/README.md)**: PCA keeps the least local
structure in every book, and adding reading order to UMAP trades
trustworthiness for order along a smooth curve.

## The emotion landscape

A height field `z = emotion(PC1, PC2)` over the PCA plane. PCA only, because
UMAP and t-SNE are non-metric: two paragraphs can land on one spot with
different scores, so a height there is not well defined. Every fit writes the
same `field_<emotion>_pca.npz`, so the geodesic scripts can walk on any of them.

| fit | where | what it is |
|---|---|---|
| kernel smoothers | `visualization/smooth_*.py`, `emotion_surface.py` | one estimator (`geometry/smoothers.py`) under different kernels and bandwidth rules, tuned by k-fold CV with a held-out 20% |
| B-spline | `b-surface/` | global least-squares tensor-product spline with a P-spline penalty so it can extend over the whole square |
| Poisson | `closed-surface/` | the landscape as the lid of a watertight solid (chi = 2), or cut open as a height field |
| mood | `mood_surface/` | the six emotions collapsed to one sadness -> humor height |

```
output/<book>/<model>/surface/
  raw/               raw_points.py          the point cloud, nothing fitted
  gaussian_nw/       smooth_gaussian_nw.py  ┐ hx, hy in standardized units,
  epanechnikov_nw/   smooth_epanechnikov…   │ tuned by k-fold CV inside a
  local_linear/      smooth_local_linear…   │ training split, scored once on
  loess/             smooth_loess.py        ┘ a held-out 20%
  isotropic_loo/     emotion_surface.py     one h in raw PCA units, LOO CV
  bandwidth_sweep/   bandwidth_grid.py, smooth_grid.py
  bspline_ls/        b-surface/fit_surface.py
  poisson_closed/    closed-surface/fit_surface.py  (poisson_open/ with --open)
```

**What the fits agree on:** under cross-validation the landscape is a broad
tilt with one or two rises, not rugged terrain. Nine B-spline control points
explain almost as much as the best fit does (about 9% of the variance for
sadness up to about 35% for wonder).

### Geodesics

`geodesic_arrows.py` and `geodesic_interactive.py` draw shortest paths
measured along the landscape rather than across the flat plane. The paths are
exact (Kirsanov's implementation of Mitchell-Mount-Papadimitriou, vendored in
`geometry/geodesic_cpp/`). The vertical scale `alpha` is the exchange rate
between score units and plane units, and so decides how far a path bends
around a hill.

## The narrative arc

The central object is the **windowed series**: slide a window of 40 paragraphs,
stepping by 20, over the book and mean-pool each window. That gives 39 windows
for Alice, 44 for Hamlet and 104 for Pride and Prejudice. Consecutive windows
share half their text, so a window is a scene rather than a remark.

- **`narrative-arc/`**: the arc as a curve (figures in
  [docs/figures](docs/figures/README.md#4-the-2-d-arc-and-its-b-spline-fit)). 2-D and 3-D paths under PCA, UMAP
  and t-SNE; a uniform B-spline fitted through the windows; the **narrative
  tube** (a cross-section per window, swept along the curve with
  rotation-minimizing frames); parameter sweeps; and two self-contained HTML
  explorers. See [its README](narrative-arc/README.md).
- **`visualization/windows.py`**: the same series for the landscape scripts,
  saved once to `output/<book>/<model>/windows/w40_s20/series.npz`.
- **Validation** (`arc_comparison.py`, `arc_comparison_projections.py`): the
  arc against the raw per-paragraph path and against a shuffled reading order.
  Read these before trusting any arc figure.
- **The arc on the landscape** (`output/…/narrative_arc_3d/`), split by where
  the height comes from:

  ```
  height_from_text/     narrative_arc_3d.py   the emotion the window carried
  height_from_terrain/  arc_on_surface.py     the fitted surface at the arc's (x, y)
  mood_axis/            arc_emotion_axis.py   six emotions collapsed to one height
  curve_smoothing/      arc_smooth.py         the arc smoothed on the surface
  ```

  `arc_smooth.py` implements distance-based curve smoothing (Pawellek et al.
  2024): one tolerance `tau` bounds how far the smoothed arc may leave the
  original.

## Pipeline reference site

```
python site/export_reference.py  # flow, parameters and algorithms, from the source AST
python site/build.py             # figures and metrics, from output/
open site/index.html
```

Six tabs: Pipeline, Algorithms (what is vendored, third-party or written here),
Parameters (every CLI flag, searchable), Tuning, Figures and Conflicts. Nothing
on the page is typed by hand, so rebuild after any run.

## Output conventions

`visualization/paths.py` decides every output directory. Parameters that
change the result become a **variant subdirectory**, so two settings never
overwrite each other, and every run writes a **`params.json`** with its full
arguments and the estimator used. `output/` is not tracked; the scripts
regenerate it.

## Third-party code

| code | where | license |
|---|---|---|
| Kirsanov, exact geodesics | `geometry/geodesic_cpp/` | see `UPSTREAM_README.txt` |
| PoissonRecon (Kazhdan) | `closed-surface/vendor/PoissonRecon/` | MIT |
| bspline-regression (Stebbing) | `narrative-arc/vendor/bspline_regression/` | MIT |
| B-spline-Curves-and-Surfaces (MATLAB) | `b-surface/vendor/` | none stated upstream; kept as the specification of the numpy port |

## References

- M. Pawellek, C. Rössl, K. Lawonn. *Distance-Based Smoothing of Curves on
  Surface Meshes.* Computer Graphics Forum 43(5), 2024.
  [doi:10.1111/cgf.15015](https://doi.org/10.1111/cgf.15015)
- M. Kazhdan, H. Hoppe. *Screened Poisson Surface Reconstruction.* ACM TOG
  32(3), 2013.
- J. Mitchell, D. Mount, C. Papadimitriou. *The Discrete Geodesic Problem.*
  SIAM J. Computing 16(4), 1987.
- W. Wang, B. Jüttler, D. Zheng, Y. Liu. *Computation of Rotation Minimizing
  Frames.* ACM TOG 27(1), 2008.
- P. Eilers, B. Marx. *Flexible Smoothing with B-splines and Penalties.*
  Statistical Science 11(2), 1996.
