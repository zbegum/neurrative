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
  plane, several different ways.
- **The narrative arc.** The book as an ordered curve: windows of paragraphs,
  mean-pooled and joined in reading order, then fitted, validated, lifted onto
  the landscape and thickened into a tube.

The goal is interpretability and visualization, not a predictive model.
Selected results, with numbers, are in **[docs/figures](docs/figures/README.md)**;
interactive pages to explore them are in **[docs/interactive](docs/interactive/README.md)**.

## Repository layout

One folder per purpose, in pipeline order. Every step folder has a README and
keeps its results next to its code, in its own `output/` (not tracked).

```
books/              DATA: three books, three embedding models, emotion scores
preprocess/         step 0  build the data: text -> paragraphs -> embeddings -> scores
geometry/           library: smoothers, meshes, exact geodesics, curve smoothing
common/             library: output paths, data loading, the windowed series

projection/         step 1  the semantic plane: PCA / UMAP / t-SNE, which to trust
surface/            step 2  the emotion landscape, one folder per fitting method
  points/             the raw point cloud, nothing fitted
  kernel/             kernel smoothing: Gaussian, Epanechnikov, local linear, LOESS
  bspline/            least-squares tensor-product B-spline
  poisson/            screened Poisson reconstruction (a watertight solid)
  mood/               six emotions -> one mood height, and the arc on it
arc/                step 3  the narrative arc
  curve/              2-D / 3-D paths, the B-spline fit, the explorer
  tube/               the narrative tube and its explorer
  sweeps/             window / fit / projection / model sweeps
  validation/         is the arc real?
  narrative_arc/      the library the arc scripts share
arc_on_surface/     step 4  the arc lifted onto the landscape, and geodesics

docs/figures/       curated figures with captions
docs/interactive/   four interactive HTML pages (the arc and tube explorers, ...)
site/               a pipeline reference page generated from the code
```

## Setup

```bash
pip install -r requirements.txt               # everything except preprocess/
pip install -r requirements-preprocess.txt    # only to add a new book
```

Two parts build native code and need only a C++ compiler: `geometry/` compiles
the exact geodesic solver automatically on first import, and
`surface/poisson/build_vendor.sh` builds PoissonRecon once.

Every script can be run from anywhere; each one puts the repository root and
`common/` on its own import path.

## Running the pipeline

Each step reads what the one before it wrote. For one book and model
(swap in any book and model from the table below):

```bash
# step 1: the plane (PCA is what every surface is fitted over)
python projection/embedding.py --book alice_wonderland --model bge-m3
python projection/chain_umap.py --book alice_wonderland --model bge-m3  # needed by surface/mood by default

# step 2: surfaces, any subset
python surface/points/raw_points.py --book alice_wonderland --model bge-m3
python surface/kernel/gaussian.py --book alice_wonderland --model bge-m3  # or epanechnikov / local_linear / loess / loo
python surface/bspline/fit_surface.py --book alice_wonderland --model bge-m3 --grid
bash surface/poisson/build_vendor.sh  # once: builds PoissonRecon
python surface/poisson/fit_surface.py --book alice_wonderland --model bge-m3 --open
python surface/mood/run.py --book alice_wonderland --model bge-m3 --figure surface

# step 3: the arc
python common/windows.py --book alice_wonderland --model bge-m3  # the windowed series
python arc/curve/arc_2d.py --book alice_wonderland --model bge-m3 --fit
python arc/validation/arc_comparison.py --book alice_wonderland --model bge-m3

# step 4: the arc on the landscape
python arc_on_surface/arc_on_surface.py --book alice_wonderland --model bge-m3
python arc_on_surface/geodesic_arrows.py --book alice_wonderland --model bge-m3
```

The READMEs in each folder list every script and its options.

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
Hamlet has no emotion scores, so the surface steps do not apply to it.

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

## Output conventions

`common/paths.py` decides every output directory:
`<step folder>/output/<book>/<model>/<figure>/`, for example
`surface/poisson/output/alice_wonderland/bge-m3/poisson_open/`. Parameters that
change the result become a **variant subdirectory**, so two settings never
overwrite each other, and every run writes a **`params.json`** with its full
arguments and the estimator used.

## Pipeline reference site

```
python site/export_reference.py  # flow, parameters and algorithms, from the source AST
python site/build.py             # figures and metrics, from every output/ folder
open site/index.html
```

Six tabs: Pipeline, Algorithms (what is vendored, third-party or written here),
Parameters (every CLI flag, searchable), Tuning, Figures and Conflicts. Nothing
on the page is typed by hand, so rebuild after any run.

## Third-party code

| code | where | license |
|---|---|---|
| Kirsanov, exact geodesics | `geometry/geodesic_cpp/` | see `UPSTREAM_README.txt` |
| PoissonRecon (Kazhdan) | `surface/poisson/vendor/PoissonRecon/` | MIT |
| bspline-regression (Stebbing) | `arc/vendor/bspline_regression/` | MIT |
| B-spline-Curves-and-Surfaces (MATLAB) | `surface/bspline/vendor/` | none stated upstream; kept as the specification of the numpy port |

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
