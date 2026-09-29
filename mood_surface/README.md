# mood_surface

One emotion surface for a book, and the narrative arc drawn on it.

Each paragraph's six emotion scores are collapsed to a single **mood** value on a
spectrum (sadness → humor). A Gaussian surface is fitted to the paragraphs'
`(x, y, mood)` points over a 2D layout of the book (PCA or chain-UMAP). The
narrative arc is a chosen set of paragraphs joined by **geodesics** that lie on
that surface.

## Run

    python mood_surface/run.py --figure surface     # the mood surface
    python mood_surface/run.py --figure plane        # the 2D layout (scatter)
    python mood_surface/run.py --figure arc          # arc over the surface (geodesic)

    python mood_surface/run.py --figure arc --sampling douglas_peucker --n 60
    python mood_surface/run.py --figure arc --legs straight   # no solver needed

Comparison grids:

    python mood_surface/grids.py --grid blends       # six emotions -> one height
    python mood_surface/grids.py --grid sampling     # which paragraphs the arc uses
    python mood_surface/grids.py --grid chapters     # coarse vs fine units
    python mood_surface/grids.py --grid angles       # one arc, four cameras

Figures land in `mood_surface/figures/`.

## Files

| file | what |
|------|------|
| `data.py`     | load paragraphs, emotion scores, chapters, 2D coordinates |
| `mood.py`     | collapse six emotions into one mood value (`banded`, `softmax`, `dominant`) |
| `surface.py`  | fit the Gaussian surface, evaluate its height anywhere |
| `sampling.py` | pick which paragraphs the arc runs through |
| `geodesic.py` | join waypoints with geodesics along the surface |
| `plot.py`     | draw the plane, the surface, and the arc |
| `run.py`      | one figure at a time |
| `grids.py`    | comparison grids |

## Key choices

- `--blend banded` gives each emotion its own band, so the height reads as an
  emotion. `softmax` is a weighted mean and stays near the middle.
- `--plane chain_umap --beta 2` spreads the paragraphs more evenly than PCA.
- `--legs geodesic` is the shortest path along the surface; `--alpha` is the
  height-to-plane exchange rate that defines "shortest". `--legs straight` lifts
  a straight line onto the surface and needs no solver.
- `--mask-floor` (a support percentile) can cut holes where paragraphs are
  sparse; off by default, so the surface covers the whole plane.

## Depends on

- Data: `books/<book>/{processed.json,paragraph_scores.json}` and the 2D
  coordinates under `output/<book>/<model>/`. Produce those first:

      python visualization/chain_umap.py --beta 2   # --plane chain_umap (default)
      python visualization/embedding.py             # --plane pca
- Geodesic legs use the `geometry/` package at the repo root (Gaussian smoother,
  mesh, exact geodesic solver, built from C++ on first import). Not needed for
  `--legs straight` or the plane/surface figures.
