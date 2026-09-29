# surface/mood — one mood surface, and the arc on it

The other surface methods fit **six** landscapes, one per emotion. This one
fits **one**: each paragraph's six scores are first collapsed to a single mood
value on a heavy → light spectrum,

    sadness → danger → confusion → curiosity → wonder → humor
     (low)                                              (high)

and one Gaussian surface is fitted to those values over a 2-D layout (PCA or
chain-UMAP). The height then reads directly as an emotion, and the narrative
arc is drawn across it as a chain of paragraphs joined by **geodesics**: the
shortest paths along the surface, so the route bends around emotional hills
instead of cutting through them.

## Run

    python surface/mood/run.py --figure surface       # the mood surface
    python surface/mood/run.py --figure plane         # the 2-D layout, by reading order and by mood
    python surface/mood/run.py --figure arc           # the arc over the surface (geodesics)
    python surface/mood/run.py --figure interactive   # rotatable HTML, every paragraph hoverable

    python surface/mood/run.py --figure arc --sampling douglas_peucker --n 60
    python surface/mood/run.py --figure arc --legs straight   # no geodesic solver needed

Comparison grids:

    python surface/mood/grids.py --grid blends      # the five ways six emotions become one height
    python surface/mood/grids.py --grid sweep       # softmax temperature x bandwidth
    python surface/mood/grids.py --grid sampling    # which paragraphs the arc runs through
    python surface/mood/grids.py --grid chapters    # coarse vs fine narrative units
    python surface/mood/grids.py --grid windows     # the arc for each saved window setting
    python surface/mood/grids.py --grid angles      # one arc from four cameras

Everything takes `--book` and `--model`. The default layout is chain-UMAP, so
run `projection/chain_umap.py --beta 2` first (or pass `--plane pca` after
`projection/embedding.py`). `--grid windows` also needs `common/windows.py`.

Output: `surface/mood/output/<book>/<model>/`.

## Files

| file | what |
|---|---|
| `mood.py` | **the single definition of mood in the repository**: the spectrum, the per-emotion normalizer and the five blends. `arc_on_surface/` imports it too. |
| `surface.py` | fit the Gaussian surface and evaluate its height anywhere |
| `sampling.py` | pick which paragraphs the arc runs through |
| `geodesic.py` | join waypoints with exact geodesics along the surface |
| `plot.py` | draw the plane, the surface and the arc |
| `interactive.py` | the hoverable HTML page |
| `data.py` | load paragraphs, scores, chapters and the 2-D layout |
| `run.py` / `grids.py` | one figure at a time / the comparison grids |

## Choices

- **Blend** (`--blend`), how six emotions become one height:
  `banded` (default) gives each emotion its own band, so the height names an
  emotion and still shows how clearly it wins; `softmax` is a weighted mean and
  sits near the middle unless one emotion dominates; `dominant` is a step
  function; `project` is a sum rather than a mean, so intensity survives; `pc1`
  lets the data choose the axis.
- **Normalizer** (`--norm`): each emotion is made comparable before blending.
  `rank` (default) uses percentiles, and tied scores share their average rank.
- `--plane chain_umap --beta 2` spreads the paragraphs more evenly than PCA.
- `--legs geodesic` is the shortest path along the surface; `--alpha` is the
  height-to-plane exchange rate that defines "shortest".
- `--mask-floor` (a support percentile) cuts holes where paragraphs are sparse;
  off by default, so the surface covers the whole plane.

## History

This folder merges three versions of the idea: this package, an earlier
single-script version, and the mood axis in `arc_on_surface/arc_emotion_axis.py`.
They now share `mood.py`. The earlier script's interactive page, temperature ×
bandwidth sweep, per-window arc grid and the `project` and `pc1` blends were
carried over; a few cosmetic options were not (`--no-points`, measured z-ticks,
drawing points at their own mood rather than on the surface).

The merge also fixed the rank normalizer: the old version ranked tied scores by
their position in the book, so two paragraphs with the same score could get
different values, and on Alice 14% of paragraphs changed dominant emotion once
ties were handled by average rank.
