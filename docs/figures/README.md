# Selected figures: projections, sweeps and the narrative arc

A curated set of the projection, sweep and narrative-arc figures, committed so
they can be read on GitHub without running anything. Every one is regenerated
by the command under it, into the `output/` folder of the step that makes it
(`projection/output/`, `arc/output/`, ...), along with the same figure for every
other book × model. Numbers are for `bge-m3` unless noted.

## 1. PCA vs UMAP vs t-SNE (paragraph level)

`python projection/embedding_grid.py --book <book> --all-models`, then
`python projection/projection_comparison.py --all`

Columns are neighbourhood scale (local / default / global): UMAP
`n_neighbors` 5 / 15 / 50, t-SNE perplexity 5 / 30 / 100; PCA has no parameter.
Each panel reports trustworthiness *T* (k = 15) and chapter purity, with the
excess over a shuffled-label control.

![Alice: PCA vs UMAP vs t-SNE](01_projection_grid_alice.png)
![Alice: the same metrics as curves](02_projection_metrics_alice.png)

| book | raw-embedding chapter purity | PCA *T* / purity | UMAP default *T* / purity | t-SNE default *T* / purity |
|---|---|---|---|---|
| Alice | 0.382 | 0.757 / 0.135 | 0.917 / 0.367 | 0.902 / 0.377 |
| Pride and Prejudice | 0.097 | 0.723 / 0.028 | 0.828 / 0.060 | 0.879 / 0.076 |
| Hamlet (acts) | 0.354 | 0.761 / 0.237 | 0.856 / 0.327 | 0.849 / 0.317 |

- **PCA loses the local structure.** Its trustworthiness is the lowest in all
  three books, and for Alice it keeps about a third of the chapter purity that
  the raw embedding has. UMAP and t-SNE keep it almost intact.
- **The scale knob matters less than the method.** Across the three columns,
  UMAP and t-SNE move by a few hundredths.
- **Pride and Prejudice's chapters are not semantic units.** Chapter purity is
  low even in the raw embedding (0.097), so no projection can show chapters
  there.

![Pride and Prejudice](03_projection_grid_pride.png)
![Hamlet](04_projection_grid_hamlet.png)

### Each method's own parameter grid

The 3×3 sweeps behind the table: UMAP over `n_neighbors` × `min_dist`, and
t-SNE over perplexity × learning rate.

![UMAP parameter grid](05_umap_parameter_grid_alice.png)
![t-SNE parameter grid](06_tsne_parameter_grid_alice.png)

## 2. Reading order as graph structure (chain-UMAP)

`python projection/chain_umap.py --book <book> --all-models`

UMAP's fuzzy graph gets extra edges between each paragraph and the next two,
weighted by `beta`. For Alice, going from `beta = 0` (plain UMAP) to `beta = 2`
raises the share of plotted neighbours that are within 5 paragraphs in the book
from 0.10 to 0.26, and chapter purity from 0.38 to 0.68. Trustworthiness falls
from 0.91 to 0.81. There is no principled `beta`; the curve is the result.

![Chain-UMAP trade-off](07_chain_umap_tradeoff_alice.png)
![Chain-UMAP layouts across beta](08_chain_umap_beta_sweep_alice.png)

## 3. The narrative arc under different projections and windows

`cd arc && python sweeps/sweep.py --all-books --all-models`, and
`python sweeps/grid_3d.py --all-books --all-models`

**Projection parameters** (window 20, stride 10: 78 windows). t-SNE perplexity
5–40 and UMAP `n_neighbors` 5–40, with how far the layout moves between seeds
(`seed_disp`) and how well a B-spline follows the windows in order.

![Arc: projection sweep](09_arc_projection_sweep_alice.png)

**Window size.** Rows are size:stride. The share of variance that 2-D PCA
captures grows with the window, from 21% at 10:5 to 49% at 80:40. The heavily
overlapping 40:5 row is there as a warning: UMAP and t-SNE trustworthiness
reach 0.998 and 0.9995 only because adjacent windows share most of their
paragraphs.

![Arc: window sweep](10_arc_window_sweep_alice.png)

**Embedding model.** The arc's geometry barely depends on the model: the
Spearman correlation between window-to-window distances is 0.94
(bge-m3 vs e5), 0.94 (bge-m3 vs qwen3) and 0.92 (e5 vs qwen3).

![Arc: model sweep](11_arc_model_sweep_alice.png)

**3-D arc and its flat views** for four window sizes, PCA and UMAP. A loop that
seems to cross itself in one flat view can be checked against the other two.

![Arc: 3-D and flat views](12_arc_3d_views_alice.png)

## 4. The 2-D arc and its B-spline fit

`cd arc && python curve/arc_2d.py --all-books --all-models --fit --show-control`
(and `arc_3d.py` with the same flags)

Windows of 40 paragraphs stepping by 20, joined in reading order: **O** is the
opening, **X** the ending, and the color is reading position. The thick curve is
a cubic uniform B-spline fitted with `bspline-regression`
(`arc/narrative_arc/curves.py`), which optimises the control points
(the dashed polygon) and each window's position on the curve at the same time.
The fit starts from reading order, so a path that crosses itself keeps its two
passes apart. Defaults: 12 control points, `lambda` 0.1.

![Alice: 2-D arc with B-spline, PCA / UMAP / t-SNE](13_arc_2d_bspline_alice.png)

**How tight should the curve be?** Columns are the number of control points,
rows are `lambda`, the penalty on the distance between neighbouring control
points. Each panel gives the residual, the share of consecutive windows that
land out of order on the curve, and the gap between the curve's ends and the
first and last windows. With few control points or a large `lambda` the curve
shrinks and misses both the loops and the ends (6 control points at
`lambda` 10: residual 0.60, end gap 0.83). With many control points and a small
`lambda` it follows every window (16 control points at `lambda` 0.01: residual
0.06, 18% out of order, end gap 0.05) but starts to loop. The default sits in
between.

![B-spline fit sweep: control points x lambda](15_bspline_fit_sweep_alice.png)

**In 3-D** the third principal component (10.7% of the variance for Alice)
separates passes that overlap in the plane. The fitter is the same one, since
bspline-regression works in any dimension.

![Alice: 3-D arc with B-spline](16_arc_3d_bspline_alice.png)

The other two books, colored by reading position:

![Pride and Prejudice: 2-D arc with B-spline](17_arc_2d_bspline_pride.png)
![Hamlet: 2-D arc with B-spline](18_arc_2d_bspline_hamlet.png)

## 5. The emotion landscape

For each emotion, a surface over the PCA plane whose height is that emotion's
score: `z = emotion(PC1, PC2)`. Every method is in its own folder under
`surface/`; see [surface/README.md](../../surface/README.md).

**The raw material.** Each paragraph as a point, nothing fitted: the scores are
noisy, and neighbouring paragraphs can disagree a lot. Every surface below is a
way of averaging this cloud.
`python surface/points/raw_points.py --book <book> --model bge-m3`

![Alice: wonder, raw points](19_points_wonder_alice.png)

**Kernel smoothing** (`surface/kernel/`): the height at a point is a weighted
average of the paragraphs near it, with the bandwidth chosen by cross-validation
and scored once on a held-out 20% of paragraphs. The second figure walks the
bandwidth from under- to over-smoothed, for every kernel.
`python surface/kernel/gaussian.py --book <book> --model bge-m3`

![Alice: wonder, Gaussian kernel surface](20_kernel_gaussian_wonder_alice.png)
![Alice: wonder, every kernel across bandwidths](21_kernel_bandwidth_wonder_alice.png)

| held-out improvement over a flat average | most structure | least structure |
|---|---|---|
| Alice | wonder 18–19% | sadness 2–5% |
| Pride and Prejudice | sadness 19–22% | wonder about 1% |

The kernels agree closely with each other; which emotion has spatial structure
depends on the book.

**B-spline surface** (`surface/bspline/`): one global smooth surface defined by
a small grid of control heights. Cross-validation keeps the grid tiny (2–3 knots
per axis). With only nine control points over the whole plane (second figure,
control net in red), the surfaces are nearly the same: the landscape is a tilt
with one bend, not terrain.
`python surface/bspline/fit_surface.py --book <book> --model bge-m3 --grid`

![Alice: B-spline surfaces, all six emotions](22_bspline_six_emotions_alice.png)
![Alice: B-spline surfaces with 3x3 control points](23_bspline_nine_control_points_alice.png)
![Pride and Prejudice: B-spline surfaces, all six emotions](27_bspline_six_emotions_pride.png)

**Poisson reconstruction** (`surface/poisson/`): the landscape as the lid of a
watertight solid (Euler characteristic 2 for all six emotions in both books),
cut back to an open sheet over the whole plane. The orange line is the edge of
where the book's paragraphs actually are; outside it the surface is extension,
not data.
`python surface/poisson/fit_surface.py --book <book> --model bge-m3 --open`

![Alice: wonder, Poisson surface](24_poisson_wonder_alice.png)

**The mood surface** (`surface/mood/`): the six emotions collapsed to one value
per paragraph on a sadness → humor spectrum, then one surface, drawn here over
the chain-UMAP layout. The blends figure shows the four ways of collapsing six
emotions to one height. For `project` and `pc1` the height is a direction
(heavy → light) rather than a position on the spectrum, so read their emotion
ticks only as low and high.
`python surface/mood/run.py --book <book> --model bge-m3 --figure surface`

![Alice: mood surface](25_mood_surface_alice.png)
![Alice: four blends](26_mood_blends_alice.png)

## 6. The arc on the landscape

The narrative arc from step 3, with a third axis that comes from the emotions.
See [arc_on_surface/README.md](../../arc_on_surface/README.md).

**On the terrain** (`arc_on_surface.py`): windows of 10 paragraphs stepping by
5, each placed on the fitted surface of one emotion. By default consecutive
windows are joined by straight lines in the plane lifted onto the surface, so
the route's shadow is exactly the 2-D arc; `--legs geodesic` joins them by
shortest paths along the terrain instead. The dashed green line is the geodesic
between the two ends, for context. Grey is terrain with too few paragraphs
under it to be measured.
`python arc_on_surface/arc_on_surface.py --book <book> --model bge-m3`

![Alice: the arc on each emotion surface](28_arc_on_terrain_alice.png)

**On one mood axis** (`arc_emotion_axis.py`): the same arc over a single surface
whose height is the mood, so following the route is watching the mood change.
Colour is reading order.

![Alice: the arc on the mood axis](29_arc_on_mood_axis_alice.png)
![Pride and Prejudice: the arc on the mood axis](34_arc_on_mood_axis_pride.png)

**Smoothing the arc on the surface** (`arc_smooth.py`, Pawellek et al. 2024):
one tolerance `tau` sets how closely the smoothed curve follows the raw arc,
from the straight geodesic between the ends (`tau` = 0) to hugging every window
(large `tau`).

![Alice: distance-based smoothing across tau](30_arc_curve_smoothing_alice.png)

**Geodesics toward an emotion** (`geodesic_arrows.py`): for the strongest rises
in wonder between consecutive paragraphs, the shortest path along the wonder
terrain. The paths bend around hills rather than going over them; how much
depends on the vertical scale `alpha`.

![Alice: geodesics along the wonder terrain](31_geodesics_wonder_alice.png)

**The arc on the mood surface** (`surface/mood/run.py --figure arc`): one
central paragraph per chapter, joined by exact geodesics on the mood surface,
coloured by reading position.

![Alice: the arc as geodesics on the mood surface](32_mood_arc_geodesic_alice.png)

**The narrative tube** (`arc/tube/arc_tube.py`): the fitted 3-D arc thickened
into a tube whose cross-section has one vertex per emotion, so the tube bulges
toward whichever emotion is strong in that part of the book.

![Alice: narrative tube](33_tube_alice.png)
![Pride and Prejudice: narrative tube](35_tube_pride.png)
