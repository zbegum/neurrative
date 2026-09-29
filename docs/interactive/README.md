# Interactive pages

Three self-contained HTML pages. Download one and open it in a browser (it loads
plotly from the web, so it needs an internet connection). GitHub shows HTML as
source rather than rendering it.

| page | what you can do | made by |
|---|---|---|
| [`arc_explorer.html`](arc_explorer.html) | The narrative arc for every book, model, projection (PCA / UMAP / t-SNE), window size and step, chosen from dropdowns. The 3-D arc with two B-spline fits and its three flat views; hover a window, or scrub the reading-position slider, to read its chapter and opening text. | `python arc/curve/explorer.py` |
| [`tube_explorer.html`](tube_explorer.html) | The narrative tube for every book, model, projection, window and cross-section type, with live sliders for thickness, smoothing and rounding. | `python arc/tube/tube_explorer.py` |
| [`mood_surface_alice.html`](mood_surface_alice.html) | The mood surface for Alice. Hover any paragraph for its reading position, mood, dominant emotion, six emotion scores and summary; click a legend entry to colour by dominant emotion. | `python surface/mood/run.py --figure interactive` |

Every other interactive figure (single arcs, tubes, the arc with emotion as
its height, the curve smoothing, and the arc on each emotion surface) is written by the scripts into their step's
`output/` folder; the two explorers already contain all the single arcs and
tubes.
