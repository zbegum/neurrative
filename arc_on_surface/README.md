# arc_on_surface — step 4: the arc on the landscape

The narrative arc (step 3) drawn on the emotion landscape (step 2): the same
curve, with a third axis that comes from the emotions. The scripts differ in
**where that height comes from**, which is also how their output is organised
under `arc_on_surface/output/<book>/<model>/narrative_arc_3d/`:

| height from | output | scripts |
|---|---|---|
| the text: the emotion each window carried | `height_from_text/` | `narrative_arc_3d.py` |
| the terrain: the fitted surface at the arc's (x, y) | `height_from_terrain/` | `arc_on_surface.py`, `arc_on_surface_views.py` |
| one mood axis: six emotions collapsed to one height | `mood_axis/` | `arc_emotion_axis.py`, `arc_emotion_grid.py`, `arc_emotion_surface_grid.py`, `arc_emotion_smoother_grid.py` |
| the terrain, with the arc smoothed on it | `curve_smoothing/` | `arc_smooth.py` |

`story_map.py` writes `output/story_map.html`: the arc moving across one
emotion's landscape (2-D contours), linked to the paragraph text and to a
timeline of that emotion. The copy in `docs/interactive/` is this page.

And, under `geodesics/`, paths between paragraphs rather than along the arc:

| script | what |
|---|---|
| `geodesic_arrows.py` | for each step where an emotion rises, the shortest path along the terrain between the two paragraphs; the strongest rises are drawn |

The scripts share code: `arc_on_surface.py` and `arc_emotion_axis.py` are also
imported by the others, which is why they sit together in one folder. The mood
axis uses the repository's single mood definition, `surface/mood/mood.py`.

## Notes

- **Only PCA can carry a terrain height.** `height_from_text` works under UMAP
  and t-SNE too; the other modes read a height off a surface fitted over PCA.
- **`arc_smooth.py`** implements distance-based curve smoothing (Pawellek,
  Rössl and Lawonn, 2024; `geometry/curve_smoothing.py`): one tolerance `tau`
  bounds how far the smoothed arc may leave the original.
- **Geodesics are exact** (Kirsanov's Mitchell-Mount-Papadimitriou solver,
  `geometry/geodesic_cpp/`). The vertical scale `--alpha` is the exchange rate
  between score units and plane units, and so decides how far a path bends
  around a hill. `geodesic_arrows.py` solves every rising pair, which takes a
  few minutes per emotion on Alice and longer on bigger books; pass
  `--emotions wonder` to run one.

## Known issue

On larger books `arc_on_surface.py` with all six emotions in one run is far
slower than the sum of its parts: on Pride and Prejudice each emotion alone
takes 34–62 s, but the six together did not finish in 51 minutes. Until that is
tracked down, run one emotion at a time with `--emotions <name>`.
