# Interactive

Three pages. Download one and open it in a browser (it loads d3 and three.js
from the web).

| page | what it shows | rebuild |
|---|---|---|
| [`story_map.html`](story_map.html) | The narrative arc crossing an emotion landscape, flat or in 3-D (the arc riding the terrain), on any saved surface (kernel, B-spline, Poisson, mood), with the text and the emotion over reading order. | `python arc_on_surface/story_map.py` |
| [`mood_viewer.html`](mood_viewer.html) | The mood surface in 3-D: its height is the emotion (sadness at the bottom, humor at the top, labelled on the axis), and the narrative arc lies on it; with the text and the mood over reading order. | `python surface/mood/mood_viewer.py` |
| [`arc_viewer.html`](arc_viewer.html) | The arc in three components under PCA, UMAP or t-SNE, with its fitted B-spline and the emotion tube, and the text. | `python arc/curve/arc_viewer.py` |

Everywhere: drag the timeline, arrow keys step (shift for 20), space plays;
drag to turn a 3-D view. The story map jumps to the nearest paragraph on a click
(double-click in 3-D; on the mood surface too). The address keeps the view, so a link such as
`story_map.html#pride_and_prejudice/sadness/900/3d` opens exactly there.
