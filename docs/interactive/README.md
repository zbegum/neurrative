# Interactive

[`index.html`](index.html): download it and open it in a browser (it loads d3
and three.js from the web). One page, three tabs:

| tab | what it shows | built by |
|---|---|---|
| story map | the narrative arc crossing an emotion landscape, flat or in 3-D (the arc riding the terrain), on a saved surface (kernel, B-spline or Poisson), with the text and the emotion over reading order | `arc_on_surface/story_map.py` |
| mood surface | the mood surface in 3-D: its height and colour are the emotion (sadness at the bottom, humor at the top), and the narrative arc lies on it | `surface/mood/mood_viewer.py` |
| arc 3-D | the arc in three components under PCA, UMAP or t-SNE, with its fitted B-spline and the emotion tube | `arc/curve/arc_viewer.py` |

The page shows *Alice's Adventures in Wonderland* (each builder takes
`--books` for others). The reading position follows you from tab to tab.
Axes: the floor is the PCA plane of the paragraph embeddings (PC1, PC2); the
height is the emotion's score (story map) or the mood, sadness to humor (mood
surface). The arc is coloured by reading order, and the timeline shows the
chapters. Everywhere: drag
the timeline, arrow keys step (shift for 20), space plays; drag to turn a 3-D
view; the story map and the mood surface jump to a paragraph on a click
(double-click in 3-D).

Rebuild everything with `python docs/interactive/build.py`: it runs the three
builders and embeds their pages. Each builder also writes its page on its own,
into its step's `output/`.

Moving through the book: ‹ and › jump to the previous / next chapter, ▶ plays the
book continuously (about 90 seconds), and **ride** puts the camera on the arc,
looking ahead along it, with the part already passed coloured by reading order.
Keys: arrows step (shift for 20), space plays, [ and ] jump a chapter.
