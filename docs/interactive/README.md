# Interactive

[`story_map.html`](story_map.html): download it and open it in a browser (it
loads d3 from the web).

The map is the PCA plane with one emotion's landscape; the line is the
narrative arc (40-paragraph windows); the timeline below is that emotion over
reading order. Drag the timeline, click the map, use the arrow keys (shift for
bigger steps) or press space to play. The address keeps the position, so a link
like `story_map.html#alice_wonderland/humor/412` opens at that paragraph.

Rebuild with `python arc_on_surface/story_map.py`. The older explorers
(`arc/curve/explorer.py`, `arc/tube/tube_explorer.py`, `surface/mood/run.py
--figure interactive`) still write their pages into `output/`.
