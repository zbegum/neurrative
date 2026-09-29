# projection — step 1: the semantic plane

The paragraphs projected to 2-D (and 3-D) with PCA, UMAP and t-SNE, and the
question of which projection to trust. PCA's plane is also what every surface
in `surface/` is fitted over, so run `embedding.py` before step 2.

| script | what |
|---|---|
| `embedding.py` | 2-D projections (PCA / UMAP / t-SNE) colored by chapter, reading order or any emotion. Writes `pca.npy`, which the surfaces read. |
| `embedding_grid.py` | 3×3 parameter sweeps for UMAP (`n_neighbors` × `min_dist`) and t-SNE (perplexity × learning rate). PCA has no parameters, which is the point. |
| `projection_comparison.py` | PCA vs UMAP vs t-SNE across neighbourhood scale, scored by trustworthiness and chapter purity over a shuffled-label control. Reuses the coordinates `embedding_grid.py` saved, so `--all` covers every book × model in seconds. |
| `chain_umap.py` | Reading order as graph structure: UMAP's fuzzy graph with reading-order edges added, weighted by `beta`, and swept. `surface/mood` uses its layout by default. |
| `emotion_3d.py` | Emotion as the third axis, or as the color over a 3-component projection. |
| `inspect_points.py` | Trace a point on a plot back to its paragraph. |

Output: `projection/output/<book>/<model>/{structure_2d, projection_comparison,
chain_umap, emotions_3d}/`.

## What it found

PCA keeps the least local structure in every book (trustworthiness 0.72–0.76
against 0.83–0.92 for UMAP and t-SNE), and the scale setting matters far less
than the choice of method. Adding reading order to UMAP trades
trustworthiness for order along a smooth curve. Figures and numbers:
[docs/figures](../docs/figures/README.md#1-pca-vs-umap-vs-t-sne-paragraph-level).
