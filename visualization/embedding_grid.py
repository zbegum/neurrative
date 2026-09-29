"""
3x3 parameter sweeps for UMAP and t-SNE: how much of the layout is the setting?

PCA has no parameters, so it does not appear here -- that asymmetry is the point.
UMAP and t-SNE both draw a different picture depending on knobs nobody has a
principled way to choose, and this is where that dependence is visible. One PNG
per color, rows and columns as the two swept parameters.

The saved `.npy` coordinates are the reusable half of the output.
`projection_comparison.py` loads them instead of refitting, which is why it can
score all nine book x model pairs in seconds -- so a sweep run here is paid for
once and reused, not thrown away after the picture is drawn.

Output lands in output/<book>/<model>/structure_2d/<projection>/, next to the
single views from `embedding.py`.

Example:

python visualization/embedding_grid.py --book alice_wonderland --model bge-m3
python visualization/embedding_grid.py --book alice_wonderland --model bge-m3 \
  --color emotions
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

import paths
from embedding_common import (add_common_args, check_perplexity, load_book,
                              models_for, open_output, panel, resolve_colors,
                              save_grid)

# Rows and columns of the UMAP grid.
UMAP_NEIGHBORS = [5, 15, 50]
UMAP_MIN_DISTS = [0.0, 0.1, 0.5]

# Rows and columns of the t-SNE grid.
TSNE_PERPLEXITIES = [5.0, 30.0, 100.0]
TSNE_LEARNING_RATES = [10.0, 200.0, 1000.0]


def draw_sweep(cells, specs, n_cols, output_dir, method, axis_label, seed):
  """One grid PNG per color, from cells already fitted.

  Fit once, redraw per color: the fitting is the slow part, and refitting for
  each coloring would multiply a several-minute sweep by the number of emotions.
  """
  for spec in specs:
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    sc = None
    for idx, (title, coords) in enumerate(cells):
      i, j = divmod(idx, n_cols)
      sc, handles = panel(axes[i][j], coords, spec, title,
                          f"{axis_label}-1", f"{axis_label}-2")
    save_grid(
      fig, sc, handles,
      f"{axis_label} parameter sweep (colored by {spec.name})",
      spec.label,
      os.path.join(output_dir, f"grid_{method}_{spec.name}_s{seed}.png"),
    )


def umap_grid(embeddings, specs, seed, structure_dir):
  output_dir = paths.projection_dir(structure_dir, "umap")
  cells = []
  for n_neighbors in UMAP_NEIGHBORS:
    for min_dist in UMAP_MIN_DISTS:
      print(f"  UMAP n_neighbors={n_neighbors} min_dist={min_dist}...")
      coords = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
      ).fit_transform(embeddings)
      np.save(
        os.path.join(output_dir, f"umap_n{n_neighbors}_d{min_dist}_s{seed}.npy"),
        coords,
      )
      cells.append((f"n_neighbors={n_neighbors}, min_dist={min_dist}", coords))

  draw_sweep(cells, specs, len(UMAP_MIN_DISTS), output_dir, "umap", "UMAP", seed)


def tsne_grid(embeddings, specs, seed, structure_dir):
  output_dir = paths.projection_dir(structure_dir, "tsne")
  cells = []
  for perplexity in TSNE_PERPLEXITIES:
    for learning_rate in TSNE_LEARNING_RATES:
      print(f"  t-SNE perplexity={perplexity} learning_rate={learning_rate}...")
      coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=seed,
      ).fit_transform(embeddings)
      np.save(
        os.path.join(output_dir, f"tsne_p{perplexity}_lr{learning_rate}_s{seed}.npy"),
        coords,
      )
      cells.append((f"perplexity={perplexity}, learning_rate={learning_rate}", coords))

  draw_sweep(cells, specs, len(TSNE_LEARNING_RATES), output_dir, "tsne", "t-SNE",
             seed)


def run_model(book, model, args):
  print(f"\n=== {book} / {model} ===")

  paragraphs, embeddings = load_book(book, model)
  specs = resolve_colors(args.color, book, paragraphs, args.min_score)
  print(f"  coloring by: {', '.join(s.name for s in specs)}")
  check_perplexity(TSNE_PERPLEXITIES, len(embeddings), book)

  output_dir = open_output(book, model, __file__, args, specs)

  umap_grid(embeddings, specs, args.seed, output_dir)
  tsne_grid(embeddings, specs, args.seed, output_dir)

  print(f"  -> {output_dir}")


def main():
  parser = add_common_args(argparse.ArgumentParser())
  args = parser.parse_args()

  for model in models_for(args, parser):
    run_model(args.book, model, args)

  print("\nDone.")


if __name__ == "__main__":
  main()
