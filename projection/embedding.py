"""
Visualize paragraph embeddings as 2-D projections of the semantic manifold.

One plot per projection at the parameters you pass: PCA, UMAP and t-SNE, plus
the storyline (UMAP with the paragraphs connected in reading order).

What the colors carry is up to you: --color takes any mix of

  chapter     the chapter a paragraph belongs to
  paragraph   its index in reading order
  emotions    every emotion in paragraph_scores.json (or name them one by one,
              e.g. --color wonder danger)

so the same layout can be read as narrative structure or as emotion. Each
projection is computed once and re-plotted per color -- the fitting is the slow
part, not the drawing.

Output lands in projection/output/<book>/<model>/structure_2d/<projection>/.

For the 3x3 parameter sweeps, see `embedding_grid.py`. For the three methods
scored against each other, see `projection_comparison.py`.

Example:

python projection/embedding.py --book alice_wonderland --model bge-m3
python projection/embedding.py --book alice_wonderland --model bge-m3 \
  --color chapter wonder danger
"""

import argparse
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from embedding_common import (ColorSpec, add_common_args, check_perplexity,
                              load_book, models_for, open_output,
                              resolve_colors, scatter_plot)


def run_pca(embeddings, specs, args, structure_dir):
  """PCA is parameter-free -- there is nothing to sweep.

  It is not, however, automatically *reproducible*. At this shape (789 x 1024)
  scikit-learn's `svd_solver="auto"` selects randomized SVD -- n_components is
  well under 0.8 * min(shape) and min(shape) > 500 -- and randomized SVD without
  a seed draws a different projection matrix every run. Measured here: two calls
  on the same embeddings differ by up to 2e-4, and `pca.npy` is the coordinate
  system every emotion surface, geodesic and arc lift is fitted over. The
  difference is far too small to move a conclusion, but "re-run it and you get
  the same coordinates" was false until this seed was passed.
  """
  print("  PCA...")
  output_dir = paths.projection_dir(structure_dir, "pca")
  pca = PCA(n_components=2, random_state=args.seed)
  coords = pca.fit_transform(embeddings)
  var = pca.explained_variance_ratio_
  print(f"  explained variance: {var.sum():.1%} over 2 components")

  np.save(os.path.join(output_dir, "pca.npy"), coords)
  for spec in specs:
    scatter_plot(
      coords, spec,
      f"PCA (colored by {spec.name})",
      f"PC1 ({var[0]:.1%} var)",
      f"PC2 ({var[1]:.1%} var)",
      os.path.join(output_dir, f"pca_{spec.name}.png"),
    )


def run_umap(embeddings, specs, args, structure_dir):
  print(f"  UMAP n_neighbors={args.neighbors} min_dist={args.min_dist}...")
  output_dir = paths.projection_dir(structure_dir, "umap")
  coords = UMAP(
    n_components=2,
    n_neighbors=args.neighbors,
    min_dist=args.min_dist,
    random_state=args.seed,
  ).fit_transform(embeddings)

  # UMAP layout depends on its parameters, so tag the filenames with them
  # to avoid overwriting runs with different settings.
  suffix = f"_n{args.neighbors}_d{args.min_dist}_s{args.seed}"
  np.save(os.path.join(output_dir, f"umap{suffix}.npy"), coords)

  for spec in specs:
    scatter_plot(
      coords, spec,
      f"UMAP (colored by {spec.name})",
      "UMAP-1", "UMAP-2",
      os.path.join(output_dir, f"umap_{spec.name}{suffix}.png"),
    )

  # The storyline is about reading order, so it is always colored that way.
  story = ColorSpec(
    "paragraph", np.arange(len(embeddings)), "plasma", "paragraph index",
    None, None, None,
  )
  scatter_plot(
    coords, story,
    "Storyline through the semantic manifold",
    "UMAP-1", "UMAP-2",
    os.path.join(output_dir, f"storyline{suffix}.png"),
    connect_story=True,
  )


def run_tsne(embeddings, specs, args, structure_dir):
  print(f"  t-SNE perplexity={args.perplexity}...")
  output_dir = paths.projection_dir(structure_dir, "tsne")
  coords = TSNE(
    n_components=2,
    perplexity=args.perplexity,
    random_state=args.seed,
  ).fit_transform(embeddings)

  # t-SNE layout depends on perplexity and the seed, so tag the filenames.
  suffix = f"_p{args.perplexity}_s{args.seed}"
  np.save(os.path.join(output_dir, f"tsne{suffix}.npy"), coords)

  for spec in specs:
    scatter_plot(
      coords, spec,
      f"t-SNE (colored by {spec.name})",
      "t-SNE-1", "t-SNE-2",
      os.path.join(output_dir, f"tsne_{spec.name}{suffix}.png"),
    )


def run_model(book, model, args):
  print(f"\n=== {book} / {model} ===")

  paragraphs, embeddings = load_book(book, model)
  specs = resolve_colors(args.color, book, paragraphs, args.min_score)
  print(f"  coloring by: {', '.join(s.name for s in specs)}")
  check_perplexity([args.perplexity], len(embeddings), book)

  output_dir = open_output(book, model, __file__, args, specs)

  run_pca(embeddings, specs, args, output_dir)
  run_umap(embeddings, specs, args, output_dir)
  run_tsne(embeddings, specs, args, output_dir)

  print(f"  -> {output_dir}")


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.add_argument("--neighbors", default=15, type=int,
                      help="UMAP n_neighbors. Matches UMAP's own default; lower "
                           "values fragment the manifold into local islands.")
  parser.add_argument("--min-dist", default=0.1, type=float)
  parser.add_argument("--perplexity", default=30.0, type=float,
                      help="t-SNE perplexity (must be < number of paragraphs).")
  args = parser.parse_args()

  for model in models_for(args, parser):
    run_model(args.book, model, args)

  print("\nDone.")


if __name__ == "__main__":
  main()
