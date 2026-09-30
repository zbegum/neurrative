"""
Visualize paragraph embeddings in 3-D against the per-paragraph emotion scores.

Two ways to spend the third axis, drawn once per emotion in paragraph_scores.json:

  proj3d   The embeddings are projected to three components and the emotion is
           the color. Every emotion shares one layout, so the plots differ only
           in color: comparing them shows where each emotion lives in the
           semantic manifold.

  score-z  The embeddings are projected to two components and the emotion score
           itself is the z axis (also mirrored in the color). Height above the
           semantic map is the emotion, so peaks are the paragraphs that carry it.

Each projection can be PCA, UMAP or t-SNE. The color is always the emotion, as
a viridis ramp, so in score-z height and color agree.

For every emotion we write a static PNG.

Example:

python projection/emotion_3d.py --book alice_wonderland --model bge-m3
python projection/emotion_3d.py --book alice_wonderland --model bge-m3 --proj umap
python projection/emotion_3d.py --book alice_wonderland --model bge-m3 --proj tsne \
  --mode score-z
"""

import os
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common"), os.path.join(_ROOT, "arc")]

import paths
from narrative_arc.projections import project as project_methods
from data import load_paragraphs, load_scores

# A single sequential ramp for every emotion: the score is a magnitude, and
# keeping one ramp across plots makes the panels directly comparable.
CMAP = "viridis"


def project(embeddings, args, n_components):
  """Reduce the embeddings to n_components dimensions (narrative_arc.projections).

  Fit at the dimensionality we actually plot. UMAP and t-SNE optimize a layout
  for a given target dimension, so the first two columns of a 3-component fit
  are not the 2-component fit -- reusing them across modes would quietly plot a
  layout nobody asked for. Returns (coords, axis labels, filename suffix).
  """
  (p,) = project_methods(embeddings, [args.proj], n_components, args.neighbors,
                         args.min_dist, args.perplexity, args.seed)
  return p.coords, list(p.labels), p.suffix


def static_plot(coords, scores, label, labels, title, output_path):
  """The points colored by `scores` (0..1), with a colorbar named `label`."""
  fig = plt.figure(figsize=(9, 8))
  ax = fig.add_subplot(111, projection="3d")

  sc = ax.scatter(
    coords[:, 0], coords[:, 1], coords[:, 2],
    c=scores, cmap=CMAP, vmin=0.0, vmax=1.0,
    s=14, alpha=0.9, linewidths=0, depthshade=False,
  )
  fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=label)

  ax.set_xlabel(labels[0])
  ax.set_ylabel(labels[1])
  ax.set_zlabel(labels[2])
  ax.set_title(title)

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True, type=str)
  parser.add_argument("--model", required=True, type=str)
  parser.add_argument("--proj", default="pca", choices=["pca", "umap", "tsne"])
  parser.add_argument("--mode", default="both",
                      choices=["proj3d", "score-z", "both"],
                      help="proj3d: z is a third component, emotion is the color. "
                           "score-z: z is the emotion score over a 2-D map.")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--neighbors", default=15, type=int,
                      help="UMAP n_neighbors. Matches UMAP's own default; lower "
                           "values fragment the manifold into local islands.")
  parser.add_argument("--min-dist", default=0.1, type=float)
  parser.add_argument("--perplexity", default=30.0, type=float,
                      help="t-SNE perplexity (must be < number of paragraphs).")
  parser.add_argument("--emotions", nargs="*", default=None,
                      help="Subset of emotions to plot (default: all).")
  args = parser.parse_args()

  paragraphs = load_paragraphs(args.book)
  embeddings = np.load(
    os.path.join("books", args.book, "embeddings", args.model, "embeddings.npy")
  )
  if len(embeddings) != len(paragraphs):
    raise ValueError(
      f"embeddings ({len(embeddings)}) and paragraphs ({len(paragraphs)}) mismatch."
    )

  emotions, matrix = load_scores(args.book, paragraphs)

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

  # No variant: the projection, mode and emotion are already in every filename.
  output_dir = paths.out_dir(args.book, args.model, paths.EMOTIONS_3D)
  paths.stamp(output_dir, __file__, args, emotions=selected)

  name = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[args.proj]
  modes = ["proj3d", "score-z"] if args.mode == "both" else [args.mode]

  for mode in modes:
    n_components = 3 if mode == "proj3d" else 2

    print(f"Running {name} with {n_components} components for {mode}...")
    coords, labels, suffix = project(embeddings, args, n_components)

    tag = "3d" if mode == "proj3d" else "scorez"
    np.save(os.path.join(output_dir, f"{args.proj}_{tag}{suffix}.npy"), coords)

    for emotion in selected:
      scores = matrix[:, emotions.index(emotion)]

      if mode == "proj3d":
        coords_3d = coords
        axis_labels = labels
      else:
        # The score itself is the z axis.
        coords_3d = np.column_stack([coords, scores])
        axis_labels = labels + [emotion]

      title = (f"{name} 3-D (colored by {emotion})" if mode == "proj3d"
               else f"{emotion} over the {name} map (z = {emotion})")
      base = f"{args.proj}_{tag}_{emotion}{suffix}"

      static_plot(coords_3d, scores, emotion, axis_labels, title,
                  os.path.join(output_dir, f"{base}.png"))

      print(f"  {emotion}: mean {scores.mean():.2f}, max {scores.max():.2f}")

  print()
  print("Done.")
  print(output_dir)


if __name__ == "__main__":
  main()
