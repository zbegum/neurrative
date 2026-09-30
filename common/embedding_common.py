"""
What the 2-D projection scripts share: colors, drawing, and where output goes.

`embedding.py` draws one plot per projection at the settings you pass;
`embedding_grid.py` draws the 3x3 parameter sweeps. They were one 435-line
script with a `--mode` flag, which meant the sweep ladders, the grid-panel
drawing and the single-figure drawing all sat in the same file and every reader
had to work out which half applied to them. Splitting them follows what the
smoothing family already does -- `surface/kernel/gaussian.py` next to
`smooth_grid.py`, over a shared `smooth_common.py`.

Everything here is the part that genuinely is common: what a color means, how a
point is drawn, and the output directory contract. The parameter ladders live
with the sweep that uses them, not here.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import paths
from data import load_embeddings, load_paragraphs, load_scores
# One ColorSpec and one resolver, in the arc package (common/data.py puts arc/
# on the path).
from narrative_arc.colors import ColorSpec, resolve_colors as _resolve_colors

class Paragraphs:
  """Single paragraphs, in the shape narrative_arc.colors.resolve_colors reads."""
  unit = "paragraph"

  def __init__(self, book, paragraphs):
    self.book = book
    self.chapters = np.array([p["chapter_id"] for p in paragraphs])
    try:
      self.emotions, self.scores = load_scores(book, paragraphs)
    except FileNotFoundError:
      self.emotions, self.scores = [], None

  def __len__(self):
    return len(self.chapters)

  def emotion(self, name):
    return self.scores[:, self.emotions.index(name)]


def resolve_colors(names, book, paragraphs):
  """Turn --color names into ColorSpecs (the arc package's resolver)."""
  return _resolve_colors(names, Paragraphs(book, paragraphs))


def draw_points(ax, coords, spec, size):
  """Draw the scatter; returns the mappable for the colorbar."""
  return ax.scatter(
    coords[:, 0], coords[:, 1], c=spec.values, cmap=spec.cmap,
    vmin=spec.vmin, vmax=spec.vmax, s=size, alpha=0.9, linewidths=0, zorder=2,
  )


def scatter_plot(coords, spec, title, xlabel, ylabel, output_path,
                 connect_story=False):
  """One full-size figure. Used by the single-view script."""
  fig, ax = plt.subplots(figsize=(8, 8))

  if connect_story:
    ax.plot(
      coords[:, 0], coords[:, 1], color="black", linewidth=0.5, alpha=0.3, zorder=1
    )

  sc = draw_points(ax, coords, spec, size=20)

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  fig.colorbar(sc, ax=ax, label=spec.label)

  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)
  print(f"  wrote {output_path}")


def panel(ax, coords, spec, title, xlabel, ylabel):
  """One cell of a sweep grid: same marks as `scatter_plot`, smaller."""
  sc = draw_points(ax, coords, spec, size=6)
  ax.set_title(title, fontsize=10)
  ax.set_xlabel(xlabel, fontsize=8)
  ax.set_ylabel(ylabel, fontsize=8)
  ax.tick_params(labelsize=6)
  return sc


def save_grid(fig, sc, suptitle, colorbar_label, output_path):
  fig.suptitle(suptitle, fontsize=14)
  fig.tight_layout(rect=[0, 0, 0.92, 0.97])
  cax = fig.add_axes([0.94, 0.08, 0.015, 0.84])
  fig.colorbar(sc, cax=cax, label=colorbar_label)
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


def load_book(book, model):
  """(paragraphs, embeddings), checked to be the same length.

  Row i of one is row i of the other, and a mismatch means the embeddings were
  built from a different `processed.json` than the one on disk -- which would
  silently mislabel every point rather than fail.
  """
  paragraphs = load_paragraphs(book)
  embeddings = load_embeddings(book, model)
  if len(embeddings) != len(paragraphs):
    raise ValueError(
      f"embeddings ({len(embeddings)}) and paragraphs ({len(paragraphs)}) mismatch."
    )
  return paragraphs, embeddings


def open_output(book, model, script, args, specs):
  """The `structure_2d` directory, migrated and stamped, ready to write into.

  No variant: every file here already carries its own parameters
  (`umap_n15_d0.1_s0`, `tsne_p30_lr200_s0`), so runs at different settings never
  collide. Figures then go one level deeper, into `pca/ umap/ tsne/`.

  The migration runs before any write, which is what keeps a directory from
  holding the flat and the nested layout at once -- the failure that got
  `emotions_3d`'s nested copies quarantined.
  """
  output_dir = paths.out_dir(book, model, paths.STRUCTURE_2D)
  moved = paths.migrate_projections(output_dir)
  if moved:
    print(f"  migrated {moved} file(s) into per-projection subdirectories")
  paths.stamp(output_dir, script, args, book=book, model=model,
              colorings=[s.name for s in specs])
  return output_dir


def add_common_args(parser):
  """The flags both scripts take, so their two `main`s stay short."""
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--model", type=str)
  parser.add_argument("--all-models", action="store_true")
  parser.add_argument("--color", nargs="+", default=["chapter", "paragraph"],
                      help="chapter, paragraph, emotions (a plot per emotion), "
                           "or individual emotion names.")
  parser.add_argument("--seed", default=0, type=int)
  return parser


def models_for(args, parser):
  """Which models to run: --all-models, or the one named."""
  if args.all_models:
    return sorted(os.listdir(os.path.join("books", args.book, "embeddings")))
  if args.model:
    return [args.model]
  parser.error("pass --model or --all-models")


def check_perplexity(perplexities, n, book):
  """t-SNE requires perplexity < n_samples; fail loudly rather than mid-run."""
  worst = max(perplexities)
  if worst >= n:
    raise ValueError(f"perplexity {worst} >= {n} paragraphs in {book}.")
