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
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import paths
from data import UNCLEAR, dominant, load_paragraphs, load_scores

# name goes in the filename, label on the colorbar. vmin/vmax are pinned to
# (0, 1) for emotions so every emotion plot shares one scale and a color means
# the same score across books; None lets matplotlib autoscale. categories is
# None for continuous colors, else a list of (label, hue, marker, symbol)
# indexed by the integer codes in values.
ColorSpec = namedtuple("ColorSpec", "name values cmap label vmin vmax categories")


def dominant_spec(emotions, matrix, min_score):
  """One discrete color per paragraph: whichever emotion scores highest."""
  codes, categories, n_unclear = dominant(emotions, matrix, min_score)
  print(f"  dominant: {n_unclear} of {len(codes)} unclear "
        f"(top score < {min_score} or tied)")
  return ColorSpec("dominant", codes, None, "dominant emotion", None, None,
                   categories)


def resolve_colors(names, book, paragraphs, min_score):
  """Turn --color names into ColorSpecs, reading the scores only if needed."""
  emotions, matrix = None, None
  specs = []

  for name in names:
    if name == "chapter":
      specs.append(ColorSpec(
        "chapter",
        np.array([p["chapter_id"] for p in paragraphs]),
        "viridis", "chapter", None, None, None,
      ))
      continue

    if name == "paragraph":
      specs.append(ColorSpec(
        "paragraph", np.arange(len(paragraphs)),
        "plasma", "paragraph index", None, None, None,
      ))
      continue

    if emotions is None:
      emotions, matrix = load_scores(book, paragraphs)

    if name == "dominant":
      specs.append(dominant_spec(emotions, matrix, min_score))
      continue

    wanted = emotions if name == "emotions" else [name]
    for emotion in wanted:
      if emotion not in emotions:
        raise ValueError(
          f"Unknown --color {emotion!r}. Available: "
          f"chapter, paragraph, dominant, emotions, {', '.join(emotions)}"
        )
      specs.append(ColorSpec(
        emotion, matrix[:, emotions.index(emotion)],
        "viridis", emotion, 0.0, 1.0, None,
      ))

  # --color emotions wonder would draw wonder twice; keep the first of each.
  seen, unique = set(), []
  for spec in specs:
    if spec.name not in seen:
      seen.add(spec.name)
      unique.append(spec)
  return unique


def draw_points(ax, coords, spec, size):
  """Draw the scatter. Returns (mappable, legend handles) -- exactly one is set,
  since a continuous color wants a colorbar and a discrete one wants a legend.
  """
  if spec.categories is None:
    sc = ax.scatter(
      coords[:, 0], coords[:, 1], c=spec.values, cmap=spec.cmap,
      vmin=spec.vmin, vmax=spec.vmax, s=size, alpha=0.9, linewidths=0, zorder=2,
    )
    return sc, None

  handles = []
  for code, (label, hue, marker, _symbol) in enumerate(spec.categories):
    mask = spec.values == code
    if not mask.any():
      continue
    recessive = label == UNCLEAR
    handle = ax.scatter(
      coords[mask, 0], coords[mask, 1],
      c=hue, marker=marker, s=size * (0.7 if recessive else 1.0),
      alpha=0.4 if recessive else 0.9,
      # Yellow and aqua sit under 3:1 against white, so every mark carries a
      # thin dark edge to stay visible on the surface.
      linewidths=0 if recessive else 0.3,
      edgecolors="none" if recessive else "#33322e",
      zorder=1 if recessive else 2,
      label=f"{label} ({int(mask.sum())})",
    )
    handles.append(handle)
  return None, handles


def scatter_plot(coords, spec, title, xlabel, ylabel, output_path,
                 connect_story=False):
  """One full-size figure. Used by the single-view script."""
  fig, ax = plt.subplots(figsize=(8, 8))

  if connect_story:
    ax.plot(
      coords[:, 0], coords[:, 1], color="black", linewidth=0.5, alpha=0.3, zorder=1
    )

  sc, handles = draw_points(ax, coords, spec, size=20)

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  if handles:
    ax.legend(handles=handles, title=spec.label, loc="best",
              frameon=True, framealpha=0.9, fontsize=9, markerscale=1.6)
  else:
    fig.colorbar(sc, ax=ax, label=spec.label)

  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)
  print(f"  wrote {output_path}")


def panel(ax, coords, spec, title, xlabel, ylabel):
  """One cell of a sweep grid: same marks as `scatter_plot`, smaller."""
  sc, handles = draw_points(ax, coords, spec, size=6)
  ax.set_title(title, fontsize=10)
  ax.set_xlabel(xlabel, fontsize=8)
  ax.set_ylabel(ylabel, fontsize=8)
  ax.tick_params(labelsize=6)
  return sc, handles


def save_grid(fig, sc, handles, suptitle, colorbar_label, output_path):
  fig.suptitle(suptitle, fontsize=14)
  fig.tight_layout(rect=[0, 0, 0.92, 0.97])
  if handles:
    fig.legend(handles=handles, title=colorbar_label, loc="center right",
               frameon=False, fontsize=10, markerscale=2.2)
  else:
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
  embeddings = np.load(
    os.path.join("books", book, "embeddings", model, "embeddings.npy")
  )
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
                      help="chapter, paragraph, dominant (one discrete color per "
                           "emotion), emotions (a plot per emotion), or "
                           "individual emotion names.")
  parser.add_argument("--min-score", default=0.2, type=float,
                      help="For --color dominant: below this top score, a "
                           "paragraph is 'unclear' rather than colored.")
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
