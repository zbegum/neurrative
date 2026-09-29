"""
PCA vs UMAP vs t-SNE, side by side, over each method's own parameters.

The project had every other cell of this table but this one: `embedding_grid.py`
sweeps UMAP and t-SNE parameters but never puts the three methods next to each
other (and cannot include PCA, which has no parameters), while
`narrative_arc/_grid_all_projections_*` compares the three but sweeps the *arc's*
window size with the projections pinned. So the obvious question -- how do the
three differ, and how much does that difference depend on knobs nobody has a
principled way to set? -- had no figure.

## Columns are neighbourhood scale, not "setting 1, 2, 3"

UMAP's `n_neighbors` and t-SNE's `perplexity` are the same kind of parameter:
how many neighbours count as "local". Aligning the ladders makes the columns
mean one thing across rows -- local, default, global -- so reading down a column
compares methods at a comparable scale rather than at arbitrary settings.

PCA has no such knob; it is global by construction and deterministic. Its row is
drawn identically in all three columns and labelled as such. That is the point,
not filler: a row that does not move next to two rows that move a lot is the
honest picture of how much of a projection is the data and how much is the
setting.

## Four numbers per panel, because the eye is not a measurement

**Trustworthiness** (Venna & Kaski) asks whether the picture lies about
locality: of the k nearest neighbours a point has in the *plot*, how many were
really its neighbours in the full embedding? 1.0 is perfect, ~0.5 is chance.
A projection can look beautifully organized and score badly -- which is the
whole lesson of `arc_comparison`, restated for the static cloud.

**Chapter purity** asks whether the layout recovers something we independently
know is true: of a paragraph's k nearest neighbours, how many are from its own
chapter? Reported as an *excess* over a shuffled-label control on the identical
neighbourhoods, in the same spirit as the arc validation -- a raw purity is not
interpretable on its own, since some value falls out of any geometry. See
`separation()` for why this replaced silhouette, which the chapters' filament
shape defeats.

**Reading-order preservation** asks whether paragraphs that are neighbours in
the *book* are neighbours in the plot: of a paragraph's k nearest neighbours, how
many are within `ORDER_WINDOW` paragraphs of it? Reading order is the one label
that costs nothing and the one every arc script silently depends on -- if the
layout does not encode it, an arc through the cloud is a trajectory that happens
to be indexed by time rather than a path the story walks. Same shuffled-label
control, which here means permuting which paragraph sits at which position.

**Character agreement** asks whether the cast list survives the projection: mean
Jaccard overlap between a paragraph's characters and its k nearest neighbours'.
Chapters are contiguous in reading order, so chapter purity and order
preservation measure overlapping things -- a layout could score well on both by
tracking position alone. A cast is not contiguous, so this is the one label of
the three that can only be recovered from content. The shuffled control matters
most here: Alice is in 45% of her own book, so a raw overlap is mostly a base
rate. Skipped for books with no `paragraph_scores.json`.

All four are also computed on the raw high-dimensional embedding, which is the
reference the projections are approximating: structure *above* that line was
manufactured by the projection, below it was lost in flattening.

## Cached coordinates

Every setting on the ladder is one that `embedding_grid.py` already fit
and saved, for all nine book x model pairs. This script loads those `.npy` files
rather than refitting -- so the comparison is nearly free everywhere, including
the seven pairs whose grids were never rendered. `--refit` forces a fresh fit
(and writes the same filenames, so it stays compatible with
`embedding_grid.py`).

Example:

  python projection/projection_comparison.py --book alice_wonderland --model bge-m3
  python projection/projection_comparison.py --all
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_characters, load_paragraphs

# The ladder, aligned by what the parameter means rather than by its value.
# Every (n_neighbors, min_dist) and (perplexity, learning_rate) pair here is one
# `embedding.py` already sweeps, so the coordinates are on disk.
SCALES = ["local", "default", "global"]
UMAP_NEIGHBORS = [5, 15, 50]
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITIES = [5.0, 30.0, 100.0]
TSNE_LEARNING_RATE = 200.0

# How many shuffles of the labels define the "no structure" floor. The spread
# across shuffles is tiny (silhouette of random labels is stable), so a handful
# is enough to place the line without pretending to a tight CI.
N_SHUFFLES = 10

# What counts as "adjacent in the book" for reading-order preservation. Five is
# roughly a beat of a scene: wide enough that a layout does not have to reproduce
# the exact sequence to score, narrow enough that landing in the right chapter is
# not sufficient. The shuffled floor is ~2*W/n, well under a percent here, so the
# excess and the raw number are nearly the same -- the control is reported anyway
# so the number is read the same way as the other two.
ORDER_WINDOW = 5

# The cast panel names a few characters and grays the rest. Four, because this is
# a scatter -- every pair of colors can end up adjacent, and on the all-pairs
# gate the reference categorical palette only clears its floors for four slots.
# Each also gets its own marker: the worst all-pairs CVD separation here is tritan
# dE 5.8, which is legal only alongside a channel that is not color. Same
# reasoning, and the same palette, as EMOTION_STYLE in data.py.
CAST_STYLE = [("#2a78d6", "o"), ("#008300", "s"),
              ("#e87ba4", "D"), ("#eda100", "^")]
CAST_OTHER = ("#b0afa8", ".")

# A character in more than this share of the book is skipped when choosing which
# names to draw. Alice is in 45% of Alice: coloring her would cover the panel in
# one hue and say nothing, because a character who is everywhere cannot tell you
# where you are. The panel is asking which *scene* a region is, and that is
# carried by the characters who come and go.
UBIQUITY = 0.25


def cached(out_dir, name, fit, refit):
  """Load `name`.npy from the structure_2d directory, or fit and save it.

  `embedding.py` writes these same filenames, so a cache hit here is literally
  the coordinates that produced the published grids -- not a re-fit that happens
  to use the same settings. UMAP and t-SNE are seeded but not bitwise stable
  across library versions, so reusing beats refitting for comparability too.
  """
  found = None if refit else paths.find_coords(out_dir, f"{name}.npy")
  if found is not None:
    print(f"  cached  {name}")
    return np.load(found)

  print(f"  fitting {name} ...")
  coords = fit()
  # A fresh fit is written in the layout embedding.py now writes: one folder per
  # projection. Reads still fall back to flat, writes never add to it.
  projection = paths.projection_of(f"{name}.npy")
  np.save(os.path.join(paths.projection_dir(out_dir, projection), f"{name}.npy"),
          coords)
  return coords


def panels(embeddings, out_dir, seed, refit):
  """The 3x3 of (method, scale) -> (title, coords).

  PCA occupies all three columns with one set of coordinates: it has nothing to
  sweep, and drawing it three times is what makes the row read as flat.
  """
  n = len(embeddings)

  # Seeded: at this shape sklearn's "auto" solver is randomized SVD, so an
  # unseeded PCA is not reproducible run to run. See embedding.run_pca.
  pca = cached(out_dir, "pca",
               lambda: PCA(n_components=2,
                           random_state=seed).fit_transform(embeddings), refit)
  rows = [("PCA", [("deterministic, no parameters", pca)] * len(SCALES))]

  umap_row = []
  for k in UMAP_NEIGHBORS:
    name = f"umap_n{k}_d{UMAP_MIN_DIST}_s{seed}"
    coords = cached(out_dir, name, lambda k=k: UMAP(
      n_components=2, n_neighbors=k, min_dist=UMAP_MIN_DIST,
      random_state=seed).fit_transform(embeddings), refit)
    umap_row.append((f"n_neighbors = {k}", coords))
  rows.append(("UMAP", umap_row))

  tsne_row = []
  for p in TSNE_PERPLEXITIES:
    # t-SNE requires perplexity < n_samples; a short book can fail the top rung.
    if p >= n:
      print(f"  skipping t-SNE perplexity={p}: only {n} paragraphs")
      tsne_row.append((f"perplexity = {p:g} (n/a, n={n})", None))
      continue
    name = f"tsne_p{p}_lr{TSNE_LEARNING_RATE}_s{seed}"
    coords = cached(out_dir, name, lambda p=p: TSNE(
      n_components=2, perplexity=p, learning_rate=TSNE_LEARNING_RATE,
      random_state=seed).fit_transform(embeddings), refit)
    tsne_row.append((f"perplexity = {p:g}", coords))
  rows.append(("t-SNE", tsne_row))

  return rows


def neighbourhoods(coords, k):
  """The k nearest neighbours of every point, itself excluded.

  Called without an argument, kneighbors() drops each point from its own
  neighbourhood -- otherwise every point would be trivially 1/k pure. Every
  label-based metric below shares this one index array, so they all describe the
  same neighbourhoods and differ only in which label they ask about.
  """
  return NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(
    return_distance=False)


def order_preservation(idx, rng):
  """Do paragraphs that are neighbours in the book stay neighbours in the plot?

  Chapter purity already rewards a layout for keeping a chapter together, and a
  chapter is a run of consecutive paragraphs, so the two questions overlap. This
  one is the sharper half: not "same chapter" but "within a few paragraphs",
  which a layout that merely clusters by topic will fail.

  The control permutes which paragraph sits at which position in the book, over
  the same neighbourhoods -- so the floor is what this exact geometry scores when
  reading order carries no information, matching the construction in
  `separation`. That floor is small by construction (~2*ORDER_WINDOW/n), so the
  excess barely differs from the raw share; it is reported for consistency, not
  because it moves the number.

  `median_gap` has no control and needs none: it is the descriptive companion,
  the typical distance in paragraphs between a point and the neighbours the plot
  gives it. Read it against the length of a chapter.
  """
  n = idx.shape[0]
  order = np.arange(n)
  gap = np.abs(idx - order[:, None])

  real = float(np.mean(gap <= ORDER_WINDOW))
  shuffled = float(np.mean([
    np.mean(np.abs((s := rng.permutation(order))[idx] - s[:, None])
            <= ORDER_WINDOW)
    for _ in range(N_SHUFFLES)
  ]))
  return {
    "order_adjacency": real,
    "order_adjacency_shuffled": shuffled,
    "order_adjacency_excess": real - shuffled,
    "order_median_gap": float(np.median(gap)),
  }


def character_agreement(idx, cast, rng):
  """How much of a paragraph's cast its plotted neighbours share.

  Jaccard rather than "shares at least one character", because Alice is in 45% of
  Alice and any all-or-nothing test would be answered by her alone. Jaccard still
  feels her -- but the shuffled control subtracts exactly the part of the overlap
  that a fixed cast distribution produces regardless of geometry, which is the
  point of reporting an excess. It also absorbs the extraction's noise (`she`,
  `speaker`, one character under two spellings) as long as that noise is spread
  evenly over the book.

  Paragraphs with no listed characters keep their row and score 0 overlap with
  everything, and the permutation moves those empty rows around too, so they
  raise neither the real number nor the floor unfairly. Coverage is reported
  alongside so a book whose extraction mostly failed is visible rather than
  silently scoring low.
  """
  def mean_jaccard(rows):
    inter = np.einsum("ij,ikj->ik", rows.astype(np.int16),
                      rows[idx].astype(np.int16))
    union = rows.sum(1)[:, None] + rows[idx].sum(2) - inter
    return float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 0.0)))

  real = mean_jaccard(cast)
  shuffled = float(np.mean([
    mean_jaccard(rng.permutation(cast, axis=0)) for _ in range(N_SHUFFLES)
  ]))
  return {
    "character_jaccard": real,
    "character_jaccard_shuffled": shuffled,
    "character_jaccard_excess": real - shuffled,
    "character_coverage": float(np.mean(cast.any(axis=1))),
  }


def separation(coords, idx, labels, rng):
  """How much of a paragraph's neighbourhood is its own chapter.

  Silhouette is the obvious choice and the wrong one here: it rewards compact
  convex blobs, and a chapter is not one. A chapter is a filament -- Alice walks
  through it, so its paragraphs trail across the cloud and often split into
  several scenes. Silhouette scores every projection *negative* on these labels,
  including the raw 1024-d embedding, which says more about the statistic than
  about the books.

  k-NN purity asks the local question instead: of the k nearest paragraphs, how
  many come from the same chapter? It does not care what shape the chapter is.
  The control shuffles the labels over the *same* neighbourhoods, so the floor is
  what this exact geometry scores when chapter identity is noise -- the same
  construction `order_preservation` and `character_agreement` use for their own
  labels, and the one the arc validation uses for reading order.

  Silhouette is still reported alongside it, so the disagreement stays visible
  rather than being quietly resolved by picking the flattering number.
  """
  real = float(np.mean(labels[idx] == labels[:, None]))
  shuffled = float(np.mean([
    np.mean((s := rng.permutation(labels))[idx] == s[:, None])
    for _ in range(N_SHUFFLES)
  ]))
  return {
    "chapter_purity": real,
    "chapter_purity_shuffled": shuffled,
    "chapter_purity_excess": real - shuffled,
    "silhouette": float(silhouette_score(coords, labels)),
  }


def label_metrics(coords, labels, cast, k, rng):
  """Everything that asks a label about a layout, over one shared k-NN graph.

  Applied to a projection's coordinates this scores the projection; applied to
  the raw embedding it produces the reference the projections are approximating.
  Same function either way, so the two are comparable by construction.
  """
  idx = neighbourhoods(coords, k)
  metrics = {**separation(coords, idx, labels, rng), **order_preservation(idx, rng)}
  if cast is not None:
    metrics.update(character_agreement(idx, cast, rng))
  return metrics


def score(embeddings, coords, labels, cast, k, rng):
  return {
    "trustworthiness": float(trustworthiness(embeddings, coords, n_neighbors=k)),
    **label_metrics(coords, labels, cast, k, rng),
  }


def grid_figure(rows, metrics, labels, book, model, k, path):
  fig, axes = plt.subplots(3, len(SCALES), figsize=(15, 15.6))

  for i, (method, cells) in enumerate(rows):
    for j, (title, coords) in enumerate(cells):
      ax = axes[i][j]
      if coords is None:
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=11,
                color="0.5", transform=ax.transAxes)
        ax.set_axis_off()
        continue

      ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis",
                 s=9, linewidths=0)
      m = metrics[method][j]
      cast = (f"   cast {m['character_jaccard']:.3f} "
              f"({m['character_jaccard_excess']:+.3f})"
              if "character_jaccard" in m else "")
      ax.set_title(f"{title}\nT = {m['trustworthiness']:.3f}   "
                   f"chapter purity = {m['chapter_purity']:.3f} "
                   f"({m['chapter_purity_excess']:+.3f})\n"
                   f"order {m['order_adjacency']:.3f} "
                   f"(median gap {m['order_median_gap']:.0f}){cast}",
                   fontsize=9)
      ax.set_xticks([])
      ax.set_yticks([])

    axes[i][0].set_ylabel(method, fontsize=15, labelpad=12)

  for j, scale in enumerate(SCALES):
    axes[0][j].text(0.5, 1.16, scale, ha="center", va="bottom", fontsize=13,
                    transform=axes[0][j].transAxes)

  fig.suptitle(
    f"{book} / {model} -- PCA vs UMAP vs t-SNE across neighbourhood scale\n"
    f"colored by chapter\n"
    f"T = trustworthiness (k={k}): are the plot's neighbours the embedding's?\n"
    f"chapter purity: share of the k nearest paragraphs from the same chapter; "
    f"order: share within {ORDER_WINDOW} paragraphs in the book;\n"
    f"cast: mean Jaccard overlap of characters "
    f"(each in parentheses: excess over a shuffled-label control)",
    fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(path, dpi=200)
  plt.close(fig)
  print(f"  wrote {path}")


def leading_cast(cast, names):
  """Which characters the cast panel names, and which one each paragraph gets.

  Returns (column indices in frequency order, per-paragraph code) where the code
  indexes into that list and equals len(list) for "none of them".

  A paragraph often has several characters, and the panel has one color per
  point, so something has to break the tie. It goes to the *rarest* of the named
  characters present: the Mock Turtle appearing tells you where you are in a way
  that the King also appearing does not. That is why the loop assigns commonest
  first and lets rarer ones overwrite.
  """
  share = cast.mean(axis=0)
  eligible = [c for c in np.argsort(-share) if share[c] <= UBIQUITY]
  chosen = eligible[:len(CAST_STYLE)]

  code = np.full(cast.shape[0], len(chosen), dtype=int)
  for rank, c in enumerate(chosen):
    code[cast[:, c]] = rank
  return chosen, code


def label_views_figure(rows, metrics, labels, cast, names, book, model, path):
  """One layout per row, recolored by each label the metrics ask about.

  The numbers say the layout tracks the cast far better than it tracks position
  in the book. This is that sentence as a picture: the same points three times,
  and the eye can check whether the colors agree with the geometry.

  Only the `default` column of the parameter ladder is drawn. The ladder's job is
  in `grid_figure`; repeating it here would triple the panels to re-answer a
  question already answered, and the metrics move little across it anyway.

  The reading-order panel draws the book as a path rather than as a color,
  because that is the claim being tested: if consecutive paragraphs were
  neighbours the line would crawl, and instead it leaps. `order_median_gap` in
  the title is the same fact as a number.
  """
  j = SCALES.index("default")
  drawn = [(method, cells[j], metrics[method][j]) for method, cells in rows]
  drawn = [(m, t, c, s) for m, (t, c), s in drawn if c is not None]

  ncols = 2 if cast is None else 3
  fig, axes = plt.subplots(len(drawn), ncols, figsize=(5.2 * ncols, 5.6 * len(drawn)),
                           squeeze=False)

  if cast is not None:
    chosen, code = leading_cast(cast, names)

  for row, (method, title, coords, m) in enumerate(drawn):
    ax = axes[row][0]
    dots = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis",
                      s=9, linewidths=0)
    ax.set_title(f"by chapter\npurity {m['chapter_purity']:.3f} "
                 f"({m['chapter_purity_excess']:+.3f} over shuffled)", fontsize=10)
    fig.colorbar(dots, ax=ax, fraction=0.046, pad=0.02, label="chapter")

    # The book as a path. Points recede to a wash so the leaps are the figure;
    # segments are colored by how far through the book they are, which is the
    # only way to tell an early tangle from a late one.
    ax = axes[row][ncols - 1]
    ax.scatter(coords[:, 0], coords[:, 1], color="0.85", s=6, linewidths=0)
    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    ax.add_collection(LineCollection(segments, cmap="viridis", linewidths=0.6,
                                     alpha=0.55,
                                     array=np.arange(len(segments))))
    ax.autoscale_view()
    ax.set_title(f"by reading order (line = the book, in order)\n"
                 f"{m['order_adjacency']:.3f} of neighbours within "
                 f"{ORDER_WINDOW} paragraphs, median gap "
                 f"{m['order_median_gap']:.0f}", fontsize=10)

    if cast is not None:
      ax = axes[row][1]
      other_color, other_marker = CAST_OTHER
      rest = code == len(chosen)
      ax.scatter(coords[rest, 0], coords[rest, 1], color=other_color,
                 marker=other_marker, s=7, linewidths=0,
                 label=f"other / none ({rest.sum()})")
      for rank, c in enumerate(chosen):
        color, marker = CAST_STYLE[rank]
        hit = code == rank
        ax.scatter(coords[hit, 0], coords[hit, 1], color=color, marker=marker,
                   s=20, linewidths=0, label=f"{names[c]} ({hit.sum()})")
      ax.legend(frameon=False, fontsize=8, loc="best", markerscale=1.4)
      ax.set_title(f"by character\nmean cast overlap "
                   f"{m['character_jaccard']:.3f} "
                   f"({m['character_jaccard_excess']:+.3f} over shuffled)",
                   fontsize=10)

    for ax in axes[row]:
      ax.set_xticks([])
      ax.set_yticks([])
    axes[row][0].set_ylabel(f"{method} -- {title}", fontsize=12, labelpad=10)

  fig.suptitle(
    f"{book} / {model} -- the same layout, colored by each label\n"
    f"every panel in a row is identical geometry; only the coloring changes,\n"
    f"so a row shows which of the three the projection actually kept",
    fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  fig.savefig(path, dpi=200)
  plt.close(fig)
  print(f"  wrote {path}")


def metrics_figure(rows, metrics, reference, book, model, k, path):
  """The same comparison as numbers, which is the part worth citing.

  Three of the four panels are excesses over a shuffled control and carry the raw
  embedding's own excess as a dashed line, so a bar is read as a fraction of the
  structure that was there to keep. Trustworthiness has no such line: it is
  defined against the raw embedding, which scores 1.0 by construction.
  """
  # (key, title, y label). Only the label-based three have a reference line.
  panels = [
    (None, f"Trustworthiness (k={k}) -- does the plot lie about locality?",
     "fraction of plotted neighbours that are real neighbours"),
    ("chapter_purity_excess", "Chapter purity above a shuffled-label control",
     "excess share of same-chapter neighbours"),
    ("order_adjacency_excess",
     f"Reading order: neighbours within {ORDER_WINDOW} paragraphs",
     "excess share of book-adjacent neighbours"),
    ("character_jaccard_excess", "Character overlap above a shuffled control",
     "excess mean Jaccard of casts"),
  ]
  if "character_jaccard_excess" not in reference:
    panels = panels[:3]

  ncols = 2 if len(panels) > 2 else len(panels)
  nrows = -(-len(panels) // ncols)
  fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
  flat = axes.ravel()
  width = 0.26
  x = np.arange(len(SCALES))

  for ax, (key, title, ylabel) in zip(flat, panels):
    for i, (method, cells) in enumerate(rows):
      values = [metrics[method][j]["trustworthiness" if key is None else key]
                if c is not None else np.nan for j, (_, c) in enumerate(cells)]
      ax.bar(x + (i - 1) * width, values, width, label=method)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if key is None:
      ax.set_ylim(0.5, 1.0)
      ax.axhline(0.5, color="0.4", lw=1, ls=":")
      # Left-aligned: the legend sits bottom-right in every panel.
      ax.text(0.02, 0.505, "chance", fontsize=8, color="0.4", ha="left",
              va="bottom", transform=ax.get_yaxis_transform())
    else:
      ax.axhline(0, color="0.3", lw=1)
      ax.axhline(reference[key], color="crimson", lw=1.2, ls="--")
      ax.text(0.02, reference[key],
              " raw embedding (the thing being approximated)",
              fontsize=9, color="crimson", va="bottom",
              transform=ax.get_yaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(UMAP k={n}, t-SNE p={p:g})"
                        for s, n, p in zip(SCALES, UMAP_NEIGHBORS,
                                           TSNE_PERPLEXITIES)], fontsize=9)
    ax.legend(frameon=False, loc="lower right")

  for ax in flat[len(panels):]:
    ax.set_axis_off()

  fig.suptitle(f"{book} / {model} -- projection comparison", fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(path, dpi=200)
  plt.close(fig)
  print(f"  wrote {path}")


def run(book, model, args):
  print(f"\n=== {book} / {model} ===")
  embeddings = np.load(
    os.path.join("books", book, "embeddings", model, "embeddings.npy"))
  paragraphs = load_paragraphs(book)
  labels = np.array([p["chapter_id"] for p in paragraphs])
  print(f"  {len(embeddings)} paragraphs, {embeddings.shape[1]}-d, "
        f"{len(set(labels))} chapters")

  # Characters come from the LLM annotation pass, which not every book has had.
  # A missing file drops one metric rather than the run: chapter and reading
  # order still work on anything with a processed.json.
  try:
    names, cast = load_characters(book, paragraphs)
    print(f"  {len(names)} distinct characters, "
          f"{np.mean(cast.any(axis=1)):.2f} of paragraphs annotated")
  except FileNotFoundError as e:
    print(f"  no character labels, skipping character agreement -- {e}")
    names, cast = [], None

  # The projections are read from and written to structure_2d, where
  # embedding.py keeps them; only the comparison figures get their own family.
  coords_dir = paths.out_dir(book, model, paths.STRUCTURE_2D)
  out_dir = paths.out_dir(book, model, paths.PROJECTION_COMPARISON,
                          variant_params={"k": args.trust_k, "s": args.seed})

  rows = panels(embeddings, coords_dir, args.seed, args.refit)

  rng = np.random.default_rng(args.seed)
  metrics = {method: [None if c is None else score(embeddings, c, labels, cast,
                                                   args.trust_k, rng)
                      for _, c in cells]
             for method, cells in rows}

  # The raw embedding is the thing being approximated: how much of each label
  # does it carry before any projection touches it?
  reference = label_metrics(embeddings, labels, cast, args.trust_k, rng)
  print(f"  raw {embeddings.shape[1]}-d chapter purity: "
        f"{reference['chapter_purity']:.4f} "
        f"({reference['chapter_purity_excess']:+.4f} over shuffled)")
  print(f"  raw reading order: {reference['order_adjacency']:.4f} within "
        f"{ORDER_WINDOW} paragraphs, median gap "
        f"{reference['order_median_gap']:.0f}")
  if cast is not None:
    print(f"  raw character Jaccard: {reference['character_jaccard']:.4f} "
          f"({reference['character_jaccard_excess']:+.4f} over shuffled)")

  grid_figure(rows, metrics, labels, book, model, args.trust_k,
              os.path.join(out_dir, "projection_grid.png"))
  label_views_figure(rows, metrics, labels, cast, names, book, model,
                     os.path.join(out_dir, "label_views.png"))
  metrics_figure(rows, metrics, reference, book, model, args.trust_k,
                 os.path.join(out_dir, "projection_metrics.png"))

  payload = {
    "book": book, "model": model,
    "n_paragraphs": int(len(embeddings)),
    "embedding_dim": int(embeddings.shape[1]),
    "n_chapters": int(len(set(labels))),
    "n_characters": len(names),
    "trust_k": args.trust_k,
    "n_shuffles": N_SHUFFLES,
    "order_window": ORDER_WINDOW,
    "raw_embedding": reference,
    "methods": {
      method: [{"setting": title, "scale": SCALES[j], **(metrics[method][j] or {})}
               for j, (title, _) in enumerate(cells)]
      for method, cells in rows
    },
  }
  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
  print(f"  wrote {os.path.join(out_dir, 'metrics.json')}")

  paths.stamp(out_dir, __file__, args, book=book, model=model,
              ladder={"umap_n_neighbors": UMAP_NEIGHBORS,
                      "umap_min_dist": UMAP_MIN_DIST,
                      "tsne_perplexity": TSNE_PERPLEXITIES,
                      "tsne_learning_rate": TSNE_LEARNING_RATE})
  return payload


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", default="alice_wonderland")
  parser.add_argument("--model", default="bge-m3")
  parser.add_argument("--all", action="store_true",
                      help="Every book x model pair that has embeddings on disk.")
  parser.add_argument("--trust-k", default=15, type=int,
                      help="Neighbourhood size for trustworthiness. Independent "
                           "of the projections' own neighbourhood parameters.")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--refit", action="store_true",
                      help="Refit the projections instead of loading the "
                           "coordinates embedding.py already saved.")
  args = parser.parse_args()

  if not args.all:
    run(args.book, args.model, args)
    return

  for book in sorted(os.listdir("books")):
    emb = os.path.join("books", book, "embeddings")
    if not os.path.isdir(emb):
      continue
    for model in sorted(os.listdir(emb)):
      if os.path.exists(os.path.join(emb, model, "embeddings.npy")):
        run(book, model, args)


if __name__ == "__main__":
  main()
