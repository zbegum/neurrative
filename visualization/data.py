"""
Loading helpers and the shared emotion conventions.

Row i of every array returned here is paragraph i in processed.json, which is
also row i of embeddings.npy -- so coordinates, chapters and scores always refer
to the same paragraph.

The dominant-emotion palette lives here too, so that a color means the same
emotion in every plot either script draws.
"""

import os
import json

import numpy as np

import paths

# Fixed hue + marker per emotion for discrete "dominant emotion" plots. The hues
# are a 6-subset of the reference categorical palette, chosen because it is the
# only 6-subset clearing the all-pairs normal-vision floor (worst pair ΔE 15.6).
# Its worst colorblind pair is ΔE 6.9, which is only legal alongside a second,
# non-color channel -- hence the distinct marker per emotion. The mapping is by
# emotion name, never by how common the emotion is, so a color means the same
# thing in every book and every plot.
#                  hue        matplotlib  plotly
EMOTION_STYLE = {
  "sadness":   ("#2a78d6", "v", "diamond-open"),   # blue
  "curiosity": ("#008300", "o", "circle"),         # green
  "humor":     ("#eda100", "^", "square-open"),    # yellow
  "confusion": ("#1baf7a", "s", "square"),         # aqua
  "wonder":    ("#4a3aa7", "D", "diamond"),        # violet
  "danger":    ("#e34948", "X", "x"),              # red
}

# Anything whose top score is too low to mean much, or is an exact tie, gets a
# recessive gray rather than a confident color on a coin flip.
UNCLEAR = "unclear"
UNCLEAR_STYLE = ("#b0afa8", ".", "circle-open")


def dominant(emotions, matrix, min_score):
  """Which emotion wins each paragraph. Returns (codes, categories, n_unclear).

  codes index into categories, a list of (label, hue, marker, symbol) whose last
  entry is always `unclear`.

  argmax alone would be misleading twice over. Where every score is near zero
  nothing is really dominant, and on an exact tie argmax silently returns the
  lowest index -- which, since the emotions are sorted, would hand every tie to
  whichever sorts first. Both cases go to `unclear` instead.
  """
  missing = [e for e in emotions if e not in EMOTION_STYLE]
  if missing:
    raise ValueError(
      f"No color assigned for {missing}. EMOTION_STYLE covers "
      f"{sorted(EMOTION_STYLE)}; the palette is validated for 6 categories."
    )

  ordered = np.sort(matrix, axis=1)
  margin = ordered[:, -1] - ordered[:, -2]
  unclear = (matrix.max(axis=1) < min_score) | (margin == 0)
  codes = np.where(unclear, len(emotions), matrix.argmax(axis=1))

  categories = [(e, *EMOTION_STYLE[e]) for e in emotions]
  categories.append((UNCLEAR, *UNCLEAR_STYLE))
  return codes, categories, int(unclear.sum())


def load_paragraphs(book):
  book_dir = os.path.join(paths.ROOT, "books", book)
  with open(os.path.join(book_dir, "processed.json"), "r") as f:
    return json.load(f)["paragraphs"]


def load_scores(book, paragraphs):
  """Return (emotion names, score matrix) aligned to the paragraph order."""
  scores_file = os.path.join(paths.ROOT, "books", book, "paragraph_scores.json")
  if not os.path.exists(scores_file):
    raise FileNotFoundError(
      f"{scores_file} not found. Run preprocess/get_annotations.py --book {book} "
      f"first, or color by chapter/paragraph instead."
    )

  with open(scores_file, "r") as f:
    records = json.load(f)

  by_id = {r["paragraph_id"]: r["scores"] for r in records}

  missing = [p["id"] for p in paragraphs if p["id"] not in by_id]
  if missing:
    raise ValueError(
      f"{len(missing)} paragraph(s) have no scores, e.g. {missing[:3]}."
    )

  emotions = sorted(by_id[paragraphs[0]["id"]])
  matrix = np.array(
    [[by_id[p["id"]].get(e, np.nan) for e in emotions] for p in paragraphs]
  )
  return emotions, matrix


def load_characters(book, paragraphs):
  """Return (character names, boolean n x c membership matrix), row-aligned.

  Who appears in a paragraph is a second ground truth alongside the chapter, and
  a differently-shaped one: chapters are contiguous in reading order, so chapter
  purity cannot separate "the embedding knows about topic" from "the embedding
  knows the paragraphs were adjacent". A cast list is not contiguous -- the
  Duchess leaves and comes back -- so agreeing with it is evidence the layout
  tracks *content*.

  These names come from the same LLM pass as the scores and are not
  canonicalized: `she`, `he` and `speaker` are frequent "characters" in Pride and
  Prejudice, and one character may appear under several spellings. That noise is
  why the metric built on this is an excess over a shuffled control rather than a
  raw agreement -- whatever the extraction does uniformly across the book cancels.

  A paragraph with no listed characters keeps its (all-False) row, so indices stay
  aligned with the embeddings; see `character_agreement` for how empty rows count.
  """
  scores_file = os.path.join(paths.ROOT, "books", book, "paragraph_scores.json")
  if not os.path.exists(scores_file):
    raise FileNotFoundError(
      f"{scores_file} not found. Run preprocess/get_annotations.py --book {book} "
      f"first."
    )

  with open(scores_file, "r") as f:
    records = json.load(f)

  by_id = {r["paragraph_id"]: r.get("characters") or [] for r in records}
  missing = [p["id"] for p in paragraphs if p["id"] not in by_id]
  if missing:
    raise ValueError(
      f"{len(missing)} paragraph(s) have no annotation record, e.g. {missing[:3]}."
    )

  names = sorted({c for p in paragraphs for c in by_id[p["id"]]})
  index = {name: i for i, name in enumerate(names)}
  matrix = np.zeros((len(paragraphs), len(names)), dtype=bool)
  for row, p in enumerate(paragraphs):
    for c in by_id[p["id"]]:
      matrix[row, index[c]] = True
  return names, matrix
