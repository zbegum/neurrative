"""
Loading helpers.

Row i of every array returned here is paragraph i in processed.json, which is
also row i of embeddings.npy -- so coordinates, chapters and scores always refer
to the same paragraph.
"""

import os
import json

import numpy as np

import paths


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
