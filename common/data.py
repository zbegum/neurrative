"""
Loading helpers.

Row i of every array returned here is paragraph i in processed.json, which is
also row i of embeddings.npy -- so coordinates, chapters and scores always refer
to the same paragraph.
"""

import os
import json
import sys

import numpy as np

import paths

# One implementation of the readers, in the arc package; these wrappers fix the
# data directory to the repository's books/.
sys.path.insert(1, os.path.join(paths.ROOT, "arc"))
from narrative_arc import data as _arc_data  # noqa: E402

DATA_DIR = os.path.join(paths.ROOT, "books")


def load_paragraphs(book):
  """The paragraphs of processed.json, in reading order."""
  return _arc_data.load_book(DATA_DIR, book)[0]


def load_embeddings(book, model):
  """(n_paragraphs, dim) embeddings; row i is paragraph i."""
  return _arc_data.load_embeddings(DATA_DIR, book, model)


def load_scores(book, paragraphs):
  """Return (emotion names, score matrix) aligned to the paragraph order."""
  return _arc_data.load_scores(DATA_DIR, book, paragraphs)


def _records(book):
  """The annotation records of paragraph_scores.json, keyed by paragraph id."""
  scores_file = os.path.join(DATA_DIR, book, "paragraph_scores.json")
  if not os.path.exists(scores_file):
    raise FileNotFoundError(
      f"{scores_file} not found. Run preprocess/get_annotations.py --book {book} "
      f"first."
    )
  with open(scores_file, "r") as f:
    return {r["paragraph_id"]: r for r in json.load(f)}


def load_summaries(book, paragraphs):
  """The annotator's one-line summary of each paragraph ('' where there is none)."""
  records = _records(book)
  return [records.get(p["id"], {}).get("summary", "") for p in paragraphs]


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
  by_id = {pid: r.get("characters") or [] for pid, r in _records(book).items()}
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
