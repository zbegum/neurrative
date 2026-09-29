"""
Loading helpers.

A book lives under a data directory laid out as

    <data-dir>/<book>/processed.json                    paragraphs, in reading order
    <data-dir>/<book>/paragraph_scores.json             emotion scores (optional)
    <data-dir>/<book>/embeddings/<model>/embeddings.npy one row per paragraph

Row i of every array returned here is paragraph i in processed.json, which is
also row i of embeddings.npy -- so coordinates, chapters and scores always refer
to the same paragraph.
"""

import os
import json

import numpy as np


def list_books(data_dir):
  """Every book under data_dir that has at least one embedding model."""
  return sorted(
    b for b in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, b, "embeddings"))
  )


def list_models(data_dir, book):
  embeddings_dir = os.path.join(data_dir, book, "embeddings")
  return sorted(
    m for m in os.listdir(embeddings_dir)
    if os.path.exists(os.path.join(embeddings_dir, m, "embeddings.npy"))
  )


def load_book(data_dir, book):
  """Return (paragraphs, chapters) from processed.json."""
  with open(os.path.join(data_dir, book, "processed.json"), "r") as f:
    processed = json.load(f)
  return processed["paragraphs"], processed.get("chapters", [])


def load_embeddings(data_dir, book, model):
  path = os.path.join(data_dir, book, "embeddings", model, "embeddings.npy")
  if not os.path.exists(path):
    raise FileNotFoundError(
      f"{path} not found. Available models for {book}: "
      f"{', '.join(list_models(data_dir, book)) or 'none'}."
    )
  return np.load(path)


def load_scores(data_dir, book, paragraphs):
  """Return (emotion names, score matrix) aligned to the paragraph order."""
  scores_file = os.path.join(data_dir, book, "paragraph_scores.json")
  if not os.path.exists(scores_file):
    raise FileNotFoundError(
      f"{scores_file} not found; color by progression or chapter instead."
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
