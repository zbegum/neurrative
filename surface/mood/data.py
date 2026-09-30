"""Load a book's paragraphs, emotion scores, chapters and 2D coordinates."""

import glob
import importlib.util
import os

import numpy as np

import paths

ROOT = paths.ROOT

# common/data.py, loaded by file: this module is also called `data`.
_spec = importlib.util.spec_from_file_location(
    "neurrative_common_data", os.path.join(ROOT, "common", "data.py"))
common_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common_data)


def load_book(book):
  """Return (paragraphs, emotions, scores, chapters, summaries)."""
  paragraphs = common_data.load_paragraphs(book)
  emotions, scores = common_data.load_scores(book, paragraphs)
  chapters = np.array([p["chapter_id"] for p in paragraphs], dtype=int)
  return paragraphs, emotions, scores, chapters, common_data.load_summaries(book, paragraphs)


def load_coords(book, model, plane="pca", beta=2.0):
    """Return the 2D layout the surface is fitted over: pca or chain_umap."""
    if plane == "pca":
        base = paths.out_dir(book, model, paths.STRUCTURE_2D, create=False)
        for path in (os.path.join(base, "pca", "pca.npy"),
                     os.path.join(base, "pca.npy")):
            if os.path.exists(path):
                return np.load(path)[:, :2].astype(float)
        raise FileNotFoundError(f"no pca.npy under {base} -- run "
                                "projection/embedding.py first")

    if plane == "chain_umap":
        base = paths.out_dir(book, model, paths.CHAIN_UMAP, create=False)
        pattern = os.path.join(base, "*", f"chain_umap_b{beta:g}_*.npy")
        hits = sorted(glob.glob(pattern))
        if not hits:
            raise FileNotFoundError(f"no chain_umap coords for beta={beta:g} "
                                    f"under {base} -- run "
                                    "projection/chain_umap.py first")
        return np.load(hits[0])[:, :2].astype(float)

    raise ValueError(f"unknown plane {plane!r}")
