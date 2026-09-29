"""Load a book's paragraphs, emotion scores, chapters and 2D coordinates."""

import glob
import json
import os

import numpy as np

import paths

ROOT = paths.ROOT


def load_book(book):
    """Return (paragraphs, emotions, scores, chapters, summaries)."""
    with open(os.path.join(ROOT, "books", book, "processed.json")) as f:
        processed = json.load(f)
    paragraphs = processed["paragraphs"]
    chapters = np.zeros(len(paragraphs), dtype=int)
    for c in processed["chapters"]:
        a, b = c["paragraph_range"]
        chapters[a:b] = c["chapter_id"]

    with open(os.path.join(ROOT, "books", book, "paragraph_scores.json")) as f:
        records = json.load(f)
    by_id = {r["paragraph_id"]: r for r in records}

    emotions = sorted(by_id[paragraphs[0]["id"]]["scores"])
    scores = np.array([[by_id[p["id"]]["scores"][e] for e in emotions]
                       for p in paragraphs])
    summaries = [by_id[p["id"]].get("summary", "") for p in paragraphs]
    return paragraphs, emotions, scores, chapters, summaries


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
