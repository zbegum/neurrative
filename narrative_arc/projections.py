"""PCA, UMAP and t-SNE of the windowed series, in 2 or 3 components.

No single projection is trustworthy on its own, so the arc is always drawn by
all three and read for agreement:

  PCA    linear and parameter-free; distances and directions are real (within
         the variance the components capture), which is why it is the only one
         whose 3-D shape can be read as geometry.
  UMAP   preserves neighbourhoods; global distances are not meaningful.
  t-SNE  preserves neighbourhoods more aggressively still; cluster sizes and
         gaps between clusters are not meaningful.

UMAP/t-SNE carry their parameters in `suffix` so runs with different settings
never overwrite each other; PCA has none.
"""

import warnings
from collections import namedtuple

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

METHODS = ("pca", "umap", "tsne")

# name: display name. tag: folder/filename spelling. labels: one axis label per
# component. info: whatever is worth recording in params.json.
Projection = namedtuple("Projection", "name tag labels coords suffix info")


def project_pca(pooled, n_components):
  pca = PCA(n_components=n_components)
  coords = pca.fit_transform(pooled)
  var = pca.explained_variance_ratio_
  print(f"  PCA: explained variance {var.sum():.1%} over {n_components} components")
  labels = tuple(f"PC{i + 1} ({v:.1%} var)" for i, v in enumerate(var))
  return Projection("PCA", "pca", labels, coords, "",
                    {"explained_variance_ratio": var.round(4).tolist()})


def project_umap(pooled, n_components, neighbors, min_dist, seed):
  # Imported here: umap-learn takes seconds to import (numba), and a PCA-only
  # run should not pay for it.
  from umap import UMAP

  n_neighbors = min(neighbors, len(pooled) - 1)
  if n_neighbors != neighbors:
    print(f"  UMAP n_neighbors clamped to {n_neighbors} ({len(pooled)} windows)")
  print(f"  UMAP n_neighbors={n_neighbors} min_dist={min_dist}...")
  with warnings.catch_warnings():
    # A fixed random_state forces n_jobs=1; UMAP warns about it every run.
    warnings.filterwarnings("ignore", message=".*n_jobs.*overridden.*")
    coords = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                  min_dist=min_dist, random_state=seed).fit_transform(pooled)
  labels = tuple(f"UMAP-{i + 1}" for i in range(n_components))
  return Projection("UMAP", "umap", labels, coords,
                    f"_n{n_neighbors}_d{min_dist}_s{seed}",
                    {"n_neighbors": n_neighbors, "min_dist": min_dist,
                     "seed": seed})


def project_tsne(pooled, n_components, perplexity, seed):
  clamped = min(perplexity, len(pooled) - 1)
  if clamped != perplexity:
    print(f"  t-SNE perplexity clamped to {clamped} ({len(pooled)} windows)")
  print(f"  t-SNE perplexity={clamped}...")
  # Barnes-Hut (sklearn's default) handles up to 3 components, which is all
  # this ever asks for.
  coords = TSNE(n_components=n_components, perplexity=clamped,
                random_state=seed).fit_transform(pooled)
  labels = tuple(f"t-SNE-{i + 1}" for i in range(n_components))
  return Projection("t-SNE", "tsne", labels, coords, f"_p{clamped:g}_s{seed}",
                    {"perplexity": clamped, "seed": seed})


def project(pooled, methods=METHODS, n_components=2, neighbors=15,
            min_dist=0.1, perplexity=30.0, seed=0):
  """One Projection per requested method, in the order asked for."""
  if n_components not in (2, 3):
    raise ValueError(f"n_components must be 2 or 3 (got {n_components}).")

  out = []
  for method in methods:
    if method == "pca":
      out.append(project_pca(pooled, n_components))
    elif method == "umap":
      out.append(project_umap(pooled, n_components, neighbors, min_dist, seed))
    elif method == "tsne":
      out.append(project_tsne(pooled, n_components, perplexity, seed))
    else:
      raise ValueError(f"unknown projection {method!r}; choose from {METHODS}")
  return out
