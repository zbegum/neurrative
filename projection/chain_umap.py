"""
UMAP over a kNN graph with the book's reading order added as edges.

`projection_comparison.py` established that reading order is the label every arc
script silently depends on and the one no projection is asked to preserve: PCA,
UMAP and t-SNE all see a bag of 789 vectors and never learn that paragraph 400
is followed by 401. A storyline drawn through such a layout is a trajectory that
happens to be indexed by time rather than a path the story walks.

This script tells the projection. UMAP does not embed points, it embeds a fuzzy
graph over them, and that graph is an object we can add to before the layout
runs -- so reading order enters as *structure* rather than as a coordinate
bolted on to the features:

    G(beta) = G_semantic  t-conorm  beta * G_chain

`G_semantic` is exactly the graph a plain UMAP builds. `G_chain` connects each
paragraph to the CHAIN_WIDTH that follow it, weighted 1/d. The combination is
the probabilistic t-conorm `a + b - ab`, which is the same operation UMAP uses
internally to symmetrize its own graph -- so it keeps every weight in [0, 1],
and at beta = 0 it returns `G_semantic` unchanged. That is the property that
makes the sweep readable: the first row is plain UMAP, not an approximation of
it, so anything that moves across the row was bought by the chain edges.

## beta is a knob nobody can set from first principles, so it is swept

There is no correct amount of reading order. Too little and the layout is the
one we already had; too much and the cloud collapses to a ribbon that says
paragraph 1 comes before paragraph 2 -- true, and free, and not worth a figure.
The useful output is therefore not a layout but a **trade-off curve**:
trustworthiness (does the plot lie about locality in the embedding?) against
reading-order preservation, with one point per beta.

Both metrics, and the shuffled-label controls under them, are imported from
`projection_comparison` rather than reimplemented, so a number here is directly
comparable to the number under the same name there. `ORDER_WINDOW`, `N_SHUFFLES`
and the trustworthiness k are all that file's.

## What to read off the result

A beta whose order preservation rises while trustworthiness holds flat is the
honest win: the chain edges resolved an ambiguity the semantic graph left open,
by choosing among layouts the embedding was indifferent between. A beta where
both move together is a trade being made, and the curve is where you decide how
much of one you will pay for the other. Chapter purity is reported alongside as
a check that the gain is not purely positional -- chapters are runs of
consecutive paragraphs, so a layout can score on order by tracking position
alone, and character agreement (content, non-contiguous) is the label that
cannot be faked that way.

Output lands in projection/output/<book>/<model>/chain_umap/<variant>/.

Example:

python projection/chain_umap.py --book alice_wonderland --model bge-m3
python projection/chain_umap.py --book alice_wonderland --model bge-m3 \
  --beta 0 0.25 1.0 --chain-width 4
"""

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from umap.umap_ import (find_ab_params, fuzzy_simplicial_set,
                        simplicial_set_embedding)

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import paths
from data import load_characters, load_paragraphs
from embedding_common import (add_common_args, load_book, models_for, panel,
                              resolve_colors, save_grid)
from projection_comparison import ORDER_WINDOW, label_metrics, score

# The ladder. 0 is plain UMAP and is always run first, because every other row
# is only interpretable as a distance from it. The rest are spaced
# multiplicatively rather than evenly: the interesting behaviour is at the low
# end, where the chain is a nudge, and by 1.0 an adjacent pair is already as
# strongly bound as a mutual nearest neighbour in the semantic graph.
#
# 2.0 is the far endpoint and not an arbitrary one: `blend` clamps the scaled
# chain to 1.0, so at beta * (1/CHAIN_WIDTH) >= 1 every chain edge is saturated
# and larger betas cannot change the graph. Measured on alice/bge-m3, beta = 4
# reproduces beta = 2 to the digit. The curve therefore has a right end rather
# than trailing off wherever the ladder happened to stop.
CHAIN_BETAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

# How many following paragraphs each one is chained to. 2, not 1: a single
# forward edge makes a path graph, whose spectral layout is dominated by its own
# length and which one missing edge (a chapter break the embedding disagrees
# about) severs. Two gives the chain a little rigidity without turning it into a
# window -- and it stays well under ORDER_WINDOW = 5, so the metric is not being
# handed the answer it is asked to check.
CHAIN_WIDTH = 2

# UMAP's own defaults, held fixed. This script sweeps beta; sweeping n_neighbors
# at the same time would make the grid a 2-D table where every cell differs in
# two ways. `embedding_grid.py` is where n_neighbors is swept.
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "cosine"

# UMAP's defaults for the layout optimizer, spelled out because
# `simplicial_set_embedding` -- unlike the UMAP class -- takes no defaults and
# would otherwise have them buried in a call site.
SPREAD = 1.0
INITIAL_ALPHA = 1.0
GAMMA = 1.0
NEGATIVE_SAMPLE_RATE = 5
N_EPOCHS = 500  # UMAP's choice for n <= 10000; every book here is far under.


def semantic_graph(embeddings, n_neighbors, seed, metric):
  """The fuzzy simplicial set a plain UMAP would build over these embeddings.

  This is UMAP's own `fuzzy_simplicial_set`, called with the same arguments
  `UMAP.fit` passes it, so the beta = 0 row of the sweep is plain UMAP rather
  than a reimplementation that resembles it. Weights are in [0, 1] and the
  matrix is already symmetrized by the set operations UMAP applies inside.
  """
  graph, _sigmas, _rhos = fuzzy_simplicial_set(
    embeddings, n_neighbors, check_random_state(seed), metric,
  )
  return graph.tocsr()


def chain_graph(n, width):
  """Reading order as a weighted graph: i -- i+d for d = 1..width, weight 1/d.

  1/d rather than a constant, so the chain does not assert that a paragraph two
  away is as close as the next one -- it is the same shape as the decay any
  reasonable kernel over reading position would give, without introducing a
  bandwidth to tune next to beta.

  Symmetrized explicitly. Reading order is directed and the story is not
  reversible, but a UMAP graph is undirected: the edge means "these two belong
  near each other", which is symmetric even when the reading is not.
  """
  rows, cols, vals = [], [], []
  for d in range(1, width + 1):
    i = np.arange(n - d)
    rows.append(i)
    cols.append(i + d)
    vals.append(np.full(n - d, 1.0 / d))

  upper = sp.coo_matrix(
    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
    shape=(n, n),
  ).tocsr()
  return upper + upper.T


def blend(semantic, chain, beta):
  """`semantic` t-conorm `beta * chain`: a + b - ab, elementwise on sparse.

  The probabilistic t-conorm is the operation UMAP itself uses to merge the two
  directed views of its own graph, so this stays inside the algorithm's own
  algebra rather than inventing a combination rule. It is what keeps the result
  a fuzzy set -- weights stay in [0, 1] however large beta gets, so a big beta
  saturates the chain edges rather than letting them outweigh everything by
  arithmetic. And b = 0 gives back a exactly, which is why beta = 0 is plain
  UMAP and not merely close to it.

  The clamp is what bounds the sweep: once beta reaches CHAIN_WIDTH every chain
  weight is pinned at 1 and the graph stops responding, so the ladder has a
  genuine far end. See CHAIN_BETAS.
  """
  if beta == 0:
    return semantic.copy()
  b = (chain * beta).minimum(1.0)
  return (semantic + b - semantic.multiply(b)).tocsr()


def embed(embeddings, graph, seed, min_dist, metric):
  """Run UMAP's layout optimizer on a graph we built ourselves.

  The UMAP class does not expose a fitted graph as an input, so the two halves
  of it -- graph construction and layout -- are called separately here. Every
  optimizer argument is UMAP's own default, listed as module constants above.

  `graph` is copied because `simplicial_set_embedding` prunes it in place
  (weights under max/n_epochs are zeroed), which would otherwise corrupt the
  cached semantic graph shared across the sweep.
  """
  a, b = find_ab_params(SPREAD, min_dist)
  coords, _aux = simplicial_set_embedding(
    embeddings, graph.copy(), 2, INITIAL_ALPHA, a, b, GAMMA,
    NEGATIVE_SAMPLE_RATE, N_EPOCHS, "spectral", check_random_state(seed),
    metric, {}, False, {}, False,
  )
  return np.asarray(coords)


def layouts(embeddings, betas, args, out_dir):
  """One layout per beta, saved and returned as (label, coords) pairs.

  The semantic graph is built once. It does not depend on beta, it is the
  expensive half (the kNN search over 1024-d vectors), and rebuilding it per row
  would also let the random state drift -- so the rows would differ in their
  neighbour graph as well as in beta, which is the one thing the sweep must rule
  out.
  """
  print(f"  semantic graph: n_neighbors={args.neighbors} metric={METRIC} ...")
  semantic = semantic_graph(embeddings, args.neighbors, args.seed, METRIC)
  chain = chain_graph(len(embeddings), args.chain_width)
  print(f"  {semantic.nnz} semantic edges, {chain.nnz} chain edges "
        f"(width {args.chain_width})")

  rows = []
  for beta in betas:
    graph = blend(semantic, chain, beta)
    name = (f"chain_umap_b{beta:g}_n{args.neighbors}"
            f"_w{args.chain_width}_d{args.min_dist:g}_s{args.seed}")
    path = os.path.join(out_dir, f"{name}.npy")

    if os.path.exists(path) and not args.refit:
      print(f"  cached  beta={beta:g}")
      coords = np.load(path)
    else:
      print(f"  fitting beta={beta:g} ({graph.nnz} edges) ...")
      coords = embed(embeddings, graph, args.seed, args.min_dist, METRIC)
      np.save(path, coords)

    label = "beta = 0 (plain UMAP)" if beta == 0 else f"beta = {beta:g}"
    rows.append((beta, label, coords))
  return rows


def layout_grid(rows, metrics, spec, book, model, path):
  """The layouts themselves, one panel per beta, in one figure.

  Each panel is titled with its two headline numbers, because the whole claim of
  this script is that a layout cannot be judged by eye -- a ribbon looks ordered
  whether or not it kept the semantics, and a blob looks disordered whether or
  not it kept the order.
  """
  ncols = min(3, len(rows))
  nrows = math.ceil(len(rows) / ncols)
  fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.4 * nrows),
                           squeeze=False)

  sc = None
  for ax, (beta, label, coords) in zip(axes.ravel(), rows):
    m = metrics[beta]
    sc = panel(
      ax, coords, spec,
      f"{label}\ntrust {m['trustworthiness']:.3f}  "
      f"order {m['order_adjacency']:.3f}",
      "UMAP-1", "UMAP-2",
    )
  for ax in axes.ravel()[len(rows):]:
    ax.set_axis_off()

  save_grid(
    fig, sc,
    f"{book} / {model} -- UMAP with reading-order edges "
    f"(n_neighbors={N_NEIGHBORS}, chain width {CHAIN_WIDTH})",
    spec.label, path,
  )


def curve_figure(rows, metrics, reference, book, model, k, path):
  """The trade-off, twice: as a Pareto curve and as metrics against beta.

  Left is the decision. Every point is one beta; up is more reading order, right
  is a more trustworthy picture of the embedding's own neighbourhoods. A segment
  that rises without moving left is order bought for free; one that rises while
  moving left is the price, and its slope is what you are agreeing to pay.

  Right is the diagnosis, and the reason chapter purity and character agreement
  are on the page at all. Order and chapter purity rising together means the
  layout is tracking position, which is most of what a chain edge can teach it.
  Character agreement is the control on that: a cast is not contiguous in
  reading order, so if it holds while order rises, the chain edges resolved the
  layout rather than overwriting its content.

  Dashed lines are the raw high-dimensional embedding, scored by the identical
  function on the identical labels -- the reference the projection approximates.
  """
  fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6))

  betas = [b for b, _, _ in rows]
  trust = [metrics[b]["trustworthiness"] for b in betas]
  order = [metrics[b]["order_adjacency"] for b in betas]

  left.plot(trust, order, "-", color="0.6", linewidth=1, zorder=1)
  left.scatter(trust, order, c=betas, cmap="viridis", s=90, zorder=2,
               edgecolors="#33322e", linewidths=0.4)
  for beta, t, o in zip(betas, trust, order):
    left.annotate(f"{beta:g}", (t, o), textcoords="offset points",
                  xytext=(7, 4), fontsize=9)
  left.axhline(reference["order_adjacency"], color="#2a78d6", linestyle="--",
               linewidth=1,
               label=f"raw embedding ({reference['order_adjacency']:.3f})")
  left.set_xlabel(f"trustworthiness (k={k})")
  left.set_ylabel(f"share of {k} plotted neighbours within "
                  f"{ORDER_WINDOW} paragraphs")
  left.set_title("What order costs in locality")
  left.legend(loc="best", fontsize=9, frameon=False)

  series = [("order_adjacency", "reading order", "#2a78d6", "o"),
            ("chapter_purity", "chapter purity", "#008300", "s"),
            ("trustworthiness", "trustworthiness", "#eda100", "^")]
  if "character_jaccard" in reference:
    series.append(("character_jaccard", "character Jaccard", "#e87ba4", "D"))

  for key, label, hue, marker in series:
    right.plot(betas, [metrics[b][key] for b in betas], marker=marker,
               color=hue, label=label, linewidth=1.4, markersize=5)
    if key in reference:
      right.axhline(reference[key], color=hue, linestyle="--", linewidth=0.8,
                    alpha=0.6)
  right.set_xlabel("beta (weight on the reading-order edges)")
  right.set_ylabel("score")
  right.set_title("Each metric against beta\n(dashed: the raw embedding)")
  right.legend(loc="best", fontsize=9, frameon=False)

  fig.suptitle(f"{book} / {model} -- reading order as graph structure",
               fontsize=14)
  fig.tight_layout(rect=[0, 0, 1, 0.95])
  fig.savefig(path, dpi=200)
  plt.close(fig)
  print(f"  wrote {path}")


def run(book, model, args):
  print(f"\n=== {book} / {model} ===")
  paragraphs, embeddings = load_book(book, model)
  labels = np.array([p["chapter_id"] for p in paragraphs])
  print(f"  {len(embeddings)} paragraphs, {embeddings.shape[1]}-d, "
        f"{len(set(labels))} chapters")

  # Same as projection_comparison: a book without the LLM annotation pass loses
  # one metric, not the run.
  try:
    names, cast = load_characters(book, paragraphs)
  except FileNotFoundError as e:
    print(f"  no character labels, skipping character agreement -- {e}")
    names, cast = [], None

  specs = resolve_colors(args.color, book, paragraphs)
  out_dir = paths.out_dir(book, model, paths.CHAIN_UMAP, variant_params={
    "n": args.neighbors, "w": args.chain_width, "d": args.min_dist,
    "k": args.trust_k, "s": args.seed,
  })

  rows = layouts(embeddings, args.beta, args, out_dir)

  rng = np.random.default_rng(args.seed)
  metrics = {beta: score(embeddings, coords, labels, cast, args.trust_k, rng)
             for beta, _, coords in rows}
  reference = label_metrics(embeddings, labels, cast, args.trust_k, rng)

  print(f"\n  {'beta':>6}  {'trust':>7}  {'order':>7}  {'chapter':>7}  {'gap':>5}")
  for beta, _, _ in rows:
    m = metrics[beta]
    print(f"  {beta:>6g}  {m['trustworthiness']:>7.4f}  "
          f"{m['order_adjacency']:>7.4f}  {m['chapter_purity']:>7.4f}  "
          f"{m['order_median_gap']:>5.0f}")
  print(f"  {'raw':>6}  {'--':>7}  {reference['order_adjacency']:>7.4f}  "
        f"{reference['chapter_purity']:>7.4f}  "
        f"{reference['order_median_gap']:>5.0f}\n")

  for spec in specs:
    layout_grid(rows, metrics, spec, book, model,
                os.path.join(out_dir, f"chain_umap_{spec.name}.png"))
  curve_figure(rows, metrics, reference, book, model, args.trust_k,
               os.path.join(out_dir, "chain_tradeoff.png"))

  payload = {
    "book": book, "model": model,
    "n_paragraphs": int(len(embeddings)),
    "embedding_dim": int(embeddings.shape[1]),
    "n_chapters": int(len(set(labels))),
    "n_characters": len(names),
    "n_neighbors": args.neighbors, "min_dist": args.min_dist, "metric": METRIC,
    "chain_width": args.chain_width,
    "trust_k": args.trust_k, "order_window": ORDER_WINDOW,
    "raw_embedding": reference,
    "betas": [{"beta": beta, **metrics[beta]} for beta, _, _ in rows],
  }
  with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
  print(f"  wrote {os.path.join(out_dir, 'metrics.json')}")

  paths.stamp(out_dir, __file__, args, book=book, model=model,
              ladder={"beta": args.beta, "chain_width": args.chain_width},
              umap={"n_neighbors": args.neighbors, "min_dist": args.min_dist,
                    "metric": METRIC, "n_epochs": N_EPOCHS})
  print(f"  -> {out_dir}")
  return payload


def main():
  parser = add_common_args(argparse.ArgumentParser())
  parser.set_defaults(color=["paragraph", "chapter"])
  parser.add_argument("--beta", nargs="+", type=float, default=CHAIN_BETAS,
                      help="Weights on the reading-order edges. 0 is plain "
                           "UMAP and is what every other value is read against.")
  parser.add_argument("--chain-width", default=CHAIN_WIDTH, type=int,
                      help="Chain each paragraph to this many that follow it, "
                           "weighted 1/d. Keep it under the metric's "
                           f"ORDER_WINDOW ({ORDER_WINDOW}).")
  parser.add_argument("--neighbors", default=N_NEIGHBORS, type=int,
                      help="UMAP n_neighbors for the semantic half of the "
                           "graph. Held fixed across the sweep.")
  parser.add_argument("--min-dist", default=MIN_DIST, type=float)
  parser.add_argument("--trust-k", default=15, type=int,
                      help="Neighbourhood size for the metrics. Independent of "
                           "--neighbors, and matching projection_comparison's "
                           "default so the numbers are comparable.")
  parser.add_argument("--refit", action="store_true",
                      help="Refit instead of loading saved coordinates.")
  args = parser.parse_args()

  if args.chain_width >= ORDER_WINDOW:
    parser.error(
      f"--chain-width {args.chain_width} >= ORDER_WINDOW {ORDER_WINDOW}: the "
      "chain would connect exactly the pairs the order metric scores, so the "
      "metric would be measuring its own input."
    )

  for model in models_for(args, parser):
    run(args.book, model, args)

  print("\nDone.")


if __name__ == "__main__":
  main()
