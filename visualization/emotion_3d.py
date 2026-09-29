"""
Visualize paragraph embeddings in 3-D against the per-paragraph emotion scores.

Two ways to spend the third axis, drawn once per emotion in paragraph_scores.json:

  proj3d   The embeddings are projected to three components and the emotion is
           the color. Every emotion shares one layout, so the plots differ only
           in color: comparing them shows where each emotion lives in the
           semantic manifold.

  score-z  The embeddings are projected to two components and the emotion score
           itself is the z axis (also mirrored in the color). Height above the
           semantic map is the emotion, so peaks are the paragraphs that carry it.

Each projection can be PCA, UMAP or t-SNE.

--color decides what the points say:

  score     the z-axis emotion as a viridis ramp, so height and color agree.

  dominant  one discrete color+marker per emotion -- whichever scores highest
            for that paragraph. Combined with score-z this is the useful one:
            the height is (say) sadness while the color is whichever emotion
            actually wins, so you can see whether the sadness peaks are really
            sadness-dominant or just tall points that humor still owns.

For every emotion we write a static PNG and, if plotly is installed, an
interactive HTML you can rotate, zoom and hover (the tooltip shows the
paragraph's chapter, score, dominant emotion and text).

Example:

python visualization/emotion_3d.py --book alice_wonderland --model bge-m3
python visualization/emotion_3d.py --book alice_wonderland --model bge-m3 --proj umap
python visualization/emotion_3d.py --book alice_wonderland --model bge-m3 --proj tsne \
  --mode score-z --color dominant
"""

import os
import argparse
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import paths
from data import UNCLEAR, dominant, load_paragraphs, load_scores

# The interactive HTML is a bonus on top of the PNGs; without plotly we still
# want the static plots rather than an import error.
try:
  import plotly.graph_objects as go
  HAS_PLOTLY = True
except ImportError:
  HAS_PLOTLY = False

# A single sequential ramp for every emotion: the score is a magnitude, and
# keeping one ramp across plots makes the panels directly comparable.
CMAP = "viridis"

# How the points are colored. categories is None for a continuous score (drawn
# with CMAP and a colorbar), else a list of (label, hue, marker, symbol) indexed
# by the integer codes in values, drawn with a legend.
Color = namedtuple("Color", "values label categories")


def Continuous(values, label, _unused):
  return Color(values, label, None)


def Discrete(codes, label, categories):
  return Color(codes, label, categories)


def project(embeddings, args, n_components):
  """Reduce the embeddings to n_components dimensions.

  Fit at the dimensionality we actually plot. UMAP and t-SNE optimize a layout
  for a given target dimension, so the first two columns of a 3-component fit
  are not the 2-component fit -- reusing them across modes would quietly plot a
  layout nobody asked for.
  """
  if args.proj == "pca":
    pca = PCA(n_components=n_components, random_state=args.seed)
    coords = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    labels = [f"PC{i + 1} ({v:.1%} var)" for i, v in enumerate(var)]
    print(f"  explained variance: {var.sum():.1%} over {n_components} components")
    # PCA is deterministic and its components are nested, so there is nothing
    # parameter-dependent to tag.
    suffix = ""

  elif args.proj == "umap":
    umap = UMAP(
      n_components=n_components,
      n_neighbors=args.neighbors,
      min_dist=args.min_dist,
      random_state=args.seed,
    )
    coords = umap.fit_transform(embeddings)
    labels = [f"UMAP-{i + 1}" for i in range(n_components)]
    # UMAP layout depends on its parameters, so tag filenames to avoid
    # overwriting runs with different settings.
    suffix = f"_n{args.neighbors}_d{args.min_dist}_s{args.seed}"

  else:
    if args.perplexity >= len(embeddings):
      raise ValueError(
        f"perplexity ({args.perplexity}) must be < number of paragraphs "
        f"({len(embeddings)})."
      )
    tsne = TSNE(
      n_components=n_components,
      perplexity=args.perplexity,
      random_state=args.seed,
    )
    coords = tsne.fit_transform(embeddings)
    labels = [f"t-SNE-{i + 1}" for i in range(n_components)]
    # t-SNE layout depends on perplexity and the seed, so tag the filenames.
    suffix = f"_p{args.perplexity}_s{args.seed}"

  return coords, labels, suffix


def static_plot(coords, color, labels, title, output_path):
  """color is either a Continuous or a Discrete (see main)."""
  fig = plt.figure(figsize=(9, 8))
  ax = fig.add_subplot(111, projection="3d")

  if color.categories is None:
    sc = ax.scatter(
      coords[:, 0], coords[:, 1], coords[:, 2],
      c=color.values, cmap=CMAP, vmin=0.0, vmax=1.0,
      s=14, alpha=0.9, linewidths=0, depthshade=False,
    )
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=color.label)
  else:
    for code, (label, hue, marker, _symbol) in enumerate(color.categories):
      mask = color.values == code
      if not mask.any():
        continue
      recessive = label == UNCLEAR
      ax.scatter(
        coords[mask, 0], coords[mask, 1], coords[mask, 2],
        c=hue, marker=marker, s=14 * (0.7 if recessive else 1.0),
        alpha=0.35 if recessive else 0.9,
        # Yellow and aqua sit under 3:1 against white, so the marks carry a
        # thin dark edge to stay visible on the surface.
        linewidths=0 if recessive else 0.3,
        edgecolors="none" if recessive else "#33322e",
        depthshade=False,
        label=f"{label} ({int(mask.sum())})",
      )
    ax.legend(title=color.label, loc="upper left", fontsize=8,
              frameon=True, framealpha=0.9, markerscale=1.4)

  ax.set_xlabel(labels[0])
  ax.set_ylabel(labels[1])
  ax.set_zlabel(labels[2])
  ax.set_title(title)

  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


def interactive_plot(coords, color, labels, title, hover, output_path):
  hover = np.asarray(hover)

  if color.categories is None:
    traces = [
      go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(
          size=3, color=color.values, colorscale="Viridis",
          cmin=0.0, cmax=1.0, opacity=0.9,
          colorbar=dict(title=color.label),
        ),
        text=hover, hoverinfo="text",
      )
    ]
  else:
    # One trace per category so the legend is clickable and each keeps its
    # own symbol -- the second channel the palette needs.
    traces = []
    for code, (label, hue, _marker, symbol) in enumerate(color.categories):
      mask = color.values == code
      if not mask.any():
        continue
      recessive = label == UNCLEAR
      traces.append(go.Scatter3d(
        x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
        mode="markers",
        name=f"{label} ({int(mask.sum())})",
        marker=dict(
          size=2.5 if recessive else 3.5,
          color=hue, symbol=symbol,
          opacity=0.35 if recessive else 0.9,
          line=dict(width=0 if recessive else 0.3, color="#33322e"),
        ),
        text=hover[mask], hoverinfo="text",
      ))

  fig = go.Figure(traces)
  fig.update_layout(
    title=title,
    legend=dict(title=color.label),
    showlegend=color.categories is not None,
    scene=dict(
      xaxis_title=labels[0],
      yaxis_title=labels[1],
      zaxis_title=labels[2],
    ),
    margin=dict(l=0, r=0, t=40, b=0),
  )

  fig.write_html(output_path, include_plotlyjs="cdn")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--book", required=True, type=str)
  parser.add_argument("--model", required=True, type=str)
  parser.add_argument("--proj", default="pca", choices=["pca", "umap", "tsne"])
  parser.add_argument("--mode", default="both",
                      choices=["proj3d", "score-z", "both"],
                      help="proj3d: z is a third component, emotion is the color. "
                           "score-z: z is the emotion score over a 2-D map.")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--neighbors", default=15, type=int,
                      help="UMAP n_neighbors. Matches UMAP's own default; lower "
                           "values fragment the manifold into local islands.")
  parser.add_argument("--min-dist", default=0.1, type=float)
  parser.add_argument("--perplexity", default=30.0, type=float,
                      help="t-SNE perplexity (must be < number of paragraphs).")
  parser.add_argument("--emotions", nargs="*", default=None,
                      help="Subset of emotions to plot (default: all).")
  parser.add_argument("--color", default="score", choices=["score", "dominant"],
                      help="score: the z-axis emotion, as a viridis ramp. "
                           "dominant: one discrete color per emotion, showing "
                           "which emotion wins each paragraph.")
  parser.add_argument("--min-score", default=0.2, type=float,
                      help="For --color dominant: below this top score, a "
                           "paragraph is 'unclear' rather than colored.")
  args = parser.parse_args()

  paragraphs = load_paragraphs(args.book)
  embeddings = np.load(
    os.path.join("books", args.book, "embeddings", args.model, "embeddings.npy")
  )
  if len(embeddings) != len(paragraphs):
    raise ValueError(
      f"embeddings ({len(embeddings)}) and paragraphs ({len(paragraphs)}) mismatch."
    )

  emotions, matrix = load_scores(args.book, paragraphs)

  if args.emotions:
    unknown = [e for e in args.emotions if e not in emotions]
    if unknown:
      raise ValueError(f"Unknown emotion(s) {unknown}. Available: {emotions}")
    selected = args.emotions
  else:
    selected = emotions

  # No variant: the projection, mode and emotion are already in every filename.
  output_dir = paths.out_dir(args.book, args.model, paths.EMOTIONS_3D)
  paths.stamp(output_dir, __file__, args, emotions=selected)

  name = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[args.proj]
  modes = ["proj3d", "score-z"] if args.mode == "both" else [args.mode]

  codes = categories = None
  if args.color == "dominant":
    codes, categories, n_unclear = dominant(emotions, matrix, args.min_score)
    print(f"Dominant emotion: {n_unclear} of {len(codes)} unclear "
          f"(top score < {args.min_score} or tied)")
    top = {c[0]: int((codes == k).sum()) for k, c in enumerate(categories)}
    print("  " + ", ".join(f"{k} {v}" for k, v in top.items()))

  for mode in modes:
    n_components = 3 if mode == "proj3d" else 2

    print(f"Running {name} with {n_components} components for {mode}...")
    coords, labels, suffix = project(embeddings, args, n_components)

    tag = "3d" if mode == "proj3d" else "scorez"
    np.save(os.path.join(output_dir, f"{args.proj}_{tag}{suffix}.npy"), coords)

    # With a dominant color and a component on z, nothing in the plot depends on
    # which emotion we are looping over -- so draw it once instead of six
    # identical times.
    per_emotion = not (mode == "proj3d" and args.color == "dominant")
    targets = selected if per_emotion else [None]

    for emotion in targets:
      scores = matrix[:, emotions.index(emotion)] if emotion else None

      if mode == "proj3d":
        coords_3d = coords
        axis_labels = labels
      else:
        # The score itself is the z axis.
        coords_3d = np.column_stack([coords, scores])
        axis_labels = labels + [emotion]

      if args.color == "dominant":
        color = Discrete(codes, "dominant emotion", categories)
        title = (f"{name} 3-D (colored by dominant emotion)" if mode == "proj3d"
                 else f"{emotion} over the {name} map "
                      f"(z = {emotion}, colored by dominant emotion)")
        base = (f"{args.proj}_{tag}_dominant{suffix}" if mode == "proj3d"
                else f"{args.proj}_{tag}_{emotion}_dominant{suffix}")
      else:
        color = Continuous(scores, emotion, None)
        title = (f"{name} 3-D (colored by {emotion})" if mode == "proj3d"
                 else f"{emotion} over the {name} map (z = {emotion})")
        base = f"{args.proj}_{tag}_{emotion}{suffix}"

      hover = []
      for i, p in enumerate(paragraphs):
        head = f"{p['id']} | chapter {p['chapter_id']}"
        if emotion:
          head += f"<br>{emotion}: {scores[i]:.2f}"
        if categories is not None:
          head += f"<br>dominant: {categories[codes[i]][0]}"
        hover.append(f"{head}<br>{short_text(p)}")

      static_plot(coords_3d, color, axis_labels, title,
                  os.path.join(output_dir, f"{base}.png"))
      if HAS_PLOTLY:
        interactive_plot(coords_3d, color, axis_labels, title, hover,
                         os.path.join(output_dir, f"{base}.html"))

      if emotion:
        print(f"  {emotion}: mean {scores.mean():.2f}, max {scores.max():.2f}")
      else:
        print(f"  wrote {base}.png")

  print()
  if not HAS_PLOTLY:
    print("plotly is not installed, so only the PNGs were written.")
    print("Install it (pip install plotly) for the rotatable HTML plots.")
    print()
  print("Done.")
  print(output_dir)


def short_text(paragraph, chars=160):
  """Short paragraph text for the hover tooltip, wrapped for plotly."""
  text = paragraph["text"].replace("\n", " ")
  if len(text) > chars:
    text = text[:chars] + " ..."
  return text


if __name__ == "__main__":
  main()
