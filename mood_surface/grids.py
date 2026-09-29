"""Comparison grids for the mood surface and the arc on it.

    python mood_surface/grids.py --grid blends      # how six emotions become one height
    python mood_surface/grids.py --grid sampling    # which paragraphs the arc uses
    python mood_surface/grids.py --grid chapters    # coarse vs fine narrative units
    python mood_surface/grids.py --grid angles      # one arc from four cameras
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import data
import geodesic
import mood as mood_mod
import plot
import sampling
import surface

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _panel(fig, rows, cols, k, GX, GY, Z, order, positions, title):
    ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
    plot.draw_surface(ax, GX, GY, Z, order, positions, alpha=0.55)
    ax.set_title(title, fontsize=10)
    return ax


def _arc(ax, GX, GY, Z, coords, idx, alpha, n_paragraphs):
    runs, solved, total = geodesic.route(GX, GY, Z, coords[idx],
                                         idx.astype(float), alpha)
    plot.draw_runs(ax, runs, n_paragraphs)
    return solved, total


def grid_blends(coords, emotions, scores, chapters, args):
    """One surface per blend: how six emotions collapse to one readable height."""
    blends = ["softmax", "dominant", "banded"]
    fig = plt.figure(figsize=(6 * len(blends), 5.5))
    for k, b in enumerate(blends):
        m, order, positions, _ = mood_mod.mood(scores, emotions, blend=b,
                                               temp=args.temp)
        GX, GY, Z, _ = surface.fit(coords, m, args.h)
        _panel(fig, 1, len(blends), k, GX, GY, Z, order, positions, f"blend {b}")
    fig.suptitle("how six emotions become one mood height", fontsize=13)
    return fig, "grid_blends.png"


def grid_sampling(coords, emotions, scores, chapters, args):
    """One panel per sampling strategy, arc on the same surface."""
    m, order, positions, _ = mood_mod.mood(scores, emotions, blend=args.blend,
                                           temp=args.temp)
    GX, GY, Z, _ = surface.fit(coords, m, args.h)
    strategies = {
        "chapter medoids": sampling.chapter_medoids(coords, chapters),
        "uniform": sampling.uniform(coords, args.n),
        "douglas peucker": sampling.douglas_peucker(m, args.n),
        "farthest point": sampling.farthest_point(coords, args.n),
    }
    fig = plt.figure(figsize=(12, 11))
    for k, (name, idx) in enumerate(strategies.items()):
        ax = _panel(fig, 2, 2, k, GX, GY, Z, order, positions,
                    f"{name} ({len(idx)})")
        _arc(ax, GX, GY, Z, coords, idx, args.alpha, len(coords))
    plot.reading_bar(fig, len(coords))
    fig.suptitle("which paragraphs the arc runs through", fontsize=13)
    return fig, "grid_sampling.png"


def grid_chapters(coords, emotions, scores, chapters, args):
    """Arc through narrative units, coarse to fine (units respect chapters)."""
    m, order, positions, _ = mood_mod.mood(scores, emotions, blend=args.blend,
                                           temp=args.temp)
    GX, GY, Z, _ = surface.fit(coords, m, args.h)
    fig = plt.figure(figsize=(18, 10))
    for k, n_units in enumerate([6, 12, 24, 48, 96]):
        units = _units(chapters, n_units)
        idx = np.array(sorted(sampling._medoid(coords, u) for u in units))
        ax = _panel(fig, 2, 3, k, GX, GY, Z, order, positions, f"{len(idx)} units")
        _arc(ax, GX, GY, Z, coords, idx, args.alpha, len(coords))
    plot.reading_bar(fig, len(coords))
    fig.suptitle("the arc through coarse vs fine narrative units", fontsize=13)
    return fig, "grid_chapters.png"


def grid_angles(coords, emotions, scores, chapters, args):
    """One arc, four cameras -- a 3D arc hides its crossings from a fixed angle."""
    m, order, positions, _ = mood_mod.mood(scores, emotions, blend=args.blend,
                                           temp=args.temp)
    GX, GY, Z, _ = surface.fit(coords, m, args.h)
    idx = sampling.pick(args.sampling, coords, m, chapters, args.n)
    views = [(26, -60), (26, 20), (55, -45), (10, -100)]
    fig = plt.figure(figsize=(5.6 * len(views), 5.6))
    for k, (elev, azim) in enumerate(views):
        ax = _panel(fig, 1, len(views), k, GX, GY, Z, order, positions,
                    f"elev {elev}  azim {azim}")
        _arc(ax, GX, GY, Z, coords, idx, args.alpha, len(coords))
        ax.view_init(elev=elev, azim=azim)
    plot.reading_bar(fig, len(coords))
    fig.suptitle(f"{args.sampling}: the same arc from four angles", fontsize=13)
    return fig, f"grid_angles_{args.sampling}.png"


def _units(chapters, target):
    ids = np.unique(chapters)
    runs = [np.where(chapters == c)[0] for c in ids]
    if target >= len(runs):
        per = max(1, round(target / len(runs)))
        return [p for r in runs for p in np.array_split(r, min(per, len(r)))]
    return [np.concatenate([runs[i] for i in g])
            for g in np.array_split(np.arange(len(runs)), target)]




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True,
                    choices=["blends", "sampling", "chapters", "angles"])
    ap.add_argument("--book", default="alice_wonderland")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--plane", default="chain_umap", choices=["pca", "chain_umap"])
    ap.add_argument("--beta", default=2.0, type=float)
    ap.add_argument("--blend", default="banded")
    ap.add_argument("--temp", default=0.15, type=float)
    ap.add_argument("--h", default=0.2, type=float)
    ap.add_argument("--alpha", default=6.0, type=float)
    ap.add_argument("--sampling", default="chapter_medoids")
    ap.add_argument("--n", default=60, type=int)
    args = ap.parse_args()

    _, emotions, scores, chapters, _ = data.load_book(args.book)
    coords = data.load_coords(args.book, args.model, args.plane, args.beta)

    builder = {"blends": grid_blends, "sampling": grid_sampling,
               "chapters": grid_chapters, "angles": grid_angles}[args.grid]
    fig, name = builder(coords, emotions, scores, chapters, args)
    fig.subplots_adjust(top=0.9)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
