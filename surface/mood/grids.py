"""Comparison grids for the mood surface and the arc on it.

    python surface/mood/grids.py --grid blends      # how six emotions become one height
    python surface/mood/grids.py --grid sweep       # softmax temperature x bandwidth
    python surface/mood/grids.py --grid sampling    # which paragraphs the arc uses
    python surface/mood/grids.py --grid chapters    # coarse vs fine narrative units
    python surface/mood/grids.py --grid windows     # the arc for each saved window setting
    python surface/mood/grids.py --grid angles      # one arc from four cameras

Figures land in surface/mood/output/<book>/<model>/.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
    _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import data
import geodesic
import mood as mood_mod
import paths
import plot
import sampling
import surface


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
    blends = mood_mod.BLENDS
    cols = 2
    rows = int(np.ceil(len(blends) / cols))
    fig = plt.figure(figsize=(6 * cols, 5.5 * rows))
    for k, b in enumerate(blends):
        m, order, positions, _ = mood_mod.mood(scores, emotions, blend=b,
                                               temp=args.temp)
        GX, GY, Z, _ = surface.fit(coords, m, args.h)
        _panel(fig, rows, cols, k, GX, GY, Z, order, positions, f"blend {b}")
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


def grid_sweep(coords, emotions, scores, chapters, args):
    """Softmax temperature (rows) x surface bandwidth (columns).

    Only the softmax blend has a temperature, so this grid always uses it: a low
    temperature snaps to the strongest emotion, a high one averages toward the
    middle. The bandwidth decides how far each paragraph's mood spreads.
    """
    temps, bws = args.temps, args.bandwidths
    fig = plt.figure(figsize=(5 * len(bws), 4.6 * len(temps)))
    for r, t in enumerate(temps):
        m, order, positions, _ = mood_mod.mood(scores, emotions, blend="softmax",
                                               temp=t)
        for c, h in enumerate(bws):
            GX, GY, Z, _ = surface.fit(coords, m, h)
            _panel(fig, len(temps), len(bws), r * len(bws) + c, GX, GY, Z,
                   order, positions, f"temp {t:g}   h {h:g}")
    fig.suptitle("softmax mood: temperature x bandwidth", fontsize=13)
    return fig, "grid_sweep.png"


def grid_windows(coords, emotions, scores, chapters, args):
    """One panel per saved window setting, all on the same mood surface.

    The surface is fitted to paragraphs, so it does not depend on the window:
    every panel shows the same terrain and only the route changes. Each window
    is represented by its medoid paragraph. Window settings are read from what
    common/windows.py has saved, so this cannot
    disagree with the rest of the repository about where a window begins.
    """
    root = paths.out_dir(args.book, args.model, paths.WINDOWS, create=False)
    settings = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        f = os.path.join(root, name, "series.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            settings.append((name, d["starts"], d["stops"]))
    if not settings:
        raise SystemExit(f"no saved window settings under {root} -- run "
                         "python common/windows.py first")

    m, order, positions, _ = mood_mod.mood(scores, emotions, blend=args.blend,
                                           temp=args.temp)
    GX, GY, Z, _ = surface.fit(coords, m, args.h)
    cols = min(3, len(settings))
    rows = int(np.ceil(len(settings) / cols))
    fig = plt.figure(figsize=(6 * cols, 5.4 * rows))
    for k, (name, starts, stops) in enumerate(settings):
        idx = np.array([sampling._medoid(coords, np.arange(a, b))
                        for a, b in zip(starts, stops)])
        centers = (starts + stops - 1) / 2.0
        ax = _panel(fig, rows, cols, k, GX, GY, Z, order, positions,
                    f"{name}   {len(idx)} windows")
        runs, solved, total = geodesic.route(GX, GY, Z, coords[idx], centers,
                                             args.alpha)
        plot.draw_runs(ax, runs, len(coords))
        print(f"  {name}: {solved}/{total} geodesic legs")
    plot.reading_bar(fig, len(coords))
    fig.suptitle("the arc on one mood surface, per window setting", fontsize=13)
    return fig, "grid_windows.png"


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
                    choices=["blends", "sweep", "sampling", "chapters",
                             "windows", "angles"])
    ap.add_argument("--book", default="alice_wonderland")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--plane", default="chain_umap", choices=["pca", "chain_umap"])
    ap.add_argument("--beta", default=2.0, type=float)
    ap.add_argument("--blend", default="banded", choices=mood_mod.BLENDS)
    ap.add_argument("--temp", default=0.15, type=float)
    ap.add_argument("--h", default=0.2, type=float)
    ap.add_argument("--alpha", default=6.0, type=float)
    ap.add_argument("--sampling", default="chapter_medoids")
    ap.add_argument("--n", default=60, type=int)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.15, 0.3, 0.6, 1.0],
                    help="--grid sweep: softmax temperatures (rows)")
    ap.add_argument("--bandwidths", nargs="+", type=float,
                    default=[0.1, 0.15, 0.2, 0.3],
                    help="--grid sweep: surface bandwidths (columns)")
    args = ap.parse_args()

    _, emotions, scores, chapters, _ = data.load_book(args.book)
    coords = data.load_coords(args.book, args.model, args.plane, args.beta)

    builder = {"blends": grid_blends, "sweep": grid_sweep,
               "sampling": grid_sampling, "chapters": grid_chapters,
               "windows": grid_windows, "angles": grid_angles}[args.grid]
    fig, name = builder(coords, emotions, scores, chapters, args)
    fig.subplots_adjust(top=0.9)
    out = paths.out_dir(args.book, args.model, paths.MOOD)
    paths.stamp(out, __file__, args)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
