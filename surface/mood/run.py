"""Build the mood surface and draw figures on it.

    python surface/mood/run.py --figure surface       # the mood surface
    python surface/mood/run.py --figure plane         # the 2D layout (scatter)
    python surface/mood/run.py --figure arc           # arc over the surface

Figures land in surface/mood/output/<book>/<model>/.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
    _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import data
import mood as mood_mod
import paths
from geometry.mesh import DEFAULT_ALPHA
import plot
import sampling
import surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="surface",
                    choices=["surface", "plane", "arc"])
    ap.add_argument("--book", default="alice_wonderland")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--plane", default="pca", choices=["pca", "chain_umap"])
    ap.add_argument("--beta", default=2.0, type=float)
    ap.add_argument("--blend", default="banded", choices=mood_mod.BLENDS,
                    help="how six emotions become one height; see mood.mood()")
    ap.add_argument("--norm", default="rank", choices=["rank", "zscore", "minmax"])
    ap.add_argument("--temp", default=0.15, type=float)
    ap.add_argument("--h", default=0.2, type=float)
    ap.add_argument("--mask-floor", default=None, type=float)
    ap.add_argument("--sampling", default="chapter_medoids",
                    choices=["chapter_medoids", "uniform", "douglas_peucker",
                             "farthest_point"])
    ap.add_argument("--n", default=60, type=int)
    ap.add_argument("--legs", default="geodesic", choices=["geodesic", "straight"],
                    help="geodesic: shortest path along the surface (needs the "
                         "geometry solver). straight: lifted straight line.")
    ap.add_argument("--alpha", default=DEFAULT_ALPHA, type=float,
                    help="height-to-plane exchange rate for geodesic legs")
    args = ap.parse_args()

    paragraphs, emotions, scores, chapters, summaries = data.load_book(args.book)
    coords = data.load_coords(args.book, args.model, args.plane, args.beta)
    m, order, positions, _ = mood_mod.mood(
        scores, emotions, blend=args.blend, norm=args.norm, temp=args.temp)
    tag = f"{args.plane}_{args.blend}_h{args.h:g}"
    out = paths.out_dir(args.book, args.model, paths.MOOD)
    paths.stamp(out, __file__, args)

    if args.figure == "plane":
        fig = plot.plane(coords, m, order, positions,
                         f"the {args.plane} plane -- {args.book} / {args.model}",
                         args.plane)
        _save(fig, out, f"plane_{args.plane}.png")
        return

    GX, GY, Z, height_at = surface.fit(coords, m, args.h, mask_floor=args.mask_floor)

    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    plot.draw_surface(ax, GX, GY, Z, order, positions,
                      alpha=0.55 if args.figure == "arc" else 0.9)

    if args.figure == "surface":
        fig.suptitle(f"mood surface -- {args.book} / {args.model}\n{tag}",
                     fontsize=12)
        name = f"mood_surface_{tag}.png"
    else:
        idx = sampling.pick(args.sampling, coords, m, chapters, args.n)
        if args.legs == "geodesic":
            import geodesic
            runs, solved, total = geodesic.route(
                GX, GY, Z, coords[idx], idx.astype(float), args.alpha)
            plot.draw_runs(ax, runs, len(coords))
            print(f"{solved}/{total} geodesic legs")
        else:
            xy, t = _lift(coords[idx], idx.astype(float))
            pts = plot.np.column_stack([xy, height_at(xy)])
            plot.draw_arc(ax, pts, t, len(coords))
        plot.reading_bar(fig, len(coords))
        fig.suptitle(
            f"narrative arc on the mood surface -- {args.book} / {args.model}\n"
            f"{tag}, {args.sampling} ({len(idx)} paragraphs)", fontsize=12)
        name = f"mood_arc_{args.sampling}_{tag}.png"

    fig.subplots_adjust(top=0.9)
    _save(fig, out, name)


def _lift(waypoints, reading_pos, steps=40):
    """Densely sample each straight xy-segment so the drawn arc lies on the
    surface (z is read from the surface) instead of tunnelling through it."""
    np = plot.np
    xy, t = [], []
    for i in range(len(waypoints) - 1):
        f = np.linspace(0, 1, steps)[:, None]
        xy.append(waypoints[i] * (1 - f) + waypoints[i + 1] * f)
        t.append(reading_pos[i] + f[:, 0] * (reading_pos[i + 1] - reading_pos[i]))
    return np.vstack(xy), np.concatenate(t)




def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
