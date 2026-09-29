"""Build the mood surface and draw figures on it.

    python mood_surface/run.py --figure surface     # the mood surface
    python mood_surface/run.py --figure plane        # the 2D layout (scatter)
    python mood_surface/run.py --figure arc          # arc over the surface
"""

import argparse
import os

import matplotlib.pyplot as plt

import data
import mood as mood_mod
import plot
import sampling
import surface

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="surface",
                    choices=["surface", "plane", "arc"])
    ap.add_argument("--book", default="alice_wonderland")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--plane", default="chain_umap", choices=["pca", "chain_umap"])
    ap.add_argument("--beta", default=2.0, type=float)
    ap.add_argument("--blend", default="banded",
                    choices=["banded", "softmax", "dominant"])
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
    ap.add_argument("--alpha", default=6.0, type=float,
                    help="height-to-plane exchange rate for geodesic legs")
    args = ap.parse_args()

    paragraphs, emotions, scores, chapters, _ = data.load_book(args.book)
    coords = data.load_coords(args.book, args.model, args.plane, args.beta)
    m, order, positions, _ = mood_mod.mood(
        scores, emotions, blend=args.blend, norm=args.norm, temp=args.temp)
    tag = f"{args.plane}_{args.blend}_h{args.h:g}"

    if args.figure == "plane":
        fig = plot.plane(coords, m, order, positions,
                         f"the {args.plane} plane -- {args.book} / {args.model}",
                         args.plane)
        name = f"plane_{args.plane}.png"
        _save(fig, name)
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
    _save(fig, name)


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




def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
