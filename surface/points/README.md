# surface/points — the raw point cloud

    python surface/points/raw_points.py --book alice_wonderland --model bge-m3

Each paragraph as a point `(PC1, PC2, score)`, one figure per emotion, with
nothing fitted. This is what every method in `surface/` smooths, so it is the
figure to look at first: it shows how noisy the scores are (neighbouring
paragraphs can disagree about an emotion by as much as 0.8) and where the plane
has no paragraphs at all.

Output: `surface/points/output/<book>/<model>/raw/`.
