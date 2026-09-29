# arc/validation — is the arc real?

Whether the arc is a feature of the embedding or an artifact of the windowing.
`arc_comparison.py` compares the windowed arc with the raw per-paragraph path
(window = 1, no pooling) and with a shuffled reading order, under PCA;
`arc_comparison_projections.py` does the same under UMAP and t-SNE. Read these
before trusting any arc figure.

    python validation/arc_comparison.py --book alice_wonderland --model bge-m3
