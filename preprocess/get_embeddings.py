"""
Compute paragraph embeddings for a processed book.

Directory structure:

books/
  alice_wonderland/
    processed.json
    embeddings/
      bge-m3/
      qwen3-embedding/
      e5-large-v2/

Examples:

python preprocess/get_embeddings.py --book standard/alice_wonderland --model bge-m3
python preprocess/get_embeddings.py --book standard/alice_wonderland --model qwen3-embedding --batch-size 8
"""

import os
import argparse
import json

import numpy as np
import torch

from models import build_model


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--book", type=str, required=True,
    help="Book directory."
  )
  parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["bge-m3", "qwen3-embedding", "e5-large-v2"],
    help="Embedding model.",
  )
  parser.add_argument(
    "--batch-size", type=int, default=16,
    help="Batch size for embedding computation."
  )
  parser.add_argument(
    "--device", type=str, default=None,
    help="cuda / cpu (default: auto detect)",
  )
  args = parser.parse_args()

  # Set device
  if args.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    device = args.device

  print(f"Using device: {device}")

  # Load processed book
  book_dir = os.path.join("books", args.book)
  processed_file = os.path.join(book_dir, "processed.json")
  if not os.path.exists(processed_file):
    raise FileNotFoundError(processed_file)

  with open(processed_file, "r", encoding="utf-8") as f:
    book = json.load(f)
  paragraphs = [p["text"] for p in book["paragraphs"]]

  print(f"Loaded {len(paragraphs)} paragraphs.")

  # Build model
  model = build_model(args.model, device=device)
  print(f"Model: {model.name}")
  print(f"HF repo: {model.hf_model}")

  # Compute embeddings
  embeddings = model.encode(paragraphs, batch_size=args.batch_size)
  print(f"Embedding shape: {embeddings.shape}")

  # Save output
  output_dir = os.path.join(book_dir, "embeddings", model.name)
  os.makedirs(output_dir, exist_ok=True)

  embedding_file = os.path.join(output_dir, "embeddings.npy")
  np.save(embedding_file, embeddings)

  metadata = {
    "model": model.name,
    "hf_model": model.hf_model,
    "num_paragraphs": len(paragraphs),
    "embedding_dimension": int(embeddings.shape[1]),
    "dtype": str(embeddings.dtype),
    "normalized": True,
    "batch_size": args.batch_size,
    "device": device,
  }
  metadata_file = os.path.join(output_dir, "metadata.json")
  with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

  print()
  print("===================================")
  print("Finished.")
  print(f"Embeddings : {embedding_file}")
  print(f"Metadata   : {metadata_file}")
  print("===================================")


if __name__ == "__main__":
  main()