"""Standalone grayscale image compression using truncated SVD.

Example:
    python svd_image_compression.py input.jpg output.jpg --k 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    """Compress a 2D grayscale image array using the largest k singular values."""
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image array.")

    max_k = min(image.shape)
    if not 1 <= k <= max_k:
        raise ValueError(f"k must be between 1 and {max_k}, inclusive.")

    # Reduced SVD avoids creating unnecessary full-sized matrices.
    u, singular_values, vt = np.linalg.svd(image.astype(np.float64), full_matrices=False)

    # Reconstruct the rank-k approximation.
    compressed = (u[:, :k] * singular_values[:k]) @ vt[:k, :]

    return compressed.round().clip(0, 255).astype(np.uint8)


def compress_image_file(input_path: Path, output_path: Path, k: int) -> None:
    """Load an image, compress it as grayscale with SVD, and save the result."""
    with Image.open(input_path) as image:
        grayscale = np.asarray(image.convert("L"))

    compressed = compress_image_svd(grayscale, k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(compressed, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress a grayscale image using truncated singular value decomposition (SVD)."
    )
    parser.add_argument("input", type=Path, help="Path to the input image")
    parser.add_argument("output", type=Path, help="Path for the compressed image")
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of singular values to keep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    compress_image_file(args.input, args.output, args.k)
    print(f"Compressed image saved to: {args.output}")


if __name__ == "__main__":
    main()
