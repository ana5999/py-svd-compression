import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()

img = Image.open(args.input).convert("L")
a = np.array(img)

u, s, vt = np.linalg.svd(a, full_matrices=False)

k = args.k

if k < 1 or k > min(a.shape):
    raise ValueError(f"k must be between 1 and {min(a.shape)}")

compressed = (u[:, :k] * s[:k]) @ vt[:k, :]
compressed = np.clip(compressed, 0, 255).astype(np.uint8)

Image.fromarray(compressed).save(args.output)

print(f"Compressed image saved to {args.output}")