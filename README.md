# SVD Image Compression

**Software:** This project demonstrates image compression using **Singular Value Decomposition (SVD)**. The program performs truncated SVD on a grayscale image by retaining the top *k* singular values, allowing users to explore the trade-off between image quality and compression. This standalone implementation was developed for educational purposes to illustrate the application of linear algebra concepts in image processing.

**Author:** Anushka Banerjee  
**Position:** M.S. Student in Artificial Intelligence  
**Institution:** Department of Computer Science, The University of Alabama at Birmingham (UAB)  
**Email:** abanerj2@uab.edu

## Educational Context

This software was originally developed as part of a coursework assignment for the M.S. in Artificial Intelligence program at the University of Alabama at Birmingham. The repository contains only the standalone SVD image compression implementation and is intended for educational and demonstration purposes.

## Requirements

- Python 3.9+
- NumPy
- Pillow

Install the required packages:

```bash
pip install numpy pillow
```

## Running the Program

```bash
python svd_image_compression.py input_image.jpg output_image.jpg --k 100
```

### Example

```bash
python svd_image_compression.py sample.jpg compressed.jpg --k 75
```

The `--k` parameter specifies the number of singular values retained during reconstruction. Larger values preserve more image detail, while smaller values provide greater compression.