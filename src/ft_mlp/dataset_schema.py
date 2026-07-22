"""Column layout of the Wisconsin breast cancer dataset.

The raw ``data.csv`` provided with the subject has no header row, so the
column names have to be supplied by hand. They live here so that every
program agrees on the same layout instead of each keeping its own copy.
"""

# Data columns name
DATA_COLUMNS_NAMES = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_std",
    "texture_std",
    "perimeter_std",
    "area_std",
    "smoothness_std",
    "compactness_std",
    "concavity_std",
    "concave_points_std",
    "symmetry_std",
    "fractal_dimension_std",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# Column used to tell a file that carries a header from a raw headerless one.
HEADER_MARKER = "diagnosis"
