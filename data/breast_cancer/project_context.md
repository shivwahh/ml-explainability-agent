# Project Name

Breast Tumour Malignancy Classifier

# Business Objective

Classify a breast tumour as malignant or benign from 30 numeric features computed
from digitized images of fine-needle-aspirate cell nuclei. The model supports
clinical decision support and diagnostic triage.

# Target Variable

Tumour class — 1 when the tumour is benign, 0 when malignant (matching the
scikit-learn Breast Cancer Wisconsin encoding).

# Key Drivers

Worst-case perimeter, worst radius, worst area, mean concave points, and mean
concavity are typically the most discriminative features between malignant and
benign tumours.

# Notes

All 30 features are numeric measurements grouped as mean, standard error, and
"worst" (largest) values across the cell nuclei. No encoding is applied.
