# Single-Cell Tokenization Analysis

This project explores how different representations (tokenization strategies) of single-cell RNA-seq data affect machine learning performance.

Inspired by recent work on single-cell foundation models (e.g., Heimdall), this study investigates how input representation impacts classification accuracy and generalization.

---

## Representations Tested

- **Raw gene expression** (baseline)
- **Highly variable genes (HVG)** — reduces noise
- **PCA embeddings** — low-dimensional representation
- **Pathway-like grouped features** — biologically inspired aggregation

---

## Method

We simulate single-cell RNA-seq-like data with:
- Multiple cell types
- Marker genes per cell type
- Dropout to mimic sparsity

Each representation is evaluated using a logistic regression classifier.

---

## Key Findings

- HVG selection improves performance by removing noisy genes  
- PCA reduces dimensionality but slightly decreases accuracy  
- Pathway grouping leads to information loss due to oversmoothing  
- Representation choice significantly impacts model performance  

---

## How to Run

```bash
pip install -r requirements.txt
python tokenization_experiment.py
