# Probing Place and Heading Representations in Synthetic Chambers

This repository contains the code and notebooks for our LIGN 167 final project, where we probe GPT-2 for emergent spatial representations in a synthetic navigation task.

Authors: **Ishaan Bal** and **Kunisha Anumula**

---

## Project Overview

Humans navigate using internal codes for *where* they are (place) and *which way* they are facing (heading). In this project, we ask whether a text-only transformer language model (GPT-2) develops analogous internal signals when reading short descriptions of movement through simple “chambers”.

We:

- Build a synthetic dataset of 2,000 rooms (“chambers”) with controlled geometry, colors, textures, objects, and headings.  
- Enforce a **compositional** train/test split by holding out specific color–geometry combinations for the test set.  
- Extract **layer-wise GPT-2 embeddings** and train linear probes to decode `place_id` and `heading_id` at each layer.  
- Run **sanity checks** and a **geometry vs direction** decomposition to understand what the heading probe is really using.

At a high level, GPT-2 encodes **heading** much more strongly than **place** in intermediate layers, with both signals collapsing in the final layer.

---

## Repository Structure

- `dataset_curation_lign167_final.ipynb`  
  End-to-end dataset generation and validation:
  - Defines the attribute space (geometries, wall colors, floor textures, objects, headings).
  - Generates natural language room descriptions from hand-written templates.
  - Implements a held-out **color–geometry** train/test split.
  - Creates `place_id` and `heading_id` labels and Julian-style **paired chambers**.
  - Runs extensive integrity checks (split sizes, overlaps, coverage).
  - Saves all CSVs and a `dataset_metadata.json` file.

- `analysis_lign167_final.ipynb`  
  Model analysis and probing:
  - Loads the curated dataset splits.
  - Loads GPT-2 and extracts **mean-pooled** embeddings for all 13 layers (embedding + 12 transformer layers).
  - Trains multinomial logistic regression probes to predict `place_id` and `heading_id` from each layer.
  - Produces the **accuracy-by-layer** plot and summary statistics.
  - Re-runs key analyses with **last-token pooling**.
  - Computes a **confusion matrix** for headings and decomposes heading accuracy into:
    - room geometry accuracy
    - facing direction accuracy

- (Optional, if you add it) `LIGN167_Final_Project.pdf`  
  Camera-ready ACL-style paper describing the project, methods, and results.

---

## Data Files

The dataset curation notebook writes the following files to disk:

- `spatial_navigation_dataset.csv`  
  Full dataset (2,000 rows). Each row is one scene with:
  - `scene_id`, `text`
  - `geometry`, `wall_color`, `floor_texture`, `object_type`, `object_side`, `door_side`
  - `place_id`, `heading_id`
  - `split` (`train` / `val` / `test`)
  - `split_condition`, `template_id`, `text_length`

- `train.csv`, `val.csv`, `test.csv`  
  Split-specific CSVs used by the analysis notebook.

- `paired_chambers.csv`  
  Julian-style pairs:
  - same `heading_id`, different `place_id`
  - columns: `pair_id`, `heading_id`, `scene_1`, `place_1`, `scene_2`, `place_2`, `same_heading`, `same_place`.

- `dataset_metadata.json`  
  Configuration and summary stats:
  - counts per split
  - number of unique places/headings
  - list of geometries, colors, textures, objects, directions
  - train/test color–geometry combinations
  - number of templates used.

---

## How to Run the Notebooks

Both notebooks were developed in Google Colab with access to GPU.

### 1. Dataset curation

1. Open `dataset_curation_lign167_final.ipynb` in Colab.
2. Run all cells in order:
   - Section 1–4 build the attribute space, templates, and dataset.
   - Section 5–8 run integrity and validation checks.
   - Section 9 saves the CSVs and `dataset_metadata.json` into the current working directory.
3. Download the generated CSVs (or move them to a known folder in your Drive).

### 2. Model analysis and probing

1. Place `train.csv`, `val.csv`, `test.csv`, and `spatial_navigation_dataset.csv` in a folder accessible to Colab (paths are currently set for Google Drive; adjust if needed).
2. Open `analysis_lign167_final.ipynb` in Colab.
3. Run the import and dataset-loading cells.
4. Run section 10:
   - Loads GPT-2 (`GPT2Tokenizer`, `GPT2Model`) from HuggingFace.
   - Extracts mean-pooled embeddings for each layer for train and test.
5. Run section 11:
   - Trains logistic regression probes for `place_id` and `heading_id` across all layers.
   - Prints per-layer accuracies and identifies best-performing layers.
6. Run section 12:
   - Generates `probing_accuracy_by_layer.png`.
   - Saves `probing_results.csv`.
   - Prints summary statistics used in the paper.
7. Run section 13:
   - Last-token pooling sanity checks for layer 12.
   - Heading confusion matrix and geometry/direction decomposition at layer 11.
   - Final interpretation text.

---

## Key Results (High Level)

- Dataset: 1,190 train, 210 val, 600 test examples; 192 unique places and 12 unique headings. [web:47][web:2]  
- Best **place** decoding: **54.2%** accuracy at layer 11 (mean pooling). [web:47]  
- Best **heading** decoding: **87.7%** accuracy at layer 10 (mean pooling). [web:47]  
- Layer 12 “collapse”: place drops to **10.5%**, heading to **19.8%**. [web:47]  
- At layer 11 with last-token pooling:
  - overall heading accuracy: **84.0%**
  - geometry accuracy: **85.0%**
  - direction accuracy: **99.0%**. [web:47]

These results suggest that GPT-2 builds a strong internal code for **heading** in intermediate layers, with weaker encoding of **place**, and that spatial information is largely lost in the final layer.

---

## Citation

If you reference this project, you can cite it informally as:

> Bal, I. and Anumula, K. (2025). *Probing Place and Heading Representations in Synthetic Chambers*. LIGN 167 Final Project, UC San Diego.
