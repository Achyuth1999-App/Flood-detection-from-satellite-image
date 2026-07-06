# Flood Detection from Satellite Images — CNN Classification Project

**Author:** Achyuth Reddy · MSc Data Science, University of Europe for Applied Sciences (UE Germany)
**Course:** Machine Learning — Phase 2 (Proposal, Code Development and Technical Implementation)
**Supervisor:** Prof. Shan Faiz

## Project Overview

This project implements a CNN-based **binary image classification** pipeline that automatically
detects flooding in satellite/aerial imagery (Flooded vs. Non-Flooded), using both a custom CNN
built from scratch and three transfer-learning backbones (MobileNetV2, VGG16, ResNet50).

## Dataset

- **Name:** Flood Area Segmentation
- **Source:** [Kaggle — faizalkarim/flood-area-segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)
- **Content:** ~290 satellite/aerial images with corresponding binary segmentation masks
- **Adaptation:** Masks are converted into binary classification labels based on the proportion
  of flood-labeled pixels (threshold = 15%)

> The dataset is not included in this repository due to size/licensing. Download it from Kaggle
> and place it locally, or attach it directly as a Kaggle Notebook input (`/kaggle/input/flood-area-segmentation`).

## Repository Structure

```
.
├── README.md                            # This file
├── Flood_Detection_Proposal.docx        # Full project proposal document
├── Flood_Detection_CNN_Notebook.ipynb   # Main Kaggle/Jupyter notebook (recommended entry point)
├── flood_detection_pipeline.py          # Standalone Python script version of the notebook
├── requirements.txt                     # Python dependencies
└── figures/                             # Generated output figures (created after running the code)
```

## How to Run

### Option A — Kaggle Notebook (recommended)
1. Create a new Kaggle Notebook.
2. Add the dataset **Flood Area Segmentation** as a data source.
3. Upload/copy the contents of `Flood_Detection_CNN_Notebook.ipynb`.
4. Enable a GPU accelerator (Settings → Accelerator → GPU T4/P100).
5. Run all cells top to bottom.

### Option B — Local / other Python environment
```bash
git clone <this-repository-url>
cd <repository-folder>
pip install -r requirements.txt

# Download the dataset from Kaggle and place it at ./data/flood-area-segmentation
# (or update DATA_DIR inside flood_detection_pipeline.py to point to your local path)

python flood_detection_pipeline.py
```

Generated figures (class distribution, sample images, accuracy/loss curves, confusion matrices,
ROC curves, Grad-CAM visualizations, error analysis, model comparison table) are saved to `./figures/`.
Trained models are saved to `./models/`.

## Pipeline Summary

1. **Dataset loading & label derivation** from segmentation masks
2. **Preprocessing** — resize to 224×224, normalize to [0, 1]
3. **Stratified train/validation/test split** (70/15/15)
4. **Data augmentation** (rotation, flips, zoom, brightness shift) — training set only
5. **Custom CNN** — 4 convolutional blocks + GAP + Dense head
6. **Transfer learning** — MobileNetV2, VGG16, ResNet50, each with two-phase fine-tuning
7. **Evaluation** — accuracy/loss curves, confusion matrix, classification report, ROC-AUC
8. **Grad-CAM visualization** and **error analysis** on misclassified samples
9. **Model comparison table** across all four models

## Research Questions Addressed

- RQ1: Custom CNN performance on flood classification
- RQ2: Transfer learning vs. training from scratch
- RQ3: Effectiveness of data augmentation strategies
- RQ4: Characteristics of commonly misclassified images
- RQ5: Impact of fine-tuning on convergence and accuracy

## Requirements

See `requirements.txt`. Key libraries: TensorFlow/Keras, OpenCV, scikit-learn, pandas, matplotlib, seaborn.

## License

This project is submitted for academic purposes as part of the Machine Learning course at UE Germany.
