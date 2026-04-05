# Heart Disease Detector

A binary classification project that predicts the presence of heart disease using a PyTorch-based logistic regression model trained on clinical patient data.

## Overview

This project loads a heart disease dataset, preprocesses it, trains a neural network model with a single linear layer and sigmoid activation, and evaluates its performance. The trained model achieves **~95.11% accuracy** on the held-out test set.

## Dataset

The dataset (`heart.csv`) contains **918 patient records** with the following 12 features:

| Feature | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | Sex of the patient (M/F) |
| `ChestPainType` | Type of chest pain (ATA, NAP, ASY, TA) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FastingBS` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `RestingECG` | Resting ECG results (Normal, ST, LVH) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y/N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease` | Target variable (1 = Heart Disease, 0 = Normal) |

## Model Architecture

```
LinearRegressionModel(
  (linear_layer): Linear(in_features=15, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```

Categorical columns (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) are one-hot encoded, expanding the feature space from 11 to **15 input features**.

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Epochs | 1000 |
| Train/Test Split | 80% / 20% (734 / 184 samples) |

## Results

| Metric | Value |
|---|---|
| Test Loss | 0.1217 |
| Test Accuracy | **95.11%** |

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

## Usage

Open and run `Heart_Disease_classifier.ipynb` in Jupyter Notebook or [Google Colab](https://colab.research.google.com/).

1. Place `heart.csv` in the same directory as the notebook.
2. Run all cells sequentially to preprocess data, train the model, and evaluate it.
3. Use the inference example at the end of the notebook to make predictions on new patient data.
