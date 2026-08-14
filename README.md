# Acoustic Anomaly Detection for Industrial Machines

## Objective

Acoustic condition monitoring and anomaly detection for industrial machines using the MIMII-DUE fan dataset.

## Dataset

MIMII-DUE fan dataset.

Three sections were used:

- Section 00
- Section 01
- Section 02

Each section contains:

- Source-domain normal training data
- Target-domain 3-shot normal training data
- Source-domain test data
- Target-domain test data

## Methods

### Classical baselines

- Isolation Forest
- One-Class SVM

### Deep learning

- CNN Autoencoder

## Acoustic Features

- RMS
- Zero Crossing Rate
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Flatness

## Experiments

1. Source → Source
2. Source → Target
3. Source + 3 Target → Target

## Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix