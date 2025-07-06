# Breast Cancer Classification using Machine Learning

## Project Overview

This project uses machine learning to classify breast tumors as malignant or benign based on a dataset sourced from the **UCI Machine Learning Repository**. The dataset contains patient tumor characteristics extracted from images and associated diagnosis labels.

We trained and hyper-parameter tuned three models:  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Classifier (SVC)  

The best performing model, selected based on recall to minimize false negatives, is used for final predictions.

## Dataset

- Source: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Features include mean, standard error, and worst (largest) values of tumor characteristics like radius, texture, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Model Performance

| Model                 | Basic Accuracy | Hyper-parameter Tuned Accuracy |
|-----------------------|----------------|--------------------------------|
| Logistic Regression    | 95%            | 95%                            |
| K-Nearest Neighbors    | 68.3%          | 85%                            |
| Support Vector Classifier (SVC) | 66.7%  | 96.7%                          |

The SVC model with the best recall was selected to reduce false negatives, improving malignant tumor detection.

## Tumor Types

- **Malignant Tumors:** Cancerous tumors capable of invading other tissues and spreading, requiring prompt diagnosis and treatment.
- **Benign Tumors:** Non-cancerous tumors that grow slowly and typically do not spread, though they may require monitoring or removal.

## Usage

The project includes a Streamlit web app interface that allows users to input tumor characteristics and receive a predicted classification (malignant or benign) along with the confidence level.

## How to Run

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the Streamlit app with:

   ```bash
   streamlit run app.py
