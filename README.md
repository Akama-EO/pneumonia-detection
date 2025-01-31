# Computer Vision - Pnuemonia Detection with Mask R-CNN 
<p align="center">
<img src="https://github.com/Akama-EO/pneumonia-detection/blob/main/sample-prediction.png" width="450" height="450">

Click [here](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) to learn more about the challenge.

## Table Of Contents
  - [Introduction](#introduction)
  - [Problem Statement](#problem-statement)
  - [Aim & Objectives](#aim--objectives)
  - [Methodology](#methodology)
  - [Tools & Technologies](#tools--technologies)
  - [Project Artifacts](#project-artifacts)
  
## Introduction
Despite their widespread use, the interpretation of chest X-Ray (CXR) images remains a challenging task, often requiring significant expertise and experience. The traditional methods of interpreting CXR images by radiologists, while essential, have proven to be time-consuming and prone to human error, particularly under the increased strain and workload brought about by the COVID-19 pandemic era. 

## Problem Statement
Pneumonia is a life-threatening infection that affects the lungs. Early detection is critical for effective treatment. This project aims to build a model that can accurately identify pneumonia from chest X-rays, assisting radiologists in diagnosis.

The dataset used consists of ~26,000 chest X-ray images provided by the Radiological Society of North America (RSNA) for the 2018 challenge. It includes labeled images indicating the presence or absence of pneumonia. It also contained some patient information along with bounding boxes for pneumonia regions (where applicable). The dataset had class imbalance (more normal cases than pneumonia). Also, there was a notable variability in the image quality and resolution.

## Aim & Objectives
The aim and objectives of the project are as follows:

- Build a deep learning model to classify chest X-rays as "normal" or "pneumonia."

- Explore techniques to handle class imbalance in the dataset.

- Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

## Methodology
The following methodologies were adopted during for the project.

  1. Data Preprocessing:

      - Resize and normalize images.

      - Augment data to address class imbalance (e.g., rotation, flipping, zooming).

      - Split data into training, validation, and test sets.

  2. Exploratory Data Analysis (EDA):

      - Visualize sample images from both classes.

      - Analyze class distribution.

      - Explore metadata (e.g., patient age, gender) for insights.

  3. Model Selection:

      - Use pre-trained convolutional neural networks (CNNs). Mask R-CNN model with ResNet-50/101 backbone.

      - Fine-tune models on the pneumonia dataset.

      - Experiment with transfer learning to leverage pre-trained weights.

  4. Training:

      - Train models using binary cross-entropy loss.

      - Optimize hyperparameters (learning rate, batch size, etc.).

      - Use techniques like early stopping and learning rate scheduling.

  5. Evaluation:

      - Evaluate models on the test set using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

      - Analyze confusion matrices to understand model performance.

  6. Interpretability:

      - Use Union over Intersection to visualize regions of interest for analysis.

      - Interpret model predictions to ensure they align with clinical knowledge.

## Tools & Technologies
- Programming Language: Python.

- Libraries/Frameworks:
  
  - TensorFlow/Keras for model development.

  - Pydicom/PIL for image processing.

  - Pandas/NumPy for data manipulation.

  - Matplotlib for visualization.

- Development Environment: Google Colab Pro
     
- Hardware: GPU (e.g., NVIDIA) for faster training.

## Project Artifacts
The project artifects were developed with notebooks on [Google Colab](https://colab.research.google.com/).

1. Click [here](https://github.com/Akama-EO/pneumonia-detection/blob/main/exploratory_analysis.ipynb) to view  the notebook on exploratory data analysis.

2. Click [here](https://github.com/Akama-EO/pneumonia-detection/blob/main/model_training.ipynb) to view  the notebook on model training.

3. Click [here](https://github.com/Akama-EO/pneumonia-detection/blob/main/model_evaluation.ipynb) to view the notebook on model evaluation.

