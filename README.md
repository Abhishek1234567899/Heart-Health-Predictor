# https://heart-health-predictor-g1k5.onrender.com

# Heart Disease Prediction

This project is a machine learning-based model that predicts the likelihood of a patient having heart disease based on various health attributes. The goal is to assist healthcare professionals in diagnosing heart disease by analyzing key features such as age, cholesterol levels, blood pressure, and more.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Component](#Component)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project uses machine learning algorithms to predict whether a person has heart disease or not. It utilizes various classification models such as Logistic Regression, Decision Trees, and Random Forest ,KNN to analyze the dataset and make predictions.

## Dataset

The dataset used in this project is the **Heart Disease UCI dataset**, which contains medical attributes from patients. The dataset is publicly available and can be downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

### Dataset Columns:
- `age`: Age of the patient.
- `sex`: Gender (1 = male, 0 = female).
- `cp`: Chest pain type.
- `trestbps`: Resting blood pressure.
- `chol`: Serum cholesterol.
- `fbs`: Fasting blood sugar.
- `restecg`: Resting electrocardiographic results.
- `thalach`: Maximum heart rate achieved.
- `exang`: Exercise induced angina.
- `oldpeak`: Depression induced by exercise relative to rest.
- `slope`: Slope of peak exercise ST segment.
- `ca`: Number of major vessels colored by fluoroscopy.
- `thal`: Thalassemia.

## Features

- **Health-related features** such as blood pressure, cholesterol, and maximum heart rate.
- **Lifestyle-related factors** including exercise-induced chest pain and blood sugar levels.

## Model

The model is built using several classification algorithms, including:
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **KNN**

These models are evaluated based on their accuracy, precision, recall, and F1-score.

## Installation

### Prerequisites
Make sure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Install Dependencies
You can install the required libraries using pip:

```bash
pip install -r requirements.txt

```source  activate ./env

```pyhton app.py