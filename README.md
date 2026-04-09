# Stroke Risk Classification

Machine learning project for predicting stroke risk from healthcare tabular data. This repository presents a classification workflow that includes exploratory data analysis, missing-value handling, feature preparation, Random Forest modeling, and evaluation with precision, recall, and F1-score.

## Why this project matters

Stroke prediction is a realistic healthcare classification problem with noisy real-world data, missing values, and class imbalance. In this project, I explored how different strategies for handling missing `bmi` values affect downstream model performance.

This project demonstrates:
- exploratory data analysis on structured healthcare data
- preprocessing of categorical and numerical variables
- comparison of multiple missing-value handling strategies
- supervised classification with Random Forest
- model evaluation using classification metrics

## Project objective

The goal is to predict the binary target variable `stroke` using patient-level demographic and health-related features, including:
- gender
- age
- hypertension
- heart disease
- ever married
- work type
- residence type
- average glucose level
- BMI
- smoking status

## Dataset

The dataset used in this project is `healthcare-dataset-stroke-data.csv`, a healthcare tabular dataset containing patient information and a binary stroke label.

## Approach

### 1. Exploratory Data Analysis

I first explored the dataset through visualizations and feature inspection to understand:
- feature distributions
- class imbalance in the target variable
- categorical variable frequencies
- missing values in the `bmi` column

### 2. Missing-value handling

A main focus of the project was the missing values in `bmi`. I compared multiple strategies:

1. **Column removal**: remove the `bmi` feature entirely
2. **Mean imputation**: fill missing `bmi` values with the column mean
3. **Linear Regression imputation**: estimate missing `bmi` values using a regression model
4. **KNN-based approach**: handle missing `bmi` values with a nearest-neighbors-based method

### 3. Modeling

For each dataset version, I trained a **Random Forest classifier** to predict stroke occurrence.

### 4. Evaluation

The dataset was split into **75% training** and **25% test** data. Model performance was evaluated with:
- Precision
- Recall
- F1-score

## Results

The table below summarizes the reported results from the original project experiments:

| Missing-value strategy | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Remove `bmi` column | 0.250 | 0.015 | 0.028 |
| Mean imputation | 0.200 | 0.018 | 0.033 |
| Linear Regression imputation | 0.500 | 0.018 | 0.035 |
| KNN-based approach | 0.500 | 0.015 | 0.029 |

### Key takeaway

Among the compared approaches, **Linear Regression imputation achieved the highest reported F1-score** in the original coursework experiments. Overall performance remained modest, which suggests the problem is challenging and strongly affected by class imbalance.

## Repository structure

```text
stroke-risk-classification/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── src/
│   ├── stroke_dataset_exploration.py
│   ├── stroke_random_forest_column_removal.py
│   ├── stroke_random_forest_evaluation.py
│   ├── stroke_random_forest_knn_imputation.py
│   └── stroke_random_forest_linear_regression_imputation.py
├── .gitignore
├── LICENSE
└── README.md
