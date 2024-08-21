

# Credit Default Risk

This project analyzes credit default risk using various machine learning models. The goal is to predict whether customers will default on loans based on their credit usage and personal information. Different machine learning techniques such as XGBoost, Logistic Regression, and data balancing methods like SMOTE are employed.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Models and Techniques](#models-and-techniques)
4. [Evaluation Metrics](#evaluation-metrics)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Contact](#contact)

## Project Overview

Credit risk is a critical issue for financial institutions. By predicting default risk, companies can better manage their lending strategies and reduce potential losses. This project explores various machine learning models to predict loan default risk. The models are trained on customer data and feature a combination of numerical and categorical data.

The primary goal is to achieve high predictive performance, focusing on minimizing the number of false positives (customers predicted not to default but who do) while maintaining strong overall performance.

## Data

The dataset used in this project consists of various features that describe customers' financial profiles, credit behavior, and demographics. The dataset includes:
- Credit card balance
- Installment payments
- Previous applications
- Demographic information such as income, age, and education

The dataset was processed and aggregated to create a comprehensive dataset for model training.

## Models and Techniques

Several machine learning models were employed to predict credit default risk, including:
- **XGBoost**: A powerful gradient boosting model known for its performance with structured data.
- **Logistic Regression**: A commonly used model for binary classification tasks.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to balance the class distribution in the training data, ensuring the model performs well on both the majority and minority classes.

## Evaluation Metrics

The models were evaluated using the following metrics:
- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **ROC-AUC Score**: The area under the ROC curve, measuring the model's ability to distinguish between the classes.
- **Precision-Recall Curve**: Focuses on the balance between precision and recall, particularly useful for imbalanced datasets.

## How to Run

To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/olivia3395/credit-default-risk.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the model training and evaluation scripts:
    ```bash
    python train_model.py
    ```

## Results

The model performed well with the following key results:
- **XGBoost without SMOTE**: Achieved an AUC-ROC score of 0.976, indicating excellent model performance on the imbalanced dataset.
- **XGBoost with SMOTE**: Achieved an AUC-ROC score of 0.750, demonstrating the trade-offs of using SMOTE to handle class imbalance.


## Future Work

1. **Model Tuning**: Further hyperparameter tuning to improve the performance of models, especially in handling the minority class.
2. **Feature Engineering**: Additional feature engineering to capture more meaningful patterns from the data.
3. **Other Techniques**: Experimenting with more advanced balancing techniques like ADASYN and cost-sensitive learning.

## Contact

If you have any questions, feel free to contact:
- **Name**: Yuyao (Olivia) Wang
- **Email**: yuyaow@bu.edu

--
