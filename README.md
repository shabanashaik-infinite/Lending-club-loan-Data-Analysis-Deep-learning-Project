# Lending-club-loan-Data-Analysis-Deep-learning-Project

Lending Club Loan Data Analysis

Project Overview

This project builds a deep learning model to predict loan defaults using historical data from Lending Club (2007–2015). The dataset is highly imbalanced, with features like credit policy, loan purpose, interest rate, and FICO score. The goal is to preprocess the data, handle class imbalance, and evaluate the model using sensitivity (recall) and ROC-AUC metrics.

Domain: Finance
Objective: Predict loan default probability (0 = paid, 1 = default)
Tools: Python, Google Colab, TensorFlow, Scikit-learn, SMOTE, Pandas, Matplotlib, Seaborn

Dataset

The dataset (loan_data.csv) contains:


not.fully.paid: Target variable (0 = paid, 1 = default).


Features:


credit.policy: 1 if meets LendingClub criteria, 0 otherwise.


purpose: Loan purpose (e.g., credit_card, debt_consolidation).



int.rate: Interest rate as a proportion.



installment: Monthly payment.



log.annual.inc: Log of annual income.



dti: Debt-to-income ratio.



fico: FICO credit score.



days.with.cr.line: Days with a credit line.



revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec: Additional financial metrics.

Project Steps


Load loan_data.csv in Google Colab.



Check and impute missing values (median for numerical, mode for categorical).



Analyze class distribution of not.fully.paid.



Balance the dataset using SMOTE.



Visualize class distribution before and after balancing.



Encode categorical variable (purpose) and scale numerical features.



Build and train a deep learning model with class weights.



Evaluate with sensitivity (recall) and ROC-AUC.



Plot training and validation loss.

Requirements



Environment: Google Colab (all libraries pre-installed).


Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn, tensorflow.

How to Run


Open Google Colab: Visit colab.research.google.com and create a new notebook.

Upload Dataset:

Save loan_data.csv locally or upload when prompted by the script.

Copy Script
Execute:

Run the cell. Upload loan_data.csv if prompted.

Outputs include null value summaries, default percentages, class distribution plots, model metrics (sensitivity, ROC-AUC), and a training loss plot.


Review Results: Check console and plots for analysis.



Notes


Imbalance Handling: SMOTE and class weights address the imbalanced dataset.

Model: Neural network with three hidden layers (64, 32, 16 neurons) and dropout (0.3).

Metrics: Sensitivity and ROC-AUC focus on minority class performance.


Troubleshooting:


FileNotFoundError: Ensure loan_data.csv is uploaded.


Memory Issues: Reduce batch_size or simplify the model.


This project is for educational purposes. For feedback or enhancements, provide dataset details or specific requirements.
