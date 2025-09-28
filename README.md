Real-Time Fraud Detection System
This repository contains the submission for the internship assignment from ServiceHive. The project focuses on developing a real-time fraud detection system to identify fraudulent transactions while minimizing false positives.

Table of Contents
Project Objective

Dataset

Methodology

1. Data Preprocessing

2. Model Development & Justification

3. Performance Evaluation & Comparison

Results

How to Run

Project Structure

Project Objective
The primary goal is to build an effective machine learning model that can:

Accurately detect fraudulent financial transactions in real-time.

Provide probabilistic outputs for fraud risk.

Minimize the number of legitimate transactions flagged as fraudulent (false positives).

Dataset
The analysis is performed on the fraud_data.csv dataset.

Source: Provided by ServiceHive for the assignment.

Size: Contains numerous transaction records.

Features:

TransactionID: Unique identifier for each transaction.

Amount: The value of the transaction.

Time: Timestamp of the transaction.

Location: Geographic location where the transaction occurred.

MerchantCategory: The category of the merchant (e.g., Electronics, Travel).

CardHolderAge: The age of the cardholder.

IsFraud: The target variable. 1 indicates a fraudulent transaction, and 0 indicates a legitimate one.

Methodology
The project follows a structured approach from data preparation to model evaluation.

1. Data Preprocessing
Before training, the data was thoroughly cleaned and prepared:

Handling Missing Values: Missing values in key features were identified and imputed using appropriate strategies (e.g., filling with the mode for categorical data) to ensure data integrity.

Categorical Encoding: Non-numeric features like Location and MerchantCategory were converted into a machine-readable format using one-hot encoding.

Feature Scaling: Numerical features (Amount, Time, CardHolderAge) were scaled using StandardScaler. This prevents features with larger ranges from disproportionately influencing the model.

Handling Imbalanced Data: The dataset is highly imbalanced, with far fewer fraudulent transactions than legitimate ones. To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to create a balanced dataset, preventing model bias towards the majority class.

2. Model Development & Justification
Chosen Model: Logistic Regression

Logistic Regression was selected as the primary model for several key reasons:

Interpretability: It is a transparent model, allowing us to easily understand the influence of each feature (e.g., transaction amount, location) on the likelihood of fraud. This is crucial for explaining the model's decisions to stakeholders.

Probabilistic Output: As required by the assignment, Logistic Regression naturally outputs a probability score (between 0 and 1), which can be used to rank transactions by risk and set a flexible threshold for flagging them.

Efficiency: The model is computationally lightweight and fast, making it an excellent candidate for real-time prediction scenarios where low latency is critical.

Strong Baseline: It serves as a robust baseline to measure the performance of more complex models.

3. Performance Evaluation & Comparison
Comparison Model: LightGBM (Light Gradient Boosting Machine)

To evaluate the effectiveness of Logistic Regression, its performance was compared against a more advanced model, LightGBM.

Why LightGBM? LightGBM is a state-of-the-art gradient boosting framework known for its high accuracy, speed, and efficiency. It can capture complex, non-linear relationships in the data that a simpler model might miss, making it a powerful contender for fraud detection.

Evaluation Metrics:
The models were evaluated using a comprehensive set of metrics suitable for imbalanced classification:

Confusion Matrix: To visualize the counts of true positives, false positives, true negatives, and false negatives.

Precision: Measures the accuracy of positive predictions (minimizes false positives). Precision = TP / (TP + FP)

Recall (Sensitivity): Measures the model's ability to identify all actual positive cases (minimizes false negatives). Recall = TP / (TP + FN)

F1-Score: The harmonic mean of Precision and Recall, providing a single score that balances both concerns.

AUC-ROC Score: Measures the model's ability to distinguish between classes. An AUC score of 1.0 represents a perfect classifier.

Results
Both models were trained on the preprocessed, balanced training set and evaluated on the original, imbalanced test set to simulate real-world conditions.

Metric

Logistic Regression

LightGBM

Accuracy

0.96

0.99

Precision

0.45

0.91

Recall

0.91

0.88

F1-Score

0.60

0.89

AUC Score

0.97

0.99

Analysis:

Logistic Regression achieved excellent recall, meaning it was very effective at identifying most of the actual fraudulent transactions. However, its lower precision indicates it produced a higher number of false positives.

LightGBM demonstrated superior overall performance, with a significantly higher F1-Score and AUC. Its high precision is particularly valuable as it successfully minimizes false positives, which was a key objective of the assignment.

How to Run
Clone the repository:

git clone <repository-url>

Navigate to the project directory:

cd <repository-name>

Install the required libraries:

pip install pandas scikit-learn matplotlib seaborn imbalanced-learn lightgbm

Launch Jupyter Notebook and open the file:

jupyter notebook "fraud-detection-system.ipynb"

Run the cells in the notebook sequentially to see the entire process from data loading to model evaluation.

Project Structure
.
├── fraud-detection-system.ipynb      # Jupyter Notebook with all code and analysis.
├── fraud_data - Sheet 1.csv          # The raw dataset used for the project.
└── README.md                         # This file.
