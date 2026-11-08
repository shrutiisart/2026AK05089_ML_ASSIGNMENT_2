**Shruti Sandbhor**  
Student ID: **2026AK05089**

# 📞 Telco Customer Churn Prediction

## a. Problem Statement

Customer churn is one of the major challenges faced by telecommunication companies. The objective of this project is to predict whether a customer will leave the telecom company (**Churn = Yes**) or stay with the company (**Churn = No**) based on customer demographics, subscribed services, account information, and billing details.

Five machine learning classification models have been implemented, evaluated, compared, and deployed using Streamlit Community Cloud.

---

## b. Dataset Description

### Dataset Name
**Telco Customer Churn Dataset**

### Source
IBM Sample Dataset (Kaggle)

### Dataset Characteristics

| Attribute | Value |
|------------|--------|
| Total Records | 7043 |
| Features | 19 |
| Target Variable | Churn |
| Type | Binary Classification |

### Target Variable

| Value | Meaning |
|---------|---------|
| Yes | Customer Leaves Company |
| No | Customer Stays With Company |

### Features Included

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

---

## c. GitHub Repository Link

### Repository

🔗 **GitHub Repository:**  
https://github.com/shrutiisart/2026AK05089_ML_ASSIGNMENT_2

---

## d. Models Used

The following machine learning classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.7984 | 0.8404 | 0.6406 | 0.5481 | 0.5908 | 0.4606 |
| Decision Tree | 0.7154 | 0.6478 | 0.4667 | 0.5053 | 0.4852 | 0.2894 |
| KNN | 0.7424 | 0.7603 | 0.5151 | 0.5027 | 0.5088 | 0.3343 |
| Naive Bayes | 0.7466 | 0.8201 | 0.5160 | 0.7326 | 0.6055 | 0.4413 |
| Support Vector Machine | 0.7935 | 0.7907 | 0.6509 | 0.4786 | 0.5516 | 0.4299 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved the highest accuracy and AUC score among all models. It demonstrated balanced performance and generalized well on the test dataset. |
| Decision Tree | Easy to interpret but showed comparatively lower performance. The model may be prone to overfitting on this dataset. |
| KNN | Produced moderate performance and benefited from feature scaling. Classification quality was better than Decision Tree but lower than Logistic Regression. |
| Naive Bayes | Achieved the highest recall score, making it effective for identifying churned customers. However, precision was comparatively lower. |
| Support Vector Machine | Achieved strong accuracy and precision. Performance was competitive with Logistic Regression but recall was comparatively lower. |
| **Overall Winner for this Dataset** | **Logistic Regression** achieved the highest Accuracy (79.84%) and AUC (84.04%), making it the best-performing model for this dataset. |

---

## Live Streamlit Application

### Streamlit Deployment Link

🚀 **Live Application:**  
https://2026ak05089mlassignment2-fpgy7pqrwmjvbvzebv2ffg.streamlit.app/

---

## Project Repository Structure

```text
2026AK05089_ML_ASSIGNMENT_2/

│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── models/
    ├── logistic.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── scaler.pkl
```

---

## Features of Streamlit Application

The deployed Streamlit application provides:

✅ Upload Test Dataset (CSV)

✅ Model Selection Dropdown

✅ Evaluation of Uploaded Test Dataset

✅ Accuracy Score

✅ AUC Score

✅ Precision Score

✅ Recall Score

✅ F1 Score

✅ Matthews Correlation Coefficient (MCC)

✅ Confusion Matrix Visualization

✅ Prediction Using Saved Models Without Retraining

---

## BITS Virtual Lab Execution

The machine learning models were trained and evaluated successfully using the BITS Virtual Lab environment. Evaluation metrics for all five machine learning models were generated and verified before deployment.
<img width="1600" height="949" alt="2026ak05089_Img1" src="https://github.com/user-attachments/assets/aab50e4b-cab7-4f5a-9b84-a9f391d593d0" />
<img width="1600" height="948" alt="2026ak05089_img2" src="https://github.com/user-attachments/assets/5e528a39-c797-4156-88ba-8f54b17860b8" />


---

## Conclusion

The Telco Customer Churn dataset was used to build and evaluate five machine learning classification models. The performance of each model was measured using Accuracy, AUC, Precision, Recall, F1 Score, and MCC. Logistic Regression achieved the highest overall performance with an Accuracy of **79.84%** and AUC of **84.04%**.

The developed Streamlit application successfully allows:

- CSV Dataset Upload
- Model Selection
- Evaluation Metric Visualization
- Confusion Matrix Generation

without retraining the machine learning models. The application was successfully deployed on Streamlit Community Cloud and made publicly accessible for evaluation.

---
