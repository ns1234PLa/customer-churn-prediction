# Customer Churn Prediction

Understanding and predicting customer churn helps businesses improve retention and target outreach.
This project uses machine learning and NLP on customer feedback to spot which customers are at risk.




## 📁 Important Project Files

+ **notebook/data_preprocess.ipynb** — End-to-end training & validation workflow  
+ **notebook/generate_full_predictions.ipynb** — Batch predictions + SHAP explanations  
+ **notebook/all_customers_predictions.csv** — Final UI-ready churn predictions  
+ **models/best_xgb_model.pkl** — Final trained XGBoost model  
+ **metrics/shap_all_customers.csv** — SHAP feature importance for each customer  

## 📋 Content 
+ Data cleaning & feature engineering (structured + text)
+ Model comparison (XGBoost, SVM, Random Forest, Logistic Regression)
+ SHAP-based explainability — see what drives churn for every customer
+ Ready-to-use predictions and explanations (CSV/JSON)
+ Outputs for dashboards or further analysis


## 🛠️ Methodology 
+ Extracted key features from customer records and feedback
+ Trained multiple classifiers and compared key performance metrics
+ Exported predictions and SHAP explanations for review and dashboard integration


## ✔️ Sections 
+ Exploratory Data Analysis (EDA)
+ Model Training & Validation
+ Explainability using SHAP
+ Batch Prediction & Export Pipeline


## How to Use
+ Run the notebooks in Jupyter Notebook or VS Code
+ Use exported CSV/JSON files for dashboards, reporting, or UI integration


## 📄 License 
MIT License—open for learning, collaboration, and reuse.











