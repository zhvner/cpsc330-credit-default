# Prediction of Credit Default: An ML Approach to Financial Risk Management

## Overview
This project explores financial risk management by predicting credit card payment defaults using machine learning techniques. By leveraging various classification models, data preprocessing techniques, and model interpretability tools, we aim to develop an effective and explainable model to assist financial institutions in decision-making.

## Dataset
We utilized the **UCI Credit Card Default Dataset** from Kaggle, which consists of **30,000 customer records**, including:
- **Demographics:** Age, marital status, education.
- **Financial Metrics:** Credit limit, bill amounts, payment history.
- **Target Variable:** Whether the customer defaulted on payment the following month.

## Challenges
- **Class Imbalance:** Only **23% of customers defaulted**, requiring resampling techniques.
- **Outliers:** Extreme values in bill amounts and credit limits required preprocessing.
- **Feature Selection:** Used Recursive Feature Elimination with Cross-Validation (RFECV) to improve model efficiency.

## Models Evaluated
We tested multiple machine learning models and optimized them through hyperparameter tuning:

| Model               | Accuracy | AUC-ROC | Speed |
|---------------------|----------|---------|-------|
| DummyClassifier     | <0.75    | Low     | Fast  |
| Logistic Regression | 0.78     | Low     | Fast  |
| Random Forest      | Moderate | Medium  | Medium |
| **XGBoost**        | **Best** | **0.778** | 0.87s |
| **LightGBM**       | **Second Best** | Competitive | **0.54s** |

### Final Model Selection
✅ **Best Model:** **XGBoost** (Highest accuracy, best AUC-ROC, strong interpretability).  
✅ **Runner-Up:** **LightGBM** (Faster than XGBoost but slightly less robust).  

## Feature Importance Analysis
To understand model decisions, we used **SHAP (SHapley Additive exPlanations)** for feature impact analysis. Key insights:
- **DELAYED_PAYMENT_1:** Strongest predictor of default.
- **Debt-to-Limit Ratio:** High values indicate financial strain.
- **Total Payment Amount:** Higher values reduce default risk.

## Code Implementation
### Feature Selection (RFECV)
```python
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Define pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=123))
])

# Recursive Feature Elimination with Cross-Validation (RFECV)
rfecv = RFECV(estimator=pipeline['model'], step=1, cv=5, scoring='accuracy', n_jobs=-1)
rfecv_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rfecv', rfecv)
])
rfecv_pipeline.fit(X_train, y_train)

# Selected Features
selected_features = rfecv.support_
print("Selected Features:", selected_features)
```

### SHAP Feature Importance Visualization
```python
import shap
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)
```

## Future Improvements
- **Handling Class Imbalance** with SMOTE or threshold tuning.
- **Ensemble Learning** by combining XGBoost and LightGBM.
- **Adapting to Economic Changes** by integrating real-time financial data.

## Personal Reflection
My journey through this project has deepened my appreciation for the intersection of machine learning and financial risk analysis. Navigating challenges such as class imbalance and feature selection reinforced the importance of data preprocessing and model interpretability in real-world applications. My passion lies in leveraging AI to build transparent and ethical decision-making systems, ensuring that machine learning models not only predict outcomes but also provide actionable insights for businesses and individuals alike. I am particularly fascinated by how advanced techniques like SHAP and ensemble learning can improve model trustworthiness, and I look forward to further exploring their applications in finance and security.

## Conclusion
This project demonstrates how machine learning can effectively predict financial risk. The combination of feature selection, model optimization, and explainability techniques enables better decision-making in the financial sector.

## References
- Kaggle: UCI Credit Card Default Dataset
- Scikit-Learn, XGBoost, LightGBM Documentation
- SHAP Explainability Techniques
