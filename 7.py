# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Sample dataset: Loan applicants' data
data = {
    'credit_score': [700, 650, 600, 720, 580, 690, 640, 710],
    'income_level': [50000, 40000, 30000, 55000, 25000, 45000, 35000, 52000],
    'employment_status': ['employed', 'unemployed', 'employed', 'employed', 'unemployed', 'employed', 'unemployed', 'employed'],
    'loan_default': [0, 1, 1, 0, 1, 0, 1, 0]  # 0 = No Default, 1 = Default
}
df = pd.DataFrame(data)

# Encode categorical variables
df['employment_status_encoded'] = df['employment_status'].astype('category').cat.codes

# Define features and target
X = df[['credit_score', 'income_level', 'employment_status_encoded']]
y = df['loan_default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get coefficients
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Print coefficients and intercept
print("Coefficients:", dict(zip(X.columns, coefficients)))
print("Intercept:", intercept)

# Predict and evaluate the model
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc)

# Odds ratio interpretation
odds_ratios = np.exp(coefficients)
print("Odds Ratios:", dict(zip(X.columns, odds_ratios)))
