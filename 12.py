# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Sample dataset: Transaction features and fraud labels
data = {
    'transaction_amount': [100, 200, 150, 500, 300, 50, 400, 600, 150, 700],
    'transaction_location': [1, 2, 1, 3, 2, 1, 2, 3, 1, 2],  
    'user_behavior_score': [0.5, 0.7, 0.6, 0.8, 0.5, 0.6, 0.7, 0.9, 0.6, 0.4],  
    'fraudulent_transaction': [0, 1, 0, 1, 0, 0, 0, 1, 0, 0]  
}
df = pd.DataFrame(data)

# Define features and target
X = df[['transaction_amount', 'transaction_location', 'user_behavior_score']]
y = df['fraudulent_transaction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution
print("Class distribution in training set before balancing:")
print(y_train.value_counts())

# 1. Resampling the minority class using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Check class distribution after balancing
print("\nClass distribution in training set after balancing:")
print(pd.Series(y_train_res).value_counts())

# Train logistic regression model on the balanced dataset
model = LogisticRegression()
model.fit(X_train_res, y_train_res)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

