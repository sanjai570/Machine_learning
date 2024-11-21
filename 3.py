# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Sample dataset: Advertising strategies and outcomes
data = {
    'strategy_type': ['online', 'offline', 'social_media', 'email', 'online', 'offline', 'social_media', 'email'],
    'budget': [1000, 500, 1500, 700, 1200, 600, 1400, 800],
    'duration': [10, 15, 12, 8, 11, 16, 13, 9],
    'target_reached': [1, 0, 1, 0, 1, 0, 1, 1]  
}
df = pd.DataFrame(data)

# Encode categorical variables
df['strategy_type_encoded'] = df['strategy_type'].astype('category').cat.codes

# Define features (X) and target (y)
X = df[['strategy_type_encoded', 'budget', 'duration']]
y = df['target_reached']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
