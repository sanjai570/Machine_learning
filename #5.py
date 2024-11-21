# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample dataset: Patient data for disease diagnosis
data = {
    'age': [25, 34, 45, 52, 23, 40, 60, 50, 33, 38],
    'symptom_severity': [7, 6, 8, 5, 9, 7, 4, 6, 8, 5],
    'medical_history_score': [3, 2, 1, 3, 2, 1, 3, 2, 2, 3],
    'disease_present': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  
}
df = pd.DataFrame(data)

# Features and target
X = df[['age', 'symptom_severity', 'medical_history_score']]
y = df['disease_present']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Determine optimal K using cross-validation
k_values = range(1, 11)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Select the best K
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")

# Train the KNN model with optimal K
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Predictions and evaluation
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with K={optimal_k}: {accuracy}")
