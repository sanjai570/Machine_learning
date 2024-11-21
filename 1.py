# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
data = {
    'study_hours': [5, 10, 15, 20, 25],
    'attendance_rate': [90, 85, 95, 80, 75],
    'socioeconomic_background': ['low', 'middle', 'middle', 'high', 'low'],
    'test_score': [65, 70, 85, 90, 60]
}
df = pd.DataFrame(data)

# Preprocessing
# Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse=False)
socioecon_encoded = encoder.fit_transform(df[['socioeconomic_background']])
socioecon_df = pd.DataFrame(socioecon_encoded, columns=encoder.get_feature_names_out(['socioeconomic_background']))

# Scale continuous variables
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['study_hours', 'attendance_rate']])
scaled_df = pd.DataFrame(scaled_features, columns=['study_hours_scaled', 'attendance_rate_scaled'])

# Combine preprocessed data
X = pd.concat([scaled_df, socioecon_df], axis=1)
y = df['test_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
