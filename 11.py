# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample dataset: Historical weather data with daily temperatures
data = {
    'day_of_year': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'temperature': [30, 32, 35, 38, 40, 42, 45, 47, 50, 52]  
}
df = pd.DataFrame(data)

# Add seasonal variation (e.g., sine wave for seasonal pattern)
df['seasonal_variation'] = 10 * np.sin(2 * np.pi * df['day_of_year'] / 365)

# Update temperature with seasonal variation
df['temperature'] += df['seasonal_variation']

# Features and target
X = df[['day_of_year']]
y = df['temperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse}")

# If the model's predictions are inaccurate, this indicates high bias due to not capturing seasonal variations.
# To improve accuracy:
# 1. Include additional features, like month of the year, or use Fourier transforms to model seasonal components.
# 2. Consider using more advanced models like decision trees or time-series models (ARIMA, etc.).
