# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample monthly electricity consumption data
data = {
    'month': pd.date_range(start='2015-01-01', end='2020-12-31', freq='M'),
    'consumption': np.random.uniform(100, 500, 72)  # Example consumption data
}
df = pd.DataFrame(data)

# Add time-based features
df['time_index'] = np.arange(len(df))  # Trend variable
df['month_sin'] = np.sin(2 * np.pi * (df['month'].dt.month / 12))  # Seasonal pattern (sine)
df['month_cos'] = np.cos(2 * np.pi * (df['month'].dt.month / 12))  # Seasonal pattern (cosine)

# Define features (X) and target (y)
X = df[['time_index', 'month_sin', 'month_cos']]
y = df['consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Add predictions to the dataframe for visualization
df['predicted_consumption'] = np.nan
df.loc[X_test.index, 'predicted_consumption'] = y_pred

# Display the dataframe
print(df.tail())
