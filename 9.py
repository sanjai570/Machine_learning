# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample dataset: Real estate features and prices
data = {
    'location': [1, 2, 3, 1, 2, 3, 1, 2],  
    'size': [1500, 1800, 1200, 1700, 2000, 1300, 1600, 1900],  
    'bedrooms': [3, 4, 2, 3, 4, 2, 3, 4],  
    'price': [400000, 500000, 350000, 450000, 550000, 370000, 460000, 510000]  
}
df = pd.DataFrame(data)

# Features and target
X = df[['location', 'size', 'bedrooms']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print results
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")

# Limitations of R-squared
# R-squared can be high even if the model is overfitting the training data.
# It does not account for the complexity of the model (i.e., it doesn't penalize overfitting).
# It also doesn’t indicate whether the predictions are biased or how well the model generalizes.
