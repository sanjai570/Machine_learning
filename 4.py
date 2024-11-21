# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample dataset: Investment opportunities and outcomes
data = {
    'market_trend': ['up', 'down', 'neutral', 'up', 'down', 'neutral', 'up', 'down'],
    'economic_indicator': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral', 'negative', 'positive'],
    'investment_size': [50000, 20000, 30000, 80000, 25000, 40000, 100000, 15000],
    'successful_investment': [1, 0, 0, 1, 0, 0, 1, 0]  # 1 = Success, 0 = Failure
}
df = pd.DataFrame(data)

# Encode categorical variables
df['market_trend_encoded'] = df['market_trend'].astype('category').cat.codes
df['economic_indicator_encoded'] = df['economic_indicator'].astype('category').cat.codes

# Define features (X) and target (y)
X = df[['market_trend_encoded', 'economic_indicator_encoded', 'investment_size']]
y = df['successful_investment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# Assume new_data is a DataFrame with updated market conditions
new_data = {
    'market_trend': ['up', 'neutral'],
    'economic_indicator': ['positive', 'neutral'],
    'investment_size': [60000, 35000],
    'successful_investment': [1, 0]
}
new_df = pd.DataFrame(new_data)
new_df['market_trend_encoded'] = new_df['market_trend'].astype('category').cat.codes
new_df['economic_indicator_encoded'] = new_df['economic_indicator'].astype('category').cat.codes
new_X = new_df[['market_trend_encoded', 'economic_indicator_encoded', 'investment_size']]
new_y = new_df['successful_investment']

# Combine new data with existing data
X_updated = pd.concat([X, new_X], ignore_index=True)
y_updated = pd.concat([y, new_y], ignore_index=True)

# Retrain model with updated data
model.fit(X_updated, y_updated)

# Evaluate the updated model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Updated Model Accuracy:", accuracy)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['Failure', 'Success'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
