# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset: Email text and labels (spam or not)
data = {
    'email_text': ['Win a million dollars now!', 'Meeting tomorrow at 10 AM', 'Congratulations, you have won!', 'Important update on your account'],
    'label': [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}
df = pd.DataFrame(data)

# Vectorize email text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['email_text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Linear SVM Model
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# Non-linear SVM Model (using RBF kernel)
non_linear_svm = SVC(kernel='rbf')
non_linear_svm.fit(X_train, y_train)
y_pred_non_linear = non_linear_svm.predict(X_test)

# Evaluate both models
print("Linear SVM Classification Report:\n", classification_report(y_test, y_pred_linear))
print("Accuracy:", accuracy_score(y_test, y_pred_linear))

print("\nNon-linear SVM Classification Report:\n", classification_report(y_test, y_pred_non_linear))
print("Accuracy:", accuracy_score(y_test, y_pred_non_linear))
