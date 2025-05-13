from sklearn import datasets
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the Iris dataset
dataset = datasets.load_iris()

# Convert to DataFrame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# Split into features and target
X = df.drop(columns='target')
y = df['target']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Create and train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Evaluate accuracy
print(f'Model accuracy on train: {accuracy_score(y_train, y_pred_train)}')
print(f'Model accuracy on test: {accuracy_score(y_test, y_pred)}')
