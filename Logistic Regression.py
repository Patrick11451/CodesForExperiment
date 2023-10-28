import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

# Load and prepare the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data (Feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a logistic regression model
model = LogisticRegression(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Generate and print classification report
report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
print("Classification Report:")
print(report)

# Visualize the decision boundary using synthetic data
# Reduce dimensionality using PCA for visualization
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Create a synthetic dataset with only two features for visualization
X_synthetic = np.vstack([X_train_reduced, X_test_reduced])
y_synthetic = np.hstack([y_train, y_test])

# Train a logistic regression model on the synthetic dataset
model_synthetic = LogisticRegression(random_state=42)
model_synthetic.fit(X_synthetic, y_synthetic)

# Create a meshgrid of points for visualization
h = .02  # Step size in the mesh
x_min, x_max = X_synthetic[:, 0].min() - 1, X_synthetic[:, 0].max() + 1
y_min, y_max = X_synthetic[:, 1].min() - 1, X_synthetic[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the class for each point in the meshgrid
Z = model_synthetic.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape the predicted class into the meshgrid shape
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X_synthetic[:, 0], X_synthetic[:, 1], c=y_synthetic, cmap=plt.cm.RdBu, edgecolors='k')
plt.title('Logistic Regression Decision Boundary (Synthetic Data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
