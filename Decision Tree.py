1.	# Import necessary libraries
2.	import numpy as np
3.	import pandas as pd
4.	from sklearn.datasets import load_breast_cancer
5.	from sklearn.metrics import confusion_matrix
6.	from sklearn.model_selection import train_test_split
7.	from sklearn.tree import DecisionTreeClassifier
8.	from sklearn.metrics import accuracy_score, classification_report
9.
10.	# Function to load the Breast Cancer dataset
11.	def importdata():
12.	    data = load_breast_cancer()
13.	    X = data.data
14.	    y = data.target
15.	    return X, y
16.
17.	# Function to split the dataset
18.	def splitdataset(X, y):
19.	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
20.	    return X_train, X_test, y_train, y_test
21.
22.	# Function to perform training with giniIndex.
23.	def train_using_gini(X_train, X_test, y_train):
24.	    # Creating the classifier object
25.	    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
26.	    # Performing training
27.	    clf_gini.fit(X_train, y_train)
28.	    return clf_gini
29.
30.	# Function to perform training with entropy.
31.	def train_using_entropy(X_train, X_test, y_train):
32.	    # DTwith entropy
33.	    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
34.	    # Performing training
35.	    clf_entropy.fit(X_train, y_train)
36.	    return clf_entropy
37.
38.	# Function to calculate accuracy
39.	def cal_accuracy(y_test, y_pred):
40.	    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
41.	    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
42.	    print("Classification Report: ", classification_report(y_test, y_pred))
43.
44.	# Function to visualize the Decision Tree
45.	def visualize_decision_tree(clf, feature_names, class_names):
46.	    plt.figure(figsize=(12, 8))
47.	    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
48.	    plt.show()
49.
50.	# Driver code
51.	def main():
52.	    # Load Breast Cancer dataset
53.	    X, y = importdata()
54.	    X_train, X_test, y_train, y_test = splitdataset(X, y)
55.	    clf_gini = train_using_gini(X_train, X_test, y_train)
56.	    clf_entropy = train_using_entropy(X_train, X_test, y_train)
57.
58.	    # Results Using Gini Index
59.	    y_pred_gini = clf_gini.predict(X_test)
60.	    print("Results Using Gini Index:")
61.	    cal_accuracy(y_test, y_pred_gini)
62.
63.	    # Results Using Entropy
64.	    y_pred_entropy = clf_entropy.predict(X_test)
65.	    print("Results Using Entropy:")
66.	    cal_accuracy(y_test, y_pred_entropy)
67.
68.	# Calling main function
69.	if __name__ == "__main__":
70.	    main()
