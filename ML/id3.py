# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data  # Features: sepal length, sepal width, petal length, petal width
y = data.target  # Target: iris species (0, 1, 2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=53)

# Create a Decision Tree Classifier with the ID3 algorithm (criterion='entropy')
clf = DecisionTreeClassifier(criterion='entropy', random_state=56)
clf.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Decision Tree: {accuracy:.2f}")

# Display the decision tree as text
tree_rules = export_text(clf, feature_names=data.feature_names)
print("\nDecision Tree Rules:\n")
print(tree_rules)

# Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree (ID3 Algorithm)")
plt.show()
