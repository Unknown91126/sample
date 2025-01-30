import math
from collections import Counter

# Define a dataset
dataset = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"]
]

# Define feature names
features = ["Outlook", "Temperature", "Humidity", "Wind"]


# Calculate entropy
def calculate_entropy(data):
    label_counts = Counter(row[-1] for row in data)
    total_instances = len(data)
    entropy = 0
    for count in label_counts.values():
        probability = count / total_instances
        entropy -= probability * math.log2(probability)
    return entropy


# Calculate information gain
def calculate_information_gain(data, feature_index):
    total_entropy = calculate_entropy(data)
    total_instances = len(data)
    
    # Partition the data based on feature values
    partitions = {}
    for row in data:
        feature_value = row[feature_index]
        if feature_value not in partitions:
            partitions[feature_value] = []
        partitions[feature_value].append(row)
    
    # Calculate weighted entropy for each partition
    weighted_entropy = 0
    for subset in partitions.values():
        subset_probability = len(subset) / total_instances
        weighted_entropy += subset_probability * calculate_entropy(subset)
    
    # Information gain = total entropy - weighted entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain


# Build the decision tree
def build_decision_tree(data, feature_names):
    # Check if all examples belong to one class
    labels = [row[-1] for row in data]
    if len(set(labels)) == 1:
        return labels[0]
    
    # Check if no features are left to split
    if not feature_names:
        return Counter(labels).most_common(1)[0][0]  # Return the most common label
    
    # Select the best feature based on information gain
    best_feature_index = max(range(len(feature_names)), key=lambda i: calculate_information_gain(data, i))
    best_feature_name = feature_names[best_feature_index]
    
    # Create the root of the tree
    tree = {best_feature_name: {}}
    
    # Partition the data and recursively build subtrees
    feature_values = set(row[best_feature_index] for row in data)
    for value in feature_values:
        subset = [row for row in data if row[best_feature_index] == value]
        subtree = build_decision_tree(subset, feature_names[:best_feature_index] + feature_names[best_feature_index + 1:])
        tree[best_feature_name][value] = subtree
    
    return tree


# Print the decision tree
def print_decision_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}{key}")
            print_decision_tree(value, indent + "  ")
    else:
        print(f"{indent}--> {tree}")


# Build and display the decision tree
decision_tree = build_decision_tree(dataset, features)
print("Decision Tree:")
print_decision_tree(decision_tree)
