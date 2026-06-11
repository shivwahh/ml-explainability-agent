from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from tools.tree_reader.decision_path_extractor import (
    DecisionPathExtractor
)

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Train model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# Create extractor
extractor = DecisionPathExtractor(
    model=model,
    feature_names=data.feature_names
)

# Explain first record
result = extractor.extract_path(
    X[0]
)

print(f"Leaf Node: {result['leaf_node']}")
print("\nDecision Path:\n")

for step in result["path"]:
    print(step["condition"])