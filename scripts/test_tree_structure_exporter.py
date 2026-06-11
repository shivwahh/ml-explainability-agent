from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from tools.tree_reader.tree_structure_exporter import (
    TreeStructureExporter
)

data = load_breast_cancer()

X = data.data
y = data.target

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

exporter = TreeStructureExporter(
    model=model,
    feature_names=data.feature_names,
    class_names=data.target_names
)

path = exporter.export_to_json(
    "artifacts/tree_structure.json"
)

print(f"Saved to: {path}")