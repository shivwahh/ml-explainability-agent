from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.tree import DecisionTreeClassifier

from tools.tree_reader.tree_structure_exporter import (
    TreeStructureExporter
)

data = fetch_covtype()

X = data.data
y = data.target

model = DecisionTreeClassifier(
    max_depth=None,
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