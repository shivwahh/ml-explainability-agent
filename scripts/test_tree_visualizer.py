from sklearn.datasets import (
    load_breast_cancer
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from tools.tree_reader.tree_visualizer import (
    TreeVisualizer
)

data = load_breast_cancer()

X = data.data
y = data.target

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

visualizer = TreeVisualizer(
    model=model,
    feature_names=data.feature_names,
    class_names=data.target_names
)

visualizer.display()

visualizer.save(
    "artifacts/visualizations/tree.png"
)

print(
    "Tree saved successfully."
)