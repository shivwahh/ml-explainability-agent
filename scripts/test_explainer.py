from sklearn.datasets import (
    load_breast_cancer
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from tools.tree_reader.decision_path_extractor import (
    DecisionPathExtractor
)

from tools.explainers.tree_prediction_explainer import (
    TreePredictionExplainer
)

# Load data
data = load_breast_cancer()

X = data.data
y = data.target

# Train model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# Extract path
extractor = DecisionPathExtractor(
    model,
    data.feature_names
)

path = extractor.extract_path(
    X[0]
)

# Explain
explainer = (
    TreePredictionExplainer()
)

print(
    explainer.explain(path)
)