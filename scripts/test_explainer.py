from sklearn.datasets import (
    load_breast_cancer,
    fetch_covtype
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
data = fetch_covtype()

X = data.data
y = data.target

# Train model
model = DecisionTreeClassifier(
    max_depth=None,
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