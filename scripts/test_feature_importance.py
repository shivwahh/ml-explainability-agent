from sklearn.datasets import (
    load_breast_cancer
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from tools.tree_reader.feature_importance_extractor import (
    FeatureImportanceExtractor
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

# Extract importance
extractor = (
    FeatureImportanceExtractor(
        model,
        data.feature_names
    )
)

print(
    extractor.get_top_features(10)
)