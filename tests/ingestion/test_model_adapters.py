import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ingestion.adapters import (
    AdapterRegistry,
    GenericModelAdapter,
    SklearnTreeAdapter,
    default_registry,
)
from ingestion.model_loader import ModelArtifactLoader


def _fit(model):
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 1, 0]
    model.fit(X, y)
    return model


def test_registry_resolves_tree_adapter():
    tree = _fit(DecisionTreeClassifier(max_depth=3, random_state=42))

    adapter = default_registry().resolve(tree)

    assert isinstance(adapter, SklearnTreeAdapter)


def test_registry_resolves_forest_adapter():
    forest = _fit(RandomForestClassifier(n_estimators=5, random_state=42))

    adapter = default_registry().resolve(forest)

    assert isinstance(adapter, SklearnTreeAdapter)


def test_registry_falls_back_to_generic():
    model = _fit(LogisticRegression())

    adapter = default_registry().resolve(model)

    assert isinstance(adapter, GenericModelAdapter)


def test_tree_adapter_adds_tree_details():
    tree = _fit(DecisionTreeRegressor(max_depth=2, random_state=42))

    metadata = SklearnTreeAdapter().extract(tree, tree, "model.joblib")

    assert metadata["adapter"] == "sklearn_tree"
    assert metadata["framework"] == "scikit-learn"
    assert metadata["tree_details"]["is_ensemble"] is False
    assert "node_count" in metadata["tree_details"]


def test_forest_adapter_reports_n_estimators():
    forest = _fit(RandomForestClassifier(n_estimators=7, random_state=42))

    metadata = SklearnTreeAdapter().extract(forest, forest, "model.joblib")

    assert metadata["tree_details"]["is_ensemble"] is True
    assert metadata["tree_details"]["n_estimators"] == 7


def test_generic_adapter_produces_normalized_fields():
    model = _fit(LogisticRegression())

    metadata = GenericModelAdapter().extract(model, model, "model.joblib")

    assert metadata["adapter"] == "generic"
    assert metadata["task_type"] == "classification"
    assert metadata["n_features"] == 2
    assert "framework" not in metadata


def test_custom_adapter_takes_priority():
    class AlwaysAdapter(SklearnTreeAdapter):
        name = "always"

        @classmethod
        def matches(cls, estimator):
            return True

    registry = AdapterRegistry()
    registry.register(AlwaysAdapter)

    adapter = registry.resolve(_fit(LogisticRegression()))

    assert isinstance(adapter, AlwaysAdapter)


def test_loader_uses_tree_adapter(tmp_path):
    tree = _fit(DecisionTreeClassifier(max_depth=3, random_state=42))
    model_path = tmp_path / "model.joblib"
    joblib.dump(tree, model_path)

    metadata = ModelArtifactLoader(str(model_path)).extract_metadata()

    assert metadata["adapter"] == "sklearn_tree"
    assert metadata["model_type"] == "DecisionTreeClassifier"
    assert metadata["task_type"] == "classification"
    assert metadata["framework"] == "scikit-learn"
