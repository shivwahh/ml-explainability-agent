import joblib
from sklearn.tree import DecisionTreeClassifier

from ingestion.project_knowledge import ProjectKnowledge


def _make_model():
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 1, 0]
    model.fit(X, y)
    return model


def _write_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_make_model(), model_path)
    return model_path


def _write_dictionary(tmp_path):
    path = tmp_path / "data_dictionary.csv"
    path.write_text(
        "Column,Meaning\n"
        "x0,First feature\n"
        "x1,Second feature\n",
        encoding="utf-8",
    )
    return path


def _write_context(tmp_path):
    path = tmp_path / "project_context.md"
    path.write_text(
        "# Project Name\n\n"
        "Toy Classifier\n",
        encoding="utf-8",
    )
    return path


def test_build_composes_all_sections(tmp_path):
    knowledge = ProjectKnowledge(
        model_path=str(_write_model(tmp_path)),
        dictionary_path=str(_write_dictionary(tmp_path)),
        context_path=str(_write_context(tmp_path)),
    ).build()

    assert knowledge["model"]["model_type"] == "DecisionTreeClassifier"
    assert len(knowledge["features"]) == 2
    assert knowledge["context"]["Project Name"] == "Toy Classifier"
    assert knowledge["validation"]["summary"]["has_model"] is True
    assert knowledge["validation"]["summary"]["has_data_dictionary"] is True
    assert knowledge["validation"]["summary"]["has_project_context"] is True


def test_build_with_no_artifacts_is_graceful():
    knowledge = ProjectKnowledge().build()

    assert knowledge["model"] is None
    assert knowledge["features"] == []
    assert knowledge["context"] == {}
    assert knowledge["validation"]["valid"] is False
    assert any(
        "Model metadata" in error
        for error in knowledge["validation"]["errors"]
    )


def test_build_with_only_model(tmp_path):
    knowledge = ProjectKnowledge(
        model_path=str(_write_model(tmp_path)),
    ).build()

    assert knowledge["model"] is not None
    assert knowledge["features"] == []
    assert knowledge["context"] == {}
    assert knowledge["validation"]["summary"]["has_model"] is True
    assert knowledge["validation"]["summary"]["has_data_dictionary"] is False


def test_build_populates_instance_attributes(tmp_path):
    builder = ProjectKnowledge(
        model_path=str(_write_model(tmp_path)),
        dictionary_path=str(_write_dictionary(tmp_path)),
    )
    builder.build()

    assert builder.model_metadata is not None
    assert len(builder.feature_records) == 2
    assert builder.validation is not None
