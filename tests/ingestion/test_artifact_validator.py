from ingestion.artifact_validator import ArtifactValidator


def _model_metadata():
    return {
        "model_type": "DecisionTreeClassifier",
        "task_type": "classification",
        "feature_names": ["Age", "Income"],
    }


def _feature_records():
    return [
        {"name": "Age", "meaning": "Customer Age"},
        {"name": "Income", "meaning": "Monthly Income"},
    ]


def _context_sections():
    return {"Project Name": "Customer Churn Prediction"}


def test_valid_report_has_no_errors():
    report = ArtifactValidator(
        model_metadata=_model_metadata(),
        feature_records=_feature_records(),
        context_sections=_context_sections(),
    ).validate()

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["summary"]["has_model"] is True
    assert report["summary"]["dictionary_feature_count"] == 2


def test_missing_model_is_error():
    report = ArtifactValidator(
        feature_records=_feature_records(),
        context_sections=_context_sections(),
    ).validate()

    assert report["valid"] is False
    assert any("Model metadata" in error for error in report["errors"])


def test_undocumented_feature_warns():
    metadata = _model_metadata()
    metadata["feature_names"] = ["Age", "Income", "Tenure"]

    report = ArtifactValidator(
        model_metadata=metadata,
        feature_records=_feature_records(),
        context_sections=_context_sections(),
    ).validate()

    assert report["valid"] is True
    assert any("not described" in warn for warn in report["warnings"])


def test_missing_context_warns():
    report = ArtifactValidator(
        model_metadata=_model_metadata(),
        feature_records=_feature_records(),
    ).validate()

    assert any(
        "Project context" in warn for warn in report["warnings"]
    )
