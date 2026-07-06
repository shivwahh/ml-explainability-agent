"""
artifact_validator.py

Validation for ingested artifacts.

Combines the normalized outputs of the model, data dictionary and
project context loaders and produces a structured validation report of
errors and warnings that downstream agents can act on.
"""


class ArtifactValidator:
    """
    Validate ingested artifacts and produce a structured report.

    The validator is tolerant: missing artifacts and inconsistencies are
    surfaced as errors or warnings rather than raising exceptions, so a
    caller can decide how to proceed.
    """

    def __init__(
        self,
        model_metadata: dict = None,
        feature_records: list = None,
        context_sections: dict = None,
    ):
        """
        Args:
            model_metadata: Output of
                :meth:`ModelArtifactLoader.extract_metadata`.
            feature_records: Output of
                :meth:`DataDictionaryLoader.to_records`.
            context_sections: Output of
                :meth:`ProjectContextLoader.to_sections`.
        """
        self.model_metadata = model_metadata
        self.feature_records = feature_records
        self.context_sections = context_sections

    def _dictionary_feature_names(self) -> set:
        """
        Return the set of feature names declared in the data dictionary.
        """
        names = set()

        for record in self.feature_records or []:
            name = record.get("name")

            if name is not None:
                names.add(str(name))

        return names

    def _validate_model(self, errors: list, warnings: list) -> None:
        """
        Validate the model metadata section.
        """
        if not self.model_metadata:
            errors.append("Model metadata is missing.")
            return

        if not self.model_metadata.get("model_type"):
            errors.append("Model type could not be determined.")

        if self.model_metadata.get("task_type") == "unknown":
            warnings.append(
                "Model task type could not be inferred "
                "(neither classifier nor regressor)."
            )

        if not self.model_metadata.get("feature_names"):
            warnings.append(
                "Model does not expose feature names; feature-level "
                "explanations may rely solely on the data dictionary."
            )

    def _validate_dictionary(self, errors: list, warnings: list) -> None:
        """
        Validate the data dictionary section.
        """
        if not self.feature_records:
            warnings.append("Data dictionary is missing or empty.")
            return

        missing_names = [
            index
            for index, record in enumerate(self.feature_records)
            if not record.get("name")
        ]

        if missing_names:
            warnings.append(
                f"{len(missing_names)} data dictionary row(s) have no "
                f"feature name."
            )

    def _validate_context(self, warnings: list) -> None:
        """
        Validate the project context section.
        """
        if not self.context_sections:
            warnings.append("Project context is missing or empty.")

    def _cross_validate(self, warnings: list) -> None:
        """
        Cross-check model features against the data dictionary.
        """
        if not self.model_metadata or not self.feature_records:
            return

        model_features = set(
            self.model_metadata.get("feature_names") or []
        )

        if not model_features:
            return

        dictionary_features = self._dictionary_feature_names()

        undocumented = model_features - dictionary_features

        if undocumented:
            warnings.append(
                f"{len(undocumented)} model feature(s) are not described "
                f"in the data dictionary: {sorted(undocumented)}."
            )

        unused = dictionary_features - model_features

        if unused:
            warnings.append(
                f"{len(unused)} data dictionary feature(s) are not used "
                f"by the model: {sorted(unused)}."
            )

    def validate(self) -> dict:
        """
        Run all validations and return a structured report.

        Returns:
            A dictionary with the keys ``valid`` (bool), ``errors``
            (list of str), ``warnings`` (list of str) and ``summary``
            (dict of presence flags).
        """
        errors = []
        warnings = []

        self._validate_model(errors, warnings)
        self._validate_dictionary(errors, warnings)
        self._validate_context(warnings)
        self._cross_validate(warnings)

        summary = {
            "has_model": bool(self.model_metadata),
            "has_data_dictionary": bool(self.feature_records),
            "has_project_context": bool(self.context_sections),
            "model_feature_count": len(
                (self.model_metadata or {}).get("feature_names") or []
            ),
            "dictionary_feature_count": len(self.feature_records or []),
        }

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }
