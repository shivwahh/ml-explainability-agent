"""
project_knowledge.py

Unified project knowledge object.

Composes the normalized outputs of the three ingestion loaders
(:class:`ModelArtifactLoader`, :class:`DataDictionaryLoader`,
:class:`ProjectContextLoader`) together with the
:class:`ArtifactValidator` into a single normalized knowledge object
that downstream agents and tools can consume from one entry point.

Any missing artifact is handled gracefully: the corresponding section is
left empty and surfaced through the validation report rather than raising.
"""

from ingestion.model_loader import ModelArtifactLoader
from ingestion.data_dictionary_loader import DataDictionaryLoader
from ingestion.project_context_loader import ProjectContextLoader
from ingestion.artifact_validator import ArtifactValidator


class ProjectKnowledge:
    """
    Aggregate ingested artifacts into one normalized knowledge object.

    The aggregator runs each loader only when its artifact path is
    provided. Loader paths are optional so a caller can build partial
    knowledge (for example, a model with no data dictionary yet). The
    validation report captures any resulting gaps or inconsistencies.
    """

    def __init__(
        self,
        model_path: str = None,
        dictionary_path: str = None,
        context_path: str = None,
    ):
        """
        Args:
            model_path: Path to the serialized model artifact, or
                ``None`` when no model is available.
            dictionary_path: Path to the data dictionary file, or
                ``None`` when no data dictionary is available.
            context_path: Path to the project context file, or ``None``
                when no project context is available.
        """
        self.model_path = model_path
        self.dictionary_path = dictionary_path
        self.context_path = context_path

        self.model_metadata = None
        self.feature_records = []
        self.context_sections = {}
        self.validation = None
        self.model = None

    def _load_model_metadata(self) -> dict:
        """
        Load normalized model metadata when a model path is provided.

        The live model object is retained on :attr:`model` so downstream
        consumers (for example agent nodes) can reuse it without loading
        the artifact a second time.

        Returns:
            The model metadata dictionary, or ``None`` when no model
            path was supplied.
        """
        if not self.model_path:
            return None

        loader = ModelArtifactLoader(self.model_path)
        metadata = loader.extract_metadata()
        self.model = loader.model

        return metadata

    def _load_feature_records(self) -> list:
        """
        Load normalized feature records when a dictionary path is given.

        Returns:
            A list of feature records, empty when no dictionary path was
            supplied.
        """
        if not self.dictionary_path:
            return []

        return DataDictionaryLoader(self.dictionary_path).to_records()

    def _load_context_sections(self) -> dict:
        """
        Load project context sections when a context path is provided.

        Returns:
            A mapping of section heading to body, empty when no context
            path was supplied.
        """
        if not self.context_path:
            return {}

        return ProjectContextLoader(self.context_path).to_sections()

    def build(self) -> dict:
        """
        Run all available loaders and the validator from one entry point.

        Returns:
            A normalized knowledge object with the keys:

            * ``model`` - model metadata dictionary, or ``None``.
            * ``features`` - list of feature records.
            * ``context`` - mapping of context section headings to bodies.
            * ``validation`` - the structured validation report.
        """
        self.model_metadata = self._load_model_metadata()
        self.feature_records = self._load_feature_records()
        self.context_sections = self._load_context_sections()

        self.validation = ArtifactValidator(
            model_metadata=self.model_metadata,
            feature_records=self.feature_records,
            context_sections=self.context_sections,
        ).validate()

        return {
            "model": self.model_metadata,
            "features": self.feature_records,
            "context": self.context_sections,
            "validation": self.validation,
        }
