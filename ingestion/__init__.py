"""
ingestion

Ingestion layer for the ML Explainability Agent.

Loads, normalizes and validates the three user-provided artifacts:

* Model artifact (.pkl / .joblib)
* Data dictionary (.xlsx / .csv)
* Project context (.md / .txt, extensible to .pdf / .docx)

into a single normalized ingestion output consumed by downstream
agents and tools.
"""

from ingestion.model_loader import ModelArtifactLoader
from ingestion.data_dictionary_loader import DataDictionaryLoader
from ingestion.project_context_loader import ProjectContextLoader
from ingestion.artifact_validator import ArtifactValidator

__all__ = [
    "ModelArtifactLoader",
    "DataDictionaryLoader",
    "ProjectContextLoader",
    "ArtifactValidator",
]
