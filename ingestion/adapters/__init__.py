"""
adapters

Model framework adapter (plugin) architecture.

Each adapter knows how to recognize and normalize models from a
particular ML framework (for example scikit-learn trees and forests)
behind a single :class:`ModelAdapter` interface. An
:class:`AdapterRegistry` resolves the correct adapter automatically from
a loaded model, falling back to :class:`GenericModelAdapter` when no
specialized adapter matches.
"""

from ingestion.adapters.base import ModelAdapter
from ingestion.adapters.generic import GenericModelAdapter
from ingestion.adapters.sklearn_tree import SklearnTreeAdapter
from ingestion.adapters.xgboost_adapter import XGBoostAdapter
from ingestion.adapters.registry import AdapterRegistry, default_registry

__all__ = [
    "ModelAdapter",
    "GenericModelAdapter",
    "SklearnTreeAdapter",
    "XGBoostAdapter",
    "AdapterRegistry",
    "default_registry",
]
