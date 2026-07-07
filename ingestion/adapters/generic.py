"""
generic.py

Generic fallback model adapter.

Used when no specialized framework adapter matches a loaded model. It
produces only the normalized metadata common to any estimator, without
raising, so unsupported models degrade gracefully.
"""

from ingestion.adapters.base import ModelAdapter


class GenericModelAdapter(ModelAdapter):
    """
    Fallback adapter that matches any estimator.

    This adapter emits the framework-agnostic normalized metadata from
    :class:`ModelAdapter` and adds no framework-specific fields.
    """

    name = "generic"

    @classmethod
    def matches(cls, estimator) -> bool:
        """
        Always match, acting as the universal fallback.
        """
        return True
