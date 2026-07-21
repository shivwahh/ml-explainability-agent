"""
registry.py

Adapter registry.

Resolves the correct :class:`ModelAdapter` for a loaded estimator,
trying registered specialized adapters in order and falling back to
:class:`GenericModelAdapter` when none match.
"""

from ingestion.adapters.generic import GenericModelAdapter
from ingestion.adapters.sklearn_tree import SklearnTreeAdapter
from ingestion.adapters.xgboost_adapter import XGBoostAdapter
from ingestion.adapters.lightgbm_adapter import LightGBMAdapter
from ingestion.adapters.catboost_adapter import CatBoostAdapter


DEFAULT_ADAPTERS = (
    SklearnTreeAdapter,
    XGBoostAdapter,
    LightGBMAdapter,
    CatBoostAdapter,
)


class AdapterRegistry:
    """
    Resolve a model adapter from a loaded estimator.

    Specialized adapters are tried in registration order; the first whose
    :meth:`ModelAdapter.matches` returns ``True`` is selected. When no
    specialized adapter matches, the generic fallback adapter is used.
    Adapters registered via :meth:`register` take priority over the
    built-in defaults.
    """

    def __init__(self, adapters=None):
        """
        Args:
            adapters: Optional iterable of adapter classes to use instead
                of the built-in defaults.
        """
        if adapters is None:
            adapters = DEFAULT_ADAPTERS

        self._adapters = list(adapters)

    def register(self, adapter_cls) -> None:
        """
        Register a specialized adapter with priority over existing ones.

        Args:
            adapter_cls: A :class:`ModelAdapter` subclass.
        """
        self._adapters.insert(0, adapter_cls)

    def resolve(self, estimator):
        """
        Return an adapter instance for the given estimator.

        Args:
            estimator: The unwrapped final estimator.

        Returns:
            A :class:`ModelAdapter` instance; the generic fallback when no
            specialized adapter matches.
        """
        for adapter_cls in self._adapters:
            if adapter_cls.matches(estimator):
                return adapter_cls()

        return GenericModelAdapter()


def default_registry() -> AdapterRegistry:
    """
    Return a fresh registry configured with the built-in adapters.
    """
    return AdapterRegistry()
