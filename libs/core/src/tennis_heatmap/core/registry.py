"""
tennis_heatmap.core.registry

Generic plugin registry implementing the Strategy + Registry pattern.
Detectors, trackers, and court detectors register themselves by a string key
using a decorator. The pipeline instantiates them purely by config name —
no import changes required when adding or swapping implementations.

Usage
-----
Registering a new detector::

    from tennis_heatmap.core.registry import DetectorRegistry
    from tennis_heatmap.core.interfaces.detector import BaseDetector

    @DetectorRegistry.register("my_custom_detector")
    class MyCustomDetector(BaseDetector):
        ...

Instantiating by name (in the pipeline factory)::

    detector = DetectorRegistry.build("my_custom_detector", **config_kwargs)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generic, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelRegistry(Generic[T]):
    """Generic registry mapping string keys to class constructors.

    Attributes:
        _registry: Internal dict of name → class.
        _registry_name: Human-readable name for error messages.
    """

    def __init__(self, registry_name: str) -> None:
        self._registry: Dict[str, Type[T]] = {}
        self._registry_name = registry_name

    def register(self, name: str):
        """Class decorator that registers a class under the given key.

        Example::

            @DetectorRegistry.register("yolov8_player")
            class YOLOv8PlayerDetector(BaseDetector):
                ...
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                logger.warning(
                    "[%s] Overwriting existing registration for key '%s'. "
                    "Old class: %s, New class: %s",
                    self._registry_name,
                    name,
                    self._registry[name].__name__,
                    cls.__name__,
                )
            self._registry[name] = cls
            logger.debug("[%s] Registered '%s' → %s", self._registry_name, name, cls.__name__)
            return cls

        return decorator

    def build(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered class by name.

        Args:
            name:   Registry key (e.g. ``"bytetrack"``).
            **kwargs: Passed directly to the class ``__init__``.

        Returns:
            An instance of the registered class.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"[{self._registry_name}] No implementation registered for '{name}'. "
                f"Available: {available}"
            )
        cls = self._registry[name]
        logger.info("[%s] Building '%s' (%s)", self._registry_name, name, cls.__name__)
        return cls(**kwargs)

    def list_available(self) -> list[str]:
        """Return a sorted list of all registered keys."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"ModelRegistry(name={self._registry_name!r}, keys={self.list_available()})"


# ---------------------------------------------------------------------------
# Global singleton registries — import these from other packages.
# ---------------------------------------------------------------------------

#: Registry for player/ball detector implementations.
DetectorRegistry: ModelRegistry = ModelRegistry("DetectorRegistry")

#: Registry for multi-object tracker implementations.
TrackerRegistry: ModelRegistry = ModelRegistry("TrackerRegistry")

#: Registry for court keypoint detector implementations.
CourtDetectorRegistry: ModelRegistry = ModelRegistry("CourtDetectorRegistry")
