from abc import ABC, abstractmethod

from semanticlog.data.oced import OCEDStore


class AbstractLoader(ABC):
    """Base class for all event-log loaders.

    Subclasses must implement ``get_oced()`` to read source data and return a
    populated :class:`~semanticlog.data.oced.OCEDStore`.
    """

    @abstractmethod
    def __init__(self, path, config):
        """Initialize the data loader with a configuration dict and path."""
        self.path = path
        self.config = config

    @abstractmethod
    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
