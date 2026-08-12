"""The interface every loader implements."""

from abc import ABC, abstractmethod

from semanticlog.data.oced import OCEDStore


class Loader(ABC):
    """Turns one source file into an OCED store.

    A loader reads and reshapes. It does not clean names, so its output is
    still in the vocabulary of the source and has to pass through
    `clean_oced_data` before a build.
    """

    @abstractmethod
    def __init__(self, path, config):
        """Record where to read from and how.

        Args:
            path: File to read.
            config: Loader-specific options. Each loader documents its own
                keys and fills in the ones it needs.
        """
        self.path = path
        self.config = config

    @abstractmethod
    def get_oced(self) -> OCEDStore:
        """Read the file and return it as a store."""
