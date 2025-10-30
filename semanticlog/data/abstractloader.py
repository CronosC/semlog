from abc import ABC, abstractmethod
from semanticlog.data.oced import OCEDStore

class AbstractLoader(ABC):
    
    @abstractmethod
    def __init__(self, path, config):
        """Initialize the data loader with a configuration dict and path."""
        self.path = path
        self.config = config
    
    @abstractmethod
    def get_oced(self) -> OCEDStore:
        """Returns the data as a OCEDStore object"""
        pass