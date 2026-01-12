"""
Training callbacks and monitoring for wave analysis training.

Implements callbacks for logging, checkpointing, and early stopping.
"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod

class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_start(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the start of each batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of each batch."""
        pass

class TrainingCallbacks:
    """Collection of training callbacks."""
    
    def __init__(self, callbacks: List[TrainingCallback] = None):
        """Initialize callback collection.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: TrainingCallback):
        """Add a callback to the collection."""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        """Call all callbacks at epoch start."""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Call all callbacks at epoch end."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_start(self, batch: int, logs: Dict[str, Any] = None):
        """Call all callbacks at batch start."""
        for callback in self.callbacks:
            callback.on_batch_start(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Call all callbacks at batch end."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)