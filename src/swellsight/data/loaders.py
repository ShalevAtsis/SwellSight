"""
Data loading utilities for training and evaluation.

Creates PyTorch DataLoaders with appropriate batching and sampling strategies.
"""

from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

from .datasets import WaveDataset, DataSplit

def create_data_loaders(
    train_dataset: WaveDataset,
    val_dataset: WaveDataset,
    test_dataset: Optional[WaveDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        use_weighted_sampling: Whether to use weighted sampling for class balance
        
    Returns:
        Dictionary with train, val, and optionally test data loaders
    """
    loaders = {}
    
    # Training loader with optional weighted sampling
    if use_weighted_sampling:
        sampler = create_weighted_sampler(train_dataset)
        shuffle = False  # Don't shuffle when using sampler
    else:
        sampler = None
        shuffle = True
    
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    
    # Validation loader
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Test loader if provided
    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    return loaders

def create_weighted_sampler(dataset: WaveDataset) -> WeightedRandomSampler:
    """Create weighted sampler for balanced training.
    
    Args:
        dataset: Wave dataset to create sampler for
        
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    # Calculate class weights based on dataset balance
    # TODO: Implement proper weight calculation based on dataset statistics
    
    # Placeholder implementation
    num_samples = len(dataset)
    weights = torch.ones(num_samples)  # Equal weights for now
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )

def create_inference_loader(
    dataset: WaveDataset,
    batch_size: int = 1,
    num_workers: int = 2
) -> DataLoader:
    """Create data loader for inference/evaluation.
    
    Args:
        dataset: Dataset for inference
        batch_size: Batch size (typically 1 for inference)
        num_workers: Number of worker processes
        
    Returns:
        DataLoader configured for inference
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )