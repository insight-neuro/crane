import random

import torch


class SessionBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that ensures each batch only contains samples from a single session.
    This is critical when different sessions have different numbers of channels or sampling rates.

    Args:
        dataset_sizes (list): List of dataset sizes for each session
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle indices within sessions and batch order
        drop_last (bool): Whether to drop incomplete batches
    """

    def __init__(
        self,
        dataset_sizes: list,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Create batches for each session
        all_batches = []
        start_idx = 0

        for size in self.dataset_sizes:
            # Create indices for this session
            session_indices = list(range(start_idx, start_idx + size))
            if self.shuffle:
                random.shuffle(session_indices)

            # Create batches for this session
            session_batches = [session_indices[i : i + self.batch_size] for i in range(0, len(session_indices), self.batch_size) if not self.drop_last or i + self.batch_size <= len(session_indices)]
            all_batches.extend(session_batches)
            start_idx += size

        # Shuffle the order of batches across sessions if needed
        if self.shuffle:
            random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        if self.drop_last:
            return sum(size // self.batch_size for size in self.dataset_sizes)
        return sum((size + self.batch_size - 1) // self.batch_size for size in self.dataset_sizes)
