import random

import torch
from torch.utils.data import Sampler

from crane.data.structures import CraneBatch


def collate_crane_batches(batch_list: list[CraneBatch]) -> CraneBatch:
    """
    Collate a list of CraneBatch objects into a single CraneBatch.
    Assumes all batches belong to the same brainset/subject/session.
    
    ..deprecated::
         `collate_crane_batches` is deprecated, use `torch_brain.datasets.collate` instead. It will be removed in a future release.

    Args:
        batch_list (list[CraneBatch]): List of CraneBatch objects to collate.

    Returns:
        CraneBatch: Collated batch.
    """
    data_tensors = [batch.data for batch in batch_list]
    collated_data = torch.stack(data_tensors, dim=1)  # shape: (n_samples, batch_size, n_channels)

    return CraneBatch(
        brainset=batch_list[0].brainset,
        subject=batch_list[0].subject,
        session=batch_list[0].session,
        citation=batch_list[0].citation,
        sampling_rate=batch_list[0].sampling_rate,
        data=collated_data,
        channels=batch_list[0].channels,
    )


class SessionBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch only contains samples from a single session.
    This is critical when different sessions have different numbers of channels or sampling rates.
    
    .. deprecated::
         `SessionBatchSampler` is deprecated, use `torch_brain.datasets.Sampler` instead. It will be removed in a future release.

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
            session_batches = [
                session_indices[i : i + self.batch_size]
                for i in range(0, len(session_indices), self.batch_size)
                if not self.drop_last or i + self.batch_size <= len(session_indices)
            ]
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
