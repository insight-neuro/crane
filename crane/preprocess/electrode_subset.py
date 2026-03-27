from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.structures import ChannelDict


def subset_electrodes[T: (np.ndarray, ChannelDict)](
    data: Tensor,
    channels: T,
    *,
    max_n_electrodes: int | None = None,
    subset: Sequence[int] | Sequence[str] | None = None,
) -> tuple[Tensor, T]:
    """
    Subset the electrodes to a random selection of electrodes. Consistent across a batch.

    One of `max_n_electrodes` or `subset` must be provided.

    Args:
        data (torch.Tensor): ([n_batch_size], n_electrodes, n_samples)
        channel_ids (np.ndarray | ChannelDict): Array of channel IDs or ChannelDict.
        max_n_electrodes (int): Maximum number of randomly selected electrodes to subset to.
        subset (Sequence[int | str] | None): Optional list of electrode indices or IDs to subset to. If None, a random subset is chosen.

    Returns:
        tuple[torch.Tensor, np.ndarray | ChannelDict]: Subsetted data tensor and corresponding channel IDs.
    """

    if (subset is None) == (max_n_electrodes is None):
        raise ValueError("Provide exactly one of max_n_electrodes or subset")

    device = data.device

    # If no subset provided, randomly select electrodes up to max_n_electrodes
    if subset is None:
        n = len(channels)
        if max_n_electrodes is None or n <= max_n_electrodes:
            return data, channels
        indices = torch.randperm(n, device=device)[:max_n_electrodes]

    # If string IDs provided, map to indices using channels.id
    elif all(isinstance(e, str) for e in subset):
        if not isinstance(channels, ChannelDict):
            raise ValueError("channels must be ChannelDict when subset contains IDs")

        id_to_idx = {cid: i for i, cid in enumerate(channels.id)}
        indices = torch.tensor(
            [id_to_idx[e] for e in subset if e in id_to_idx],
            dtype=torch.long,
            device=device,
        )

    # If integer indices provided, use directly
    else:
        indices = torch.as_tensor(subset, dtype=torch.long, device=device)

    # Apply subset to data and channels
    dim = 1 if data.ndim == 3 else 0
    data = torch.index_select(data, dim, indices)
    
    idx_np = indices.detach().cpu().numpy()
    channels = channels[idx_np]
    
    return data, channels
