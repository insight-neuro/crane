import numpy as np
from torch import Tensor
from typing import Sequence
from ..data.structures import ChannelDict


def subset_electrodes[T: (np.ndarray, ChannelDict)](
    data: Tensor,
    channels: T,
    max_n_electrodes: int,
    *,
    batch_first: bool = True,
    subset: Sequence[int | str] | None = None,
) -> tuple[Tensor, T]:
    """
    Subset the electrodes to a random selection of electrodes. Consistent across a batch.
    If the number of electrodes is already less than or equal to the maximum, no subsetting is performed.

    Args:
        data (torch.Tensor): ([n_batch_size], n_electrodes, n_samples)
        channel_ids (np.ndarray | ChannelDict): Array of channel IDs or ChannelDict
        max_n_electrodes (int): Maximum number of electrodes to keep.
        batch_first (bool): Whether the batch dimension is the first dimension in the data tensor.
        subset (Sequence[int | str] | None): Optional list of electrode indices or IDs to subset to. If None, a random subset is chosen.

    Returns:
        tuple[torch.Tensor, np.ndarray | ChannelDict]: Subsetted data tensor and corresponding channel IDs.
    """
    # Determine indices to subset
    if not subset:
        if len(channels) <= max_n_electrodes:
            return data, channels
        indices = np.random.choice(len(channels), max_n_electrodes, replace=False)

    elif all(isinstance(e, str) for e in subset):
        if not isinstance(channels, ChannelDict):
            raise ValueError("channels must be a ChannelDict if subset contains channel IDs")

        id_set = set(subset)
        indices = [i for i, cid in enumerate(channels.id) if cid in id_set]
    else:
        indices = np.array(set(subset))

    # Subset data and channels
    if batch_first:
        data = data[:, indices, ...]
    else:
        data = data[indices, ...]
    channels = channels[indices]

    return data, channels
