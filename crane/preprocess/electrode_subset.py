import numpy as np
from torch import Tensor

from ..data.structures import ChannelDict


def subset_electrodes[T: (np.ndarray, ChannelDict)](
    data: Tensor,
    channels: T,
    max_n_electrodes: int,
    batch_first: bool = True,
) -> tuple[Tensor, T]:
    """
    Subset the electrodes to a random selection of electrodes. Consistent across a batch.
    If the number of electrodes is already less than or equal to the maximum, no subsetting is performed.

    Args:
        data (torch.Tensor): ([n_batch_size], n_electrodes, n_samples)
        channel_ids (np.ndarray | ChannelDict): Array of channel IDs or ChannelDict
        max_n_electrodes (int): Maximum number of electrodes to keep.
        batch_first (bool): Whether the batch dimension is the first dimension in the data tensor.

    Returns:
        tuple[torch.Tensor, np.ndarray | ChannelDict]: Subsetted data tensor and corresponding channel IDs.
    """

    if len(data) > max_n_electrodes:  # Else no-op
        selected_indices = np.random.choice(len(channels), max_n_electrodes, replace=False)
        if batch_first:
            data = data[:, selected_indices, ...]
        else:
            data = data[selected_indices, ...]
        channels = channels[selected_indices]

    return data, channels
