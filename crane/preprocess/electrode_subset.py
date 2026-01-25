from typing import overload

import numpy as np
import torch

from ..data.structures import ChannelDict


@overload
def subset_electrodes(
    data: torch.Tensor,
    channels: np.ndarray,
    max_n_electrodes: int,
) -> tuple[torch.Tensor, np.ndarray]: ...


@overload
def subset_electrodes(
    data: torch.Tensor,
    channels: ChannelDict,
    max_n_electrodes: int,
) -> tuple[torch.Tensor, ChannelDict]: ...


def subset_electrodes(
    data: torch.Tensor,
    channels: np.ndarray | ChannelDict,
    max_n_electrodes: int,
) -> tuple[torch.Tensor, np.ndarray | ChannelDict]:
    """
    Subset the electrodes to a maximum number of electrodes.

    Args:
        data (torch.Tensor): iEEG data tensor of shape (n_samples, n_electrodes)
        channel_ids (np.ndarray | ChannelDict): Array of channel IDs or ChannelDict
        max_n_electrodes (int): Maximum number of electrodes to keep.

    """

    if len(data) > max_n_electrodes:  # Else no-op
        selected_indices = np.random.choice(len(channels), max_n_electrodes, replace=False)
        data = data[:, selected_indices, :]
        channels = channels[selected_indices]

    return data, channels
