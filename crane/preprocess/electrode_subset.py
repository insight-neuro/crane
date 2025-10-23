from copy import deepcopy

import numpy as np


def electrode_subset_batch(batch: dict, max_n_electrodes: int, inplace: bool = False) -> dict:
    """
    Subset the electrodes to a maximum number of electrodes.

    Args:
        batch (dict): dictionary from SingleSessionDataset with keys:
            'ieeg': {'data': torch.Tensor[batch_size, n_channels, n_samples], 'sampling_rate': int}
            'channels': {'id': np.array}
            'metadata': dict
        max_n_electrodes (int): the maximum number of electrodes to subset to
        inplace (bool): if True, modify the batch dictionary in place

    Returns:
        batch: dictionary with subsetted data.
    """
    if not inplace:
        batch = deepcopy(batch)

    electrode_data = batch["ieeg"]["data"]  # shape: (n_electrodes, n_samples)
    electrode_labels = batch["channels"]["id"]

    if len(electrode_labels) > max_n_electrodes:  # Else no-op
        selected_indices = np.random.choice(len(electrode_labels), max_n_electrodes, replace=False)
        electrode_data = electrode_data[:, selected_indices, :]
        electrode_labels = electrode_labels[selected_indices]

    batch["ieeg"]["data"] = electrode_data
    batch["channels"]["id"] = electrode_labels

    return batch
