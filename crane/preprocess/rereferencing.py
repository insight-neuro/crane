from collections import defaultdict

import numpy as np
import torch


def _get_all_laplacian_electrodes(electrode_labels: list[str]) -> tuple[list[str], dict]:
    """
    Get all laplacian electrodes for a given subject. This function is originally from
    https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)

    Modified to preserve original electrode label formatting (e.g., PT02 stays PT02, not PT2)

    Args:
        electrode_labels (list of str): List of electrode labels for the subject.

    Returns:
        laplacian_labels,neighbors (tuple[list[str], dict]): A tuple containing:
            laplacian_labels (list of str): List of electrode labels that are laplacian electrodes.
            neighbors (dict): Dictionary mapping each laplacian electrode label to a list of its neighboring electrode labels.
    """

    def _stem_electrode_name(name: str):
        # names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
        # names look like 'T1b2
        found_stem_end = False
        stem, num = [], []
        for c in reversed(name):
            if c.isalpha():
                found_stem_end = True
            if found_stem_end:
                stem.append(c)
            else:
                num.append(c)
        return "".join(reversed(stem)), int("".join(reversed(num)))

    def _has_neighbors(stem: tuple[str, int], stems: list) -> bool:
        (x, y) = stem
        return ((x, y + 1) in stems) or ((x, y - 1) in stems)

    def _get_neighbors_original_labels(label: str, label_to_stem: dict, stem_to_labels: dict) -> list[str]:
        stem_prefix, stem_num = label_to_stem[label]
        neighbor_nums = [stem_num + 1, stem_num - 1]
        neighbors = []
        for num in neighbor_nums:
            key = (stem_prefix, num)
            if key in stem_to_labels:
                neighbors.extend(stem_to_labels[key])
        return neighbors

    # Create mappings to preserve original labels
    label_to_stem = {label: _stem_electrode_name(label) for label in electrode_labels}
    stems = list(label_to_stem.values())

    # Create reverse mapping: stem -> list of original labels with that stem
    stem_to_labels = defaultdict(list)
    for label, stem in label_to_stem.items():
        stem_to_labels[stem].append(label)

    # Find laplacian electrodes (those with neighbors)
    laplacian_labels = [label for label, stem in label_to_stem.items() if _has_neighbors(stem, stems)]

    # Build neighbors dict using original labels
    neighbors = {
        label: _get_neighbors_original_labels(label, label_to_stem, stem_to_labels) for label in laplacian_labels
    }

    return laplacian_labels, neighbors


def _rereference_electrodes(
    electrode_data: torch.Tensor,
    electrode_labels: list[str],
    remove_non_laplacian: bool = True,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """
    Rereference the neural data using the laplacian method (subtract the mean of the neighbors, as determined by the electrode labels)

    Args:
        electrode_data (torch.Tensor): torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels (list of str): list of electrode labels
        remove_non_laplacian (bool): if True, remove the non-laplacian electrodes from the data; if false, keep them without rereferencing

    Returns:
        rereferenced_data,rereferenced_labels (tuple[torch.Tensor, list[str]]): A tuple containing:
            rereferenced_data (torch.Tensor): torch tensor of shape (batch_size, n_electrodes_rereferenced, n_samples) or (n_electrodes_rereferenced, n_samples)
            rereferenced_labels (list of str): list of electrode labels of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
    """
    batch_unsqueeze = False
    if len(electrode_data.shape) == 2:
        batch_unsqueeze = True
        electrode_data = electrode_data.unsqueeze(0)

    laplacian_electrodes, laplacian_neighbors = _get_all_laplacian_electrodes(electrode_labels)
    laplacian_neighbor_indices = {
        laplacian_electrode_label: [electrode_labels.index(neighbor_label) for neighbor_label in neighbors]
        for laplacian_electrode_label, neighbors in laplacian_neighbors.items()
    }

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    rereferenced_data = torch.zeros(
        (batch_size, rereferenced_n_electrodes, n_samples),
        dtype=electrode_data.dtype,
        device=electrode_data.device,
    )

    electrode_i = 0
    original_electrode_indices = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        if electrode_label in laplacian_electrodes:
            rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i] - torch.mean(
                electrode_data[:, laplacian_neighbor_indices[electrode_label]], dim=1
            )
            original_electrode_indices.append(original_electrode_index)
            electrode_i += 1
        else:
            if remove_non_laplacian:
                continue  # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i]
                original_electrode_indices.append(original_electrode_index)
                electrode_i += 1

    if batch_unsqueeze:
        rereferenced_data = rereferenced_data.squeeze(0)

    return (
        rereferenced_data,
        laplacian_electrodes if remove_non_laplacian else electrode_labels,
        original_electrode_indices,
    )


def laplacian_rereference(batch: dict, remove_non_laplacian: bool = True, inplace: bool = False):
    """
    Apply Laplacian rereferencing to a batch of neural data (subtract the mean of the neighbors, as determined by the electrode labels)

    Args:
        batch (dict): dictionary from SingleSessionDataset with keys:
            'ieeg': {'data': torch.Tensor[batch_size, n_channels, n_samples], 'sampling_rate': int}
            'channels': {'id': np.array}
            'metadata': dict
        remove_non_laplacian (bool, default=True): if True, remove the non-laplacian electrodes from the data
        inplace (bool, default=False): if True, modify the batch dictionary in place

    Returns:
        batch (dict): dictionary with rereferenced data:
            'ieeg': {'data': torch.Tensor[batch_size, n_channels_rereferenced, n_samples], 'sampling_rate': int}
            'channels': {'id': np.array}
            'metadata': dict
    """
    assert inplace, "laplacian_rereference_batch currently only supports inplace=True"

    electrode_data = batch["ieeg"]["data"]  # shape: (n_electrodes, n_samples)
    electrode_labels = batch["channels"]["id"].tolist()

    # _rereference_electrodes expects (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
    rereferenced_data, rereferenced_labels, original_electrode_indices = _rereference_electrodes(
        electrode_data, electrode_labels, remove_non_laplacian=remove_non_laplacian
    )

    # Update batch with rereferenced data
    batch["ieeg"]["data"] = rereferenced_data
    batch["channels"]["id"] = np.array(rereferenced_labels)

    return batch
