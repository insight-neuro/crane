from collections.abc import Sequence
from typing import cast

import torch

from crane.featurizer import CraneFeature


def subset_electrodes(
    data: CraneFeature,
    *,
    inplace: bool = False,
    max_n_electrodes: int | None = None,
    subset: Sequence[int] | Sequence[str] | None = None,
) -> CraneFeature:
    """
    Subset channel electrodes, consistent across a batch.

    Exactly one of `max_n_electrodes` or `subset` must be provided.

    Args:
        data (CraneFeature): The input feature data.
        max_n_electrodes (int): Maximum number of randomly selected electrodes to subset to.
        subset (Sequence[int | str] | None): Optional list of electrode indices or IDs to subset to. If None, a random subset is chosen.

    Returns:
        CraneFeature: The subsetted feature data.
    """

    if (subset is None) == (max_n_electrodes is None):
        raise ValueError("Provide exactly one of max_n_electrodes or subset")

    if not inplace:
        data = data.copy()

    # If no subset provided, randomly select electrodes up to max_n_electrodes
    if subset is None:
        n = len(data)
        if max_n_electrodes is None or n <= max_n_electrodes:
            return data
        indices = torch.randperm(n, device=data.device)[:max_n_electrodes]

    # If string IDs provided, map to indices using channel_labels
    elif all(isinstance(e, str) for e in subset):
        subset = cast(Sequence[str], subset)

        id_to_idx = {cid: i for i, cid in enumerate(data.channel_labels)}
        indices = torch.tensor(
            [id_to_idx[e] for e in subset if e in id_to_idx],
            dtype=torch.long,
            device=data.device,
        )

    # If integer indices provided, use directly
    else:
        indices = torch.as_tensor(subset, dtype=torch.long, device=data.device)

    data.signals = torch.index_select(data.signals, data.channel_dim, indices)
    data.channel_coordinates = torch.index_select(data.channel_coordinates, 0, indices)
    data.channel_labels = [data.channel_labels[i] for i in indices.tolist()]

    return data
