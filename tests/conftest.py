import math
from collections.abc import Callable

import pytest
import torch

from crane import CraneFeature


@pytest.fixture
def make_batch() -> Callable[..., CraneFeature]:
    """Create a synthetic EEG batch for testing.

    Args:
        batch_size (int, optional): Number of samples in the batch. Defaults to 2.
        n_elec (int, optional): Number of electrodes. Defaults to 3.
        n_samples (int, optional): Number of samples per electrode. Defaults to 1024.
        sampling_rate (int, optional): Sampling rate in Hz. Defaults to 256.

    Returns:
        CraneFeature: A batch of synthetic EEG data with specified dimensions and metadata.
    """

    def _make_batch(
        batch_size: int = 2, n_elec: int = 3, n_samples: int = 1024, sampling_rate: int = 256
    ) -> CraneFeature:
        # simple reproducible signal (sum of low-freq sines + noise)
        torch.manual_seed(0)
        t = torch.arange(n_samples) / sampling_rate
        sig = (
            0.8 * torch.sin(2 * math.pi * 10 * t)
            + 0.3 * torch.sin(2 * math.pi * 20 * t)
            + 0.05 * torch.randn(n_samples)
        )
        x = sig.repeat(batch_size, n_elec, 1).clone()
        if batch_size == 1:
            x = x.squeeze(0)  # remove batch dim for unbatched case
        return CraneFeature(
            brainset="test_brainset",
            subject="test_subject",
            session="test_session",
            signals=x,
            channel_labels=[f"chan{i}" for i in range(n_elec)],
            channel_coordinates=torch.randn(n_elec, 3),
            sampling_rate=sampling_rate,
        )

    return _make_batch
