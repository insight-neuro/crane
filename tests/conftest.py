import math

import numpy as np
import pytest
import torch


@pytest.fixture
def make_batch():
    """Create a synthetic EEG batch for testing.

    Args:
        batch_size (int, optional): Number of samples in the batch. Defaults to 2.
        n_elec (int, optional): Number of electrodes. Defaults to 3.
        n_samples (int, optional): Number of samples per electrode. Defaults to 1024.
        sampling_rate (int, optional): Sampling rate in Hz. Defaults to 256.

    Returns:
        dict:
        A dictionary containing the synthetic EEG data and metadata:
            {
                "ieeg": {
                    "data": Tensor of shape (batch_size, n_electrodes, n_samples),
                    "sampling_rate": sampling_rate,
                },
                "channels": {
                    "id": np.array([f"chan{i}" for i in range(n_elec)]),
                },
                "metadata": {},
            }
    """

    def _make_batch(batch_size=2, n_elec=3, n_samples=1024, sampling_rate=256) -> dict:
        # simple reproducible signal (sum of low-freq sines + noise)
        torch.manual_seed(0)
        t = torch.arange(n_samples) / sampling_rate
        sig = 0.8 * torch.sin(2 * math.pi * 10 * t) + 0.3 * torch.sin(2 * math.pi * 20 * t) + 0.05 * torch.randn(n_samples)
        x = sig.repeat(batch_size, n_elec, 1).clone()
        return {
            "ieeg": {
                "data": x.to(dtype=torch.float32),
                "sampling_rate": sampling_rate,
            },
            "channels": {
                "id": np.array([f"chan{i}" for i in range(n_elec)]),
            },
            "metadata": {},
        }

    return _make_batch
