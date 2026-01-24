from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class Spectrogram(nn.Module):
    """
    Spectrogram Preprocessor, computes the spectrogram of iEEG data using STFT.

    Args:
        segment_length (float): The length of each segment (in seconds) for the STFT.
        p_overlap (float): The proportion of overlap between segments (between 0 and 1).
        *,
        min_frequency (float): The minimum frequency (in Hz) to include in the spectrogram.
        max_frequency (float): The maximum frequency (in Hz) to include in the spectrogram.
        window (Literal["hann", "boxcar"]): The type of window to use for the STFT.
        remove_line_noise (bool): Whether to remove line noise frequencies (e.g., 50/60 Hz).
        output_dim (int, default=-1): The dimension of the output features. If -1, the output feature dimension will be the same as the number of frequency bins. Otherwise, they will be projected to this dimension.
    """

    def __init__(
        self,
        segment_length: float,
        p_overlap: float,
        *,
        min_frequency: float,
        max_frequency: float,
        window: Literal["hann", "boxcar"],
        remove_line_noise: bool,
        output_dim: int = -1,
    ):
        super().__init__()
        self.segment_length = segment_length
        self.p_overlap = p_overlap

        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.window = window
        self.remove_line_noise = remove_line_noise

        self.output_dim = output_dim

        # from https://docs.pytorch.org/docs/stable/generated/torch.fft.rfftfreq.html
        # if n is nperseg, and d is 1/sampling_rate, then f = torch.arange((n + 1) // 2) / (d * n)
        # note: nperseg is always going to be even, so it simplifies to torch.arange(n/2) / n * sampling_rate
        # note: n = sampling_rate * tperseg, so it simplifies to torch.arange(sampling_rate * tperseg / 2) / tperseg
        #    which is a list that goes from 0 to sampling_rate / 2 in increments of sampling_rate / nperseg = 1 / tperseg
        # so max frequency bin is max_frequency * tperseg + 1 (adding one to make the endpoint inclusive)
        self.max_frequency_bin = round(self.max_frequency * self.segment_length + 1)
        self.min_frequency_bin = round(self.min_frequency * self.segment_length)
        self.n_freqs = self.max_frequency_bin - self.min_frequency_bin

        # Transform FFT output to match expected output dimension
        self.output_transform = nn.Identity() if self.output_dim == -1 else nn.Linear(self.n_freqs, self.output_dim)

        if self.remove_line_noise:
            example_sampling_rate = 2048
            nperseg = round(self.segment_length * example_sampling_rate)
            freq_bins = torch.fft.rfftfreq(nperseg, d=1.0 / example_sampling_rate)[
                self.min_frequency_bin : self.max_frequency_bin
            ]  # Calculate frequency bins (in Hz)
            self.line_noise_mask = self._compute_line_noise_mask(
                freq_bins=freq_bins, line_noise_freqs=[50, 60], margin=2.0
            )
        else:
            self.line_noise_mask = None

    def forward(self, data: torch.Tensor, sampling_rate: int, z_score: bool = True) -> torch.Tensor:
        """
        Perform the forward pass of the SpectrogramPreprocessor.

        Args:
            data (torch.Tensor): A tensor of shape (batch_size, n_electrodes, n_samples) representing the iEEG data.
            sampling_rate (int): An integer representing the sampling rate of the iEEG data.
            z_score (bool): Whether to apply z-score normalization to the spectrogram. Default is True.

        Returns:
            torch.Tensor: The processed spectrogram with shape (batch_size, n_electrodes, n_timebins, n_freqs or output_dim).
        """
        batch_size, n_electrodes, n_samples = data.shape

        # Reshape for STFT
        x = data.reshape(batch_size * n_electrodes, -1)
        x = x.to(dtype=torch.float32)  # Convert to float32 for STFT

        # STFT parameters
        nperseg = round(self.segment_length * sampling_rate)
        noverlap = round(self.p_overlap * nperseg)
        hop_length = nperseg - noverlap

        window = {
            "hann": torch.hann_window,
            "boxcar": torch.ones,
        }[self.window](nperseg, device=x.device)

        # Compute STFT
        x = torch.stft(
            x,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            return_complex=True,
            normalized=False,
            center=True,
        )

        # Take magnitude
        x = torch.abs(x)

        # Calculate frequency bins (in Hz)
        # These represent the center frequency of each frequency bin in the spectrogram
        freq_bins = torch.fft.rfftfreq(nperseg, d=1.0 / sampling_rate, device=x.device)

        # Calculate time bins (in seconds)
        # These represent the center time of each time window in the spectrogram
        n_times = x.shape[2]
        # time_bins = (
        #     torch.arange(n_times, device=x.device, dtype=torch.float32)
        #     * hop_length
        #     / sampling_rate
        # )

        # Trim to max frequency (using a pre-calculated max frequency bin)
        x = x[:, self.min_frequency_bin : self.max_frequency_bin, :]
        freq_bins = freq_bins[self.min_frequency_bin : self.max_frequency_bin]

        # Reshape back
        _, n_freqs, n_times = x.shape
        x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
        x = x.transpose(2, 3)  # (batch_size, n_electrodes, n_timebins, n_freqs)

        # Z-score normalization
        if z_score:
            x = x - x.mean(dim=[0, 2], keepdim=True)
            x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

        if self.line_noise_mask is not None:  # If removing line noise, set line noise to 0
            self.line_noise_mask = self.line_noise_mask.to(x.device)
            x = x.masked_fill(self.line_noise_mask.view(1, 1, 1, -1), 0)

        # Transform to match expected output dimension
        x = self.output_transform(x)  # shape: (batch_size, n_electrodes, n_timebins, output_dim)

        x = x.to(dtype=data.dtype)
        return x

    def _compute_line_noise_mask(
        self,
        freq_bins: torch.Tensor,
        line_noise_freqs: list | None = None,
        margin: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute a mask for line noise frequencies in the spectrogram.

        Args:
            freq_bins (torch.Tensor): The frequency bins of the spectrogram.
            line_noise_freqs (list, optional): The line noise frequencies to mask. If none, defaults to [50, 60].
            margin (float, optional): The margin around the line noise frequencies to include in the mask. Defaults to 2.0.

        Returns:
            torch.Tensor: A boolean mask indicating the line noise frequencies.
        """
        # 60 Hz and its harmonics
        line_noise_mask = torch.zeros(freq_bins.shape[0], device=freq_bins.device, dtype=torch.bool)

        if line_noise_freqs is None:
            line_noise_freqs = [50, 60]

        for freq in line_noise_freqs:
            line_noise_mask |= torch.abs(freq_bins - freq) <= margin

        return line_noise_mask
