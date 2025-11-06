import pytest
import torch
from helpers import make_batch

from crane.preprocess import Spectrogram


@pytest.mark.parametrize("window", ["hann", "boxcar"])
def test_shape_dtype_and_windows(window):
    batch = make_batch()
    seg_len = 0.5  # s -> nperseg = 128 @ 256 Hz
    proc = Spectrogram(
        segment_length=seg_len,
        p_overlap=0.5,
        min_frequency=0.0,
        max_frequency=100.0,
        window=window,
        remove_line_noise=False,
        output_dim=-1,
    )
    out = proc(batch)
    assert out.dtype == batch["ieeg"]["data"].dtype  # dtype preserved

    B, E, _, _ = out.shape
    assert (B, E) == (2, 3)

    # n_freqs should equal computed band size
    expected_nfreqs = proc.max_frequency_bin - proc.min_frequency_bin
    assert out.shape[-1] == expected_nfreqs
    # time dimension > 0
    assert out.shape[2] > 0


def test_output_dim_projection():
    batch = make_batch()
    proc = Spectrogram(
        segment_length=0.5,
        p_overlap=0.5,
        min_frequency=0.0,
        max_frequency=100.0,
        window="hann",
        remove_line_noise=False,
        output_dim=32,
    )
    out = proc(batch)
    assert out.shape[-1] == 32  # projected to requested dim


def test_zscore_normalization_centers_over_batch_and_time():
    batch = make_batch()
    proc = Spectrogram(
        segment_length=0.5,
        p_overlap=0.5,
        min_frequency=0.0,
        max_frequency=100.0,
        window="hann",
        remove_line_noise=False,
        output_dim=-1,
    )
    with torch.no_grad():
        out = proc(batch, z_score=True)

    # Mean/std computed over batch and time per electrode & freq in the module.
    # Check they are approximately centered/scaled (tolerant due to eps in denom).
    mean_bt = out.mean(dim=(0, 2))  # (elec, freq)
    std_bt = out.std(dim=(0, 2))
    assert torch.allclose(mean_bt, torch.zeros_like(mean_bt), atol=5e-2)
    assert torch.all(std_bt > 0)  # not collapsed


def test_line_noise_mask_zeroes_masked_bins():
    batch = make_batch()
    proc = Spectrogram(
        segment_length=0.5,
        p_overlap=0.5,
        min_frequency=0.0,
        max_frequency=100.0,
        window="hann",
        remove_line_noise=True,
        output_dim=-1,
    )
    assert proc.line_noise_mask is not None
    masked_idx = torch.where(proc.line_noise_mask)[0]
    assert masked_idx.numel() > 0  # we expect some bins around 50/60 Hz

    with torch.no_grad():
        out = proc(batch, z_score=False)  # z_score off to simplify check

    # All masked frequency bins should be exactly zero for every batch, electrode, time
    masked_vals = out[..., masked_idx]
    assert torch.count_nonzero(masked_vals) == 0


def test_compute_line_noise_mask():
    # Create a dummy frequency bins tensor
    freq_bins = torch.linspace(0, 100, steps=201)  # 0 to 100 Hz with 0.5 Hz steps

    # Instantiate the preprocessor
    proc = Spectrogram(
        segment_length=0.5,
        p_overlap=0.5,
        min_frequency=0.0,
        max_frequency=100.0,
        window="hann",
        remove_line_noise=True,
        output_dim=-1,
    )

    # Compute the line noise mask
    mask = proc._compute_line_noise_mask(freq_bins, line_noise_freqs=[50, 60], margin=2.0)

    # Check that the mask has the correct shape
    assert mask.shape == freq_bins.shape

    # Check that frequencies around 50 Hz and 60 Hz are masked
    for line_freq in [50, 60]:
        masked_region = (freq_bins >= (line_freq - 2.0)) & (freq_bins <= (line_freq + 2.0))
        assert torch.all(mask[masked_region])

    # Check that frequencies outside the masked regions are not masked
    unmasked_region = (freq_bins < 48.0) | ((freq_bins > 52.0) & (freq_bins < 58.0)) | (freq_bins > 62.0)
    assert torch.all(~mask[unmasked_region])

    default_mask = proc._compute_line_noise_mask(freq_bins, margin=2.0)
    assert torch.equal(mask, default_mask)
