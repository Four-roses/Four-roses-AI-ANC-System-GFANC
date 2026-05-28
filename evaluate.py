"""Evaluation utilities for offline ANC simulation."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch


def compute_power_db(signal: np.ndarray) -> float:
    """Compute signal power in dB.

    Args:
        signal: Input signal with shape (num_samples,).

    Returns:
        power_db: Scalar power value in dB.
    """
    power = float(np.mean(np.asarray(signal, dtype=float) ** 2))
    return 10.0 * np.log10(power + 1e-12)


def compute_noise_reduction_db(before: np.ndarray, after: np.ndarray) -> float:
    """Compute noise reduction in dB.

    Args:
        before: Baseline noise signal with shape (num_samples,).
        after: Residual error signal with shape (num_samples,).

    Returns:
        reduction_db: Scalar noise reduction in dB.
    """
    return compute_power_db(before) - compute_power_db(after)


def bandpass_signal(signal: np.ndarray, fs: int, band: tuple[float, float]) -> np.ndarray:
    """Band-pass filter a signal for band-limited evaluation.

    Args:
        signal: Input signal with shape (num_samples,).
        fs: Sample rate as an integer.
        band: Low and high cutoff frequencies in Hz.

    Returns:
        filtered: Band-limited signal with shape (num_samples,).
    """
    low, high = band
    sos_b, sos_a = butter(4, [low, high], btype="bandpass", fs=fs)
    return filtfilt(sos_b, sos_a, signal)


def compute_band_reduction_db(
    before: np.ndarray,
    after: np.ndarray,
    fs: int,
    band: tuple[float, float],
) -> float:
    """Compute band-limited noise reduction in dB.

    Args:
        before: Baseline noise signal with shape (num_samples,).
        after: Residual error signal with shape (num_samples,).
        fs: Sample rate as an integer.
        band: Evaluation band as (low_hz, high_hz).

    Returns:
        reduction_db: Scalar band-limited noise reduction in dB.
    """
    before_band = bandpass_signal(before, fs, band)
    after_band = bandpass_signal(after, fs, band)
    return compute_noise_reduction_db(before_band, after_band)


def plot_results(
    before: np.ndarray,
    after: np.ndarray,
    fs: int,
    output_dir: str,
    band: tuple[float, float],
) -> None:
    """Plot PSD and convergence results.

    Args:
        before: Baseline error signal with shape (num_samples,).
        after: Residual error signal with shape (num_samples,).
        fs: Sample rate as an integer.
        output_dir: Directory where figures are saved.
        band: Control band as (low_hz, high_hz).

    Returns:
        None.
    """
    import os
    from matplotlib import pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    freq_before, psd_before = welch(before, fs=fs, nperseg=1024)
    freq_after, psd_after = welch(after, fs=fs, nperseg=1024)

    plt.figure(figsize=(9, 5))
    plt.semilogy(freq_before, psd_before + 1e-18, label="Before ANC")
    plt.semilogy(freq_after, psd_after + 1e-18, label="After ANC")
    plt.axvspan(band[0], band[1], color="tab:green", alpha=0.12, label="Control band")
    plt.xlim(0, 600)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("PSD Before/After ANC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psd_before_after.png"), dpi=150)
    plt.close()

    window_len = max(fs // 20, 1)
    kernel = np.ones(window_len) / window_len
    error_power = np.convolve(after**2, kernel, mode="same")

    plt.figure(figsize=(9, 4))
    plt.plot(10.0 * np.log10(error_power + 1e-12))
    plt.xlabel("Sample")
    plt.ylabel("Smoothed error power (dB)")
    plt.title("Error Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_convergence.png"), dpi=150)
    plt.close()
