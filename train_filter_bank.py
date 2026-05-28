"""Train the stage-2 fixed FxNLMS control filter bank offline."""

from __future__ import annotations

import json
import os

import config
import numpy as np
from evaluate import compute_band_reduction_db
from filter_bank import FixedFilterBank, save_filter_bank
from fxnlms import FxNLMSResidualAdapter
from matplotlib import pyplot as plt
from path_model import (
    create_default_primary_path,
    create_default_secondary_path,
    create_estimated_secondary_path,
)
from scipy.signal import butter, filtfilt, freqz


FILTER_BANDS = [
    (50, 80),
    (80, 110),
    (110, 150),
    (150, 200),
    (200, 250),
    (250, 300),
    (50, 150),
    (150, 300),
]


def generate_band_limited_noise(
    fs: int,
    duration_sec: float,
    band: tuple[float, float],
    seed: int,
) -> np.ndarray:
    """Generate band-limited training noise.

    Args:
        fs: Sample rate as an integer.
        duration_sec: Signal duration in seconds.
        band: Noise band as (low_hz, high_hz).
        seed: Random seed.

    Returns:
        noise: Band-limited reference signal with shape (num_samples,).
    """
    num_samples = int(fs * duration_sec)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(num_samples)
    b, a = butter(4, band, btype="bandpass", fs=fs)
    noise = filtfilt(b, a, white)
    peak = np.max(np.abs(noise))
    if peak > 0.0:
        noise = 0.8 * noise / peak
    return noise.astype(float)


def train_one_filter(
    reference: np.ndarray,
    primary_path,
    secondary_path,
    secondary_path_hat,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train one fixed control filter with offline FxNLMS.

    Args:
        reference: Band-limited reference signal with shape (num_samples,).
        primary_path: Primary FIRPath P(z).
        secondary_path: True secondary FIRPath S(z).
        secondary_path_hat: Estimated secondary FIRPath S_hat(z).

    Returns:
        trained_filter: Final W_adapt with shape (filter_len,).
        desired: Primary noise d(n) with shape (num_samples,).
        error: Residual error e(n) with shape (num_samples,).
    """
    desired = primary_path.process(reference)
    error = np.zeros_like(reference)

    adapter = FxNLMSResidualAdapter(
        filter_len=config.filter_len,
        step_size=config.step_size,
        regularization=config.regularization,
        leakage=config.leakage,
        rho_adapt=config.rho_adapt,
        adapt_norm_limit=config.adapt_norm_limit,
        adapt_tap_clip=config.adapt_tap_clip,
    )

    x_buffer = np.zeros(config.filter_len, dtype=float)
    filtered_x_buffer = np.zeros(config.filter_len, dtype=float)
    x_secondary_hat_buffer = np.zeros(config.secondary_path_len, dtype=float)
    y_secondary_buffer = np.zeros(config.secondary_path_len, dtype=float)
    w_gfanc = np.zeros(config.filter_len, dtype=float)

    for n, sample in enumerate(reference):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = sample

        x_secondary_hat_buffer[1:] = x_secondary_hat_buffer[:-1]
        x_secondary_hat_buffer[0] = sample
        filtered_x_sample = secondary_path_hat.sample(x_secondary_hat_buffer)
        filtered_x_buffer[1:] = filtered_x_buffer[:-1]
        filtered_x_buffer[0] = filtered_x_sample

        w_final = adapter.final_filter(w_gfanc)
        y = float(np.dot(w_final, x_buffer))
        y = float(np.clip(y, -config.y_limit, config.y_limit))

        y_secondary_buffer[1:] = y_secondary_buffer[:-1]
        y_secondary_buffer[0] = y
        error[n] = desired[n] + secondary_path.sample(y_secondary_buffer)

        adapter.update(error[n], filtered_x_buffer, w_gfanc)

    return adapter.get_filter(), desired, error


def plot_filter_responses(filters: np.ndarray, output_dir: str) -> None:
    """Plot frequency responses for all fixed filters.

    Args:
        filters: Fixed filter bank with shape (num_filters, filter_len).
        output_dir: Directory where the figure is saved.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for index, taps in enumerate(filters):
        frequencies, response = freqz(taps, worN=2048, fs=config.fs)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
        low, high = FILTER_BANDS[index]
        plt.plot(frequencies, magnitude_db, label=f"W{index + 1}: {low}-{high} Hz")
    plt.axvspan(config.control_band[0], config.control_band[1], color="tab:green", alpha=0.08)
    plt.xlim(0, 600)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Fixed Control Filter Bank Frequency Responses")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "filter_bank_frequency_responses.png"), dpi=150)
    plt.close()


def main() -> None:
    """Train and save the fixed control filter bank.

    Args:
        None.

    Returns:
        None.
    """
    filters_dir = os.path.join("data", "filters")
    os.makedirs(filters_dir, exist_ok=True)

    primary_path = create_default_primary_path(config.primary_path_len)
    secondary_path = create_default_secondary_path(config.secondary_path_len)
    secondary_path_hat = create_estimated_secondary_path(secondary_path, perturbation=0.0)

    trained_filters = []
    meta = {
        "num_filters": config.num_filters,
        "filter_len": config.filter_len,
        "fs": config.fs,
        "bands_hz": FILTER_BANDS,
        "training_duration_sec": config.simulation_duration_sec,
        "control_band_hz": config.control_band,
        "items": [],
    }

    for index, band in enumerate(FILTER_BANDS):
        reference = generate_band_limited_noise(
            fs=config.fs,
            duration_sec=config.simulation_duration_sec,
            band=band,
            seed=100 + index,
        )
        trained_filter, desired, error = train_one_filter(
            reference,
            primary_path,
            secondary_path,
            secondary_path_hat,
        )
        reduction_db = compute_band_reduction_db(desired, error, config.fs, config.control_band)
        trained_filters.append(trained_filter)
        meta["items"].append(
            {
                "index": index,
                "name": f"{band[0]}-{band[1]}Hz",
                "band_hz": band,
                "reduction_50_300_db": reduction_db,
                "filter_norm": float(np.linalg.norm(trained_filter)),
            }
        )
        print(f"Filter {index + 1}/8 {band[0]}-{band[1]} Hz: 50-300 Hz reduction {reduction_db:.2f} dB")

    filters = np.asarray(trained_filters, dtype=float)
    if filters.shape != (config.num_filters, config.filter_len):
        raise RuntimeError(f"Unexpected filter bank shape: {filters.shape}")

    bank = FixedFilterBank(filters)
    save_filter_bank(os.path.join(filters_dir, "filter_bank.npy"), bank)
    with open(os.path.join(filters_dir, "filter_bank_meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    plot_filter_responses(filters, filters_dir)

    print(f"Saved filter bank: {os.path.join(filters_dir, 'filter_bank.npy')}")
    print(f"Saved metadata: {os.path.join(filters_dir, 'filter_bank_meta.json')}")
    print(f"Saved response plot: {os.path.join(filters_dir, 'filter_bank_frequency_responses.png')}")


if __name__ == "__main__":
    main()
