"""Diagnose saved offline ANC outputs."""

from __future__ import annotations

import os

import config
import numpy as np
import soundfile as sf
from evaluate import compute_band_reduction_db, compute_noise_reduction_db, plot_results
from scipy.signal import freqz


def _load_wav(path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Load a WAV file if it exists.

    Args:
        path: WAV file path.

    Returns:
        audio_and_fs: Tuple (audio, fs), or (None, None) when absent.
    """
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None, None
    audio, fs = sf.read(path)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio.astype(float), fs


def plot_error_waveform(before: np.ndarray, after: np.ndarray, output_dir: str) -> None:
    """Plot before/after error waveforms.

    Args:
        before: Baseline error with shape (num_samples,).
        after: Residual error with shape (num_samples,).
        output_dir: Directory for plot output.

    Returns:
        None.
    """
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(before, label="Before ANC", alpha=0.7)
    plt.plot(after, label="After ANC", alpha=0.7)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Error Waveform")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_error_waveform.png"), dpi=150)
    plt.close()


def plot_alpha_if_available(output_dir: str) -> None:
    """Plot alpha history if outputs/alpha.npy exists.

    Args:
        output_dir: Directory containing output arrays.

    Returns:
        None.
    """
    path = os.path.join(output_dir, "alpha.npy")
    if not os.path.exists(path):
        return
    from matplotlib import pyplot as plt

    alpha = np.load(path)
    plt.figure(figsize=(10, 5))
    for index in range(alpha.shape[1]):
        plt.plot(alpha[:, index], label=f"alpha {index + 1}")
    plt.xlabel("Frame")
    plt.ylabel("Alpha")
    plt.title("Alpha Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_alpha_over_time.png"), dpi=150)
    plt.close()


def plot_band_energy_if_available(output_dir: str) -> None:
    """Plot smoothed band-energy traces if available.

    Args:
        output_dir: Directory containing output arrays.

    Returns:
        None.
    """
    path = os.path.join(output_dir, "band_energy_smooth.npy")
    if not os.path.exists(path):
        return
    from matplotlib import pyplot as plt

    energy = np.load(path)
    plt.figure(figsize=(10, 6))
    for index in range(energy.shape[1]):
        plt.plot(energy[:, index], label=f"band {index + 1}")
    plt.xlabel("Frame")
    plt.ylabel("Smoothed energy")
    plt.title("Band Energy Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_band_energy.png"), dpi=150)
    plt.close()


def plot_filter_responses_if_available(output_dir: str) -> None:
    """Plot final saved W_GFANC, W_adapt, and W_final frequency responses.

    Args:
        output_dir: Directory containing saved filter histories.

    Returns:
        None.
    """
    paths = {
        "W_GFANC": os.path.join(output_dir, "W_gfanc.npy"),
        "W_adapt": os.path.join(output_dir, "W_adapt.npy"),
        "W_final": os.path.join(output_dir, "W_final.npy"),
    }
    if not all(os.path.exists(path) for path in paths.values()):
        return
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 5))
    for label, path in paths.items():
        history = np.load(path)
        taps = history[-1] if history.ndim == 2 else history
        frequencies, response = freqz(taps, worN=2048, fs=config.fs)
        plt.plot(frequencies, 20.0 * np.log10(np.maximum(np.abs(response), 1e-12)), label=label)
    plt.xlim(0, 600)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Saved Filter Frequency Responses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_filter_responses.png"), dpi=150)
    plt.close()


def print_recommendations(band_reduction: float, anti_noise: np.ndarray | None) -> None:
    """Print diagnostic suggestions from output metrics.

    Args:
        band_reduction: 50-300 Hz noise reduction in dB.
        anti_noise: Optional anti-noise signal with shape (num_samples,).

    Returns:
        None.
    """
    print("Diagnostic suggestions:")
    if band_reduction < 0.0:
        print("- Low-frequency noise increased: check that e(n)=d(n)+S(z)*y(n) is used consistently.")
    if anti_noise is not None and np.max(np.abs(anti_noise)) > 0.95 * config.y_limit:
        print("- y is near its limit: consider lowering y_limit, top_k, W_gfanc_norm_limit, or step_size.")
    w_adapt_path = os.path.join(config.output_dir, "W_adapt.npy")
    if os.path.exists(w_adapt_path):
        w_adapt = np.load(w_adapt_path)
        if np.max(np.linalg.norm(w_adapt, axis=1)) > 0.9 * config.adapt_norm_limit:
            print("- W_adapt is large: consider lowering rho_adapt or step_size.")
    print("- If band switching is too jumpy, increase band_energy_smoothing or min_hold_frames.")
    print("- If fixed-filter control has little effect, inspect data/filters/filter_bank_frequency_responses.png.")


def main() -> None:
    """Diagnose saved outputs under outputs/.

    Args:
        None.

    Returns:
        None.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    before, fs_before = _load_wav(os.path.join(config.output_dir, "error_before.wav"))
    after, fs_after = _load_wav(os.path.join(config.output_dir, "error_after.wav"))
    anti_noise, _ = _load_wav(os.path.join(config.output_dir, "anti_noise.wav"))
    if before is None or after is None:
        print("Cannot compute diagnostics until error_before.wav and error_after.wav exist.")
        return
    fs = fs_before or config.fs
    broadband = compute_noise_reduction_db(before, after)
    band = compute_band_reduction_db(before, after, fs, config.control_band)
    print(f"Broadband reduction: {broadband:.2f} dB")
    print(f"{config.control_band[0]}-{config.control_band[1]} Hz reduction: {band:.2f} dB")
    if anti_noise is not None:
        print(f"Max abs anti-noise: {np.max(np.abs(anti_noise)):.4f}")

    plot_results(before, after, fs, config.output_dir, config.control_band)
    plot_error_waveform(before, after, config.output_dir)
    plot_alpha_if_available(config.output_dir)
    plot_band_energy_if_available(config.output_dir)
    plot_filter_responses_if_available(config.output_dir)
    print_recommendations(band, anti_noise)


if __name__ == "__main__":
    main()
