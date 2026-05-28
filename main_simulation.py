"""Offline ANC simulation entry point."""

from __future__ import annotations

import argparse
import os

import config
import numpy as np
import soundfile as sf

from evaluate import compute_band_reduction_db, compute_noise_reduction_db, plot_results
from filter_bank import load_filter_bank
from fxnlms import FxNLMSResidualAdapter
from mock_gfanc import BandEnergyGate
from path_model import (
    create_default_primary_path,
    create_default_secondary_path,
    create_estimated_secondary_path,
)


MODES = ("fxnlms_only", "band_gated", "band_gated_fxnlms")
MODE_ALIASES = {
    "rule_gfanc": "band_gated",
    "oracle_gfanc": "band_gated",
    "gfanc_kalman": "band_gated",
    "gfanc_kalman_fxnlms": "band_gated_fxnlms",
    "full_system": "band_gated_fxnlms",
}


def generate_low_frequency_noise(fs: int, duration_sec: float) -> np.ndarray:
    """Generate a deterministic low-frequency reference noise signal.

    Args:
        fs: Sample rate as an integer.
        duration_sec: Signal duration in seconds.

    Returns:
        reference: Test noise signal with shape (num_samples,).
    """
    num_samples = int(fs * duration_sec)
    time = np.arange(num_samples) / fs
    reference = (
        0.45 * np.sin(2.0 * np.pi * 80.0 * time)
        + 0.30 * np.sin(2.0 * np.pi * 160.0 * time + 0.4)
        + 0.20 * np.sin(2.0 * np.pi * 240.0 * time + 1.1)
    )
    reference += 0.02 * np.random.default_rng(config.path_random_seed).standard_normal(num_samples)
    peak = np.max(np.abs(reference))
    if peak > 0.0:
        reference = 0.8 * reference / peak
    return reference.astype(float)


def _make_adapter() -> FxNLMSResidualAdapter:
    """Create the residual FxNLMS adapter from config.

    Args:
        None.

    Returns:
        adapter: FxNLMSResidualAdapter instance.
    """
    return FxNLMSResidualAdapter(
        filter_len=config.filter_len,
        step_size=config.step_size,
        regularization=config.regularization,
        leakage=config.leakage,
        rho_adapt=config.rho_adapt,
        adapt_norm_limit=config.adapt_norm_limit,
        adapt_tap_clip=config.adapt_tap_clip,
    )


def run_fxnlms_simulation(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run baseline FxNLMS-only ANC.

    Args:
        reference: External reference microphone signal with shape (num_samples,).

    Returns:
        error_before: Primary noise d(n) with shape (num_samples,).
        error_after: Residual error e(n) with shape (num_samples,).
        anti_noise: Clipped loudspeaker command y(n) with shape (num_samples,).
    """
    primary_path = create_default_primary_path(config.primary_path_len)
    secondary_path = create_default_secondary_path(config.secondary_path_len)
    secondary_path_hat = create_estimated_secondary_path(secondary_path, perturbation=0.0)
    desired = primary_path.process(reference)

    adapter = _make_adapter()
    x_buffer = np.zeros(config.filter_len, dtype=float)
    filtered_x_buffer = np.zeros(config.filter_len, dtype=float)
    x_hat_buffer = np.zeros(config.secondary_path_len, dtype=float)
    y_secondary_buffer = np.zeros(config.secondary_path_len, dtype=float)
    w_gfanc = np.zeros(config.filter_len, dtype=float)
    error = np.zeros_like(reference, dtype=float)
    anti_noise = np.zeros_like(reference, dtype=float)

    for n, sample in enumerate(reference):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = sample

        x_hat_buffer[1:] = x_hat_buffer[:-1]
        x_hat_buffer[0] = sample
        filtered_x_buffer[1:] = filtered_x_buffer[:-1]
        filtered_x_buffer[0] = secondary_path_hat.sample(x_hat_buffer)

        w_final = adapter.final_filter(w_gfanc)
        y = float(np.clip(np.dot(w_final, x_buffer), -config.y_limit, config.y_limit))
        anti_noise[n] = y

        y_secondary_buffer[1:] = y_secondary_buffer[:-1]
        y_secondary_buffer[0] = y
        error[n] = desired[n] + secondary_path.sample(y_secondary_buffer)

        if config.use_fxnlms_refinement:
            adapter.update(error[n], filtered_x_buffer, w_gfanc)

    return desired, error, anti_noise


def _limit_w_gfanc(w_gfanc: np.ndarray) -> np.ndarray:
    """Limit W_GFANC by norm.

    Args:
        w_gfanc: Combined fixed control filter with shape (filter_len,).

    Returns:
        limited: Norm-limited W_GFANC with shape (filter_len,).
    """
    norm = float(np.linalg.norm(w_gfanc))
    if norm > config.w_gfanc_norm_limit > 0.0:
        return w_gfanc * (config.w_gfanc_norm_limit / norm)
    return w_gfanc


def run_band_gated_simulation(
    reference: np.ndarray,
    use_residual: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run band-energy-gated fixed-filter ANC.

    Args:
        reference: External reference microphone signal with shape (num_samples,).
        use_residual: If True, update W_adapt with FxNLMS. If False, use only
            the selected fixed filters.

    Returns:
        error_before: Primary noise d(n) with shape (num_samples,).
        error_after: Residual error e(n) with shape (num_samples,).
        anti_noise: Clipped loudspeaker command y(n) with shape (num_samples,).
        alpha_history: Binary alpha history with shape (num_frames, num_filters).
        w_gfanc_history: W_GFANC history with shape (num_samples, filter_len).
        w_adapt_history: W_adapt history with shape (num_samples, filter_len).
        w_final_history: W_final history with shape (num_samples, filter_len).
        band_energy_history: Smoothed band energy with shape
            (num_frames, num_filters).
    """
    filter_bank_path = os.path.join("data", "filters", "filter_bank.npy")
    if not os.path.exists(filter_bank_path):
        raise FileNotFoundError("Missing filter bank. Run python train_filter_bank.py first.")

    bank = load_filter_bank(filter_bank_path)
    primary_path = create_default_primary_path(config.primary_path_len)
    secondary_path = create_default_secondary_path(config.secondary_path_len)
    secondary_path_hat = create_estimated_secondary_path(secondary_path, perturbation=0.0)
    desired = primary_path.process(reference)

    gate = BandEnergyGate(
        fs=config.fs,
        num_filters=bank.shape[0],
        top_k=config.band_gate_top_k,
        threshold_ratio=config.band_gate_threshold_ratio,
        smoothing=config.band_energy_smoothing,
        min_hold_frames=config.band_gate_min_hold_frames,
    )
    adapter = _make_adapter()

    num_samples = reference.shape[0]
    num_frames = int(np.ceil(num_samples / config.frame_len))
    error = np.zeros(num_samples, dtype=float)
    anti_noise = np.zeros(num_samples, dtype=float)
    alpha_history = np.zeros((num_frames, bank.shape[0]), dtype=float)
    band_energy_history = np.zeros((num_frames, bank.shape[0]), dtype=float)
    w_gfanc_history = np.zeros((num_samples, config.filter_len), dtype=np.float32)
    w_adapt_history = np.zeros((num_samples, config.filter_len), dtype=np.float32)
    w_final_history = np.zeros((num_samples, config.filter_len), dtype=np.float32)

    x_buffer = np.zeros(config.filter_len, dtype=float)
    filtered_x_buffer = np.zeros(config.filter_len, dtype=float)
    x_hat_buffer = np.zeros(config.secondary_path_len, dtype=float)
    y_secondary_buffer = np.zeros(config.secondary_path_len, dtype=float)
    previous_w_gfanc = np.zeros(config.filter_len, dtype=float)

    for frame_index, frame_start in enumerate(range(0, num_samples, config.frame_len)):
        frame_end = min(frame_start + config.frame_len, num_samples)
        frame_len = frame_end - frame_start
        x_segment = reference[frame_start:frame_end]
        if x_segment.shape[0] < config.frame_len:
            x_segment = np.pad(x_segment, (0, config.frame_len - x_segment.shape[0]))

        alpha, _, smooth_energy = gate.update(x_segment)
        selected_count = max(int(np.sum(alpha)), 1)
        combine_alpha = alpha / selected_count if config.normalize_selected_filters else alpha
        target_w_gfanc = _limit_w_gfanc(bank.combine(combine_alpha))
        alpha_history[frame_index] = alpha
        band_energy_history[frame_index] = smooth_energy

        for local_index, n in enumerate(range(frame_start, frame_end)):
            interpolation = (local_index + 1) / max(frame_len, 1)
            w_gfanc = (1.0 - interpolation) * previous_w_gfanc + interpolation * target_w_gfanc

            x_buffer[1:] = x_buffer[:-1]
            x_buffer[0] = reference[n]

            x_hat_buffer[1:] = x_hat_buffer[:-1]
            x_hat_buffer[0] = reference[n]
            filtered_x_buffer[1:] = filtered_x_buffer[:-1]
            filtered_x_buffer[0] = secondary_path_hat.sample(x_hat_buffer)

            adapter.constrain(w_gfanc)
            w_adapt = adapter.get_filter()
            w_final = w_gfanc + w_adapt
            y = float(np.clip(np.dot(w_final, x_buffer), -config.y_limit, config.y_limit))
            anti_noise[n] = y

            y_secondary_buffer[1:] = y_secondary_buffer[:-1]
            y_secondary_buffer[0] = y
            error[n] = desired[n] + secondary_path.sample(y_secondary_buffer)

            w_gfanc_history[n] = w_gfanc.astype(np.float32)
            w_adapt_history[n] = w_adapt.astype(np.float32)
            w_final_history[n] = w_final.astype(np.float32)

            if use_residual:
                adapter.update(error[n], filtered_x_buffer, w_gfanc)

        previous_w_gfanc = target_w_gfanc

    return (
        desired,
        error,
        anti_noise,
        alpha_history,
        w_gfanc_history,
        w_adapt_history,
        w_final_history,
        band_energy_history,
    )


def classify_status(
    band_reduction_db: float,
    max_abs_y: float,
    nan_or_inf: bool,
    adapt_limit_ok: bool = True,
) -> str:
    """Classify one simulation result.

    Args:
        band_reduction_db: Band-limited noise reduction in dB.
        max_abs_y: Maximum absolute loudspeaker command.
        nan_or_inf: True if any output contains NaN or Inf.
        adapt_limit_ok: True if W_adapt respects the configured norm limit.

    Returns:
        status: One of "PASS", "WARNING", or "FAIL".
    """
    if nan_or_inf or max_abs_y > config.y_limit + 1e-9 or not adapt_limit_ok:
        return "FAIL"
    if band_reduction_db < -3.0:
        return "WARNING"
    return "PASS"


def build_result(
    mode: str,
    error_before: np.ndarray,
    error_after: np.ndarray,
    anti_noise: np.ndarray,
    w_gfanc_history: np.ndarray | None = None,
    w_adapt_history: np.ndarray | None = None,
) -> dict[str, object]:
    """Build a diagnostic result dictionary for one simulation.

    Args:
        mode: Simulation mode name.
        error_before: Primary noise d(n) with shape (num_samples,).
        error_after: Residual error e(n) with shape (num_samples,).
        anti_noise: Loudspeaker command y(n) with shape (num_samples,).
        w_gfanc_history: Optional W_GFANC history with shape
            (num_samples, filter_len).
        w_adapt_history: Optional W_adapt history with shape
            (num_samples, filter_len).

    Returns:
        result: Dictionary with reduction, safety, and status fields.
    """
    arrays = [error_before, error_after, anti_noise]
    if w_gfanc_history is not None:
        arrays.append(w_gfanc_history)
    if w_adapt_history is not None:
        arrays.append(w_adapt_history)
    nan_or_inf = not all(np.all(np.isfinite(array)) for array in arrays)

    w_gfanc_norm = 0.0
    w_adapt_norm = 0.0
    adapt_limit_ok = True
    if w_gfanc_history is not None and w_gfanc_history.size:
        w_gfanc_norms = np.linalg.norm(w_gfanc_history, axis=1)
        w_gfanc_norm = float(np.max(w_gfanc_norms))
    if w_adapt_history is not None and w_adapt_history.size:
        w_adapt_norms = np.linalg.norm(w_adapt_history, axis=1)
        w_adapt_norm = float(np.max(w_adapt_norms))
        if w_gfanc_history is not None and w_gfanc_history.size:
            limits = config.rho_adapt * np.maximum(np.linalg.norm(w_gfanc_history, axis=1), 1e-6)
            adapt_limit_ok = bool(np.all(w_adapt_norms <= limits + 1e-6))

    broadband_reduction = compute_noise_reduction_db(error_before, error_after)
    band_reduction = compute_band_reduction_db(
        error_before,
        error_after,
        config.fs,
        config.control_band,
    )
    max_abs_y = float(np.max(np.abs(anti_noise))) if anti_noise.size else 0.0
    max_abs_e = float(np.max(np.abs(error_after))) if error_after.size else 0.0
    return {
        "mode": mode,
        "broadband_reduction_db": float(broadband_reduction),
        "band_reduction_db": float(band_reduction),
        "max_abs_y": max_abs_y,
        "max_abs_e": max_abs_e,
        "W_gfanc_norm": w_gfanc_norm,
        "W_adapt_norm": w_adapt_norm,
        "nan_or_inf": bool(nan_or_inf),
        "adapt_limit_ok": bool(adapt_limit_ok),
        "diverged": bool(max_abs_e > 10.0 * (np.max(np.abs(error_before)) + 1e-12)),
        "status": classify_status(band_reduction, max_abs_y, nan_or_inf, adapt_limit_ok),
    }


def write_audio(path: str, audio: np.ndarray, fs: int) -> None:
    """Write clipped audio to disk.

    Args:
        path: Output WAV path.
        audio: Audio signal with shape (num_samples,).
        fs: Sample rate as an integer.

    Returns:
        None.
    """
    sf.write(path, np.clip(audio, -1.0, 1.0), fs)


def save_gate_traces(alpha: np.ndarray, band_energy: np.ndarray, output_dir: str) -> None:
    """Save binary alpha and smoothed band-energy traces.

    Args:
        alpha: Binary alpha history with shape (num_frames, num_filters).
        band_energy: Smoothed band energy with shape (num_frames, num_filters).
        output_dir: Output directory.

    Returns:
        None.
    """
    np.save(os.path.join(output_dir, "alpha.npy"), alpha)
    np.save(os.path.join(output_dir, "band_energy_smooth.npy"), band_energy)


def run_simulation(
    mode: str = "fxnlms_only",
    save_outputs: bool = True,
    reference: np.ndarray | None = None,
) -> dict[str, object]:
    """Run one named ANC simulation and return a result dictionary.

    Args:
        mode: One of "fxnlms_only", "band_gated", or "band_gated_fxnlms".
            Legacy mode names are accepted as aliases.
        save_outputs: If True, save WAV, trace, and plot files under outputs/.
        reference: Optional reference signal with shape (num_samples,).

    Returns:
        result: Dictionary with mode, reduction metrics, safety metrics, and
            status.
    """
    mode = MODE_ALIASES.get(mode, mode)
    if reference is None:
        reference = generate_low_frequency_noise(config.fs, config.simulation_duration_sec)

    alpha = band_energy = None
    w_gfanc_history = w_adapt_history = w_final_history = None
    if mode == "fxnlms_only":
        error_before, error_after, anti_noise = run_fxnlms_simulation(reference)
    elif mode == "band_gated":
        (
            error_before,
            error_after,
            anti_noise,
            alpha,
            w_gfanc_history,
            w_adapt_history,
            w_final_history,
            band_energy,
        ) = run_band_gated_simulation(reference, use_residual=False)
    elif mode == "band_gated_fxnlms":
        (
            error_before,
            error_after,
            anti_noise,
            alpha,
            w_gfanc_history,
            w_adapt_history,
            w_final_history,
            band_energy,
        ) = run_band_gated_simulation(reference, use_residual=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    result = build_result(
        mode,
        error_before,
        error_after,
        anti_noise,
        w_gfanc_history,
        w_adapt_history,
    )

    if save_outputs:
        os.makedirs(config.output_dir, exist_ok=True)
        write_audio(os.path.join(config.output_dir, "error_before.wav"), error_before, config.fs)
        write_audio(os.path.join(config.output_dir, "error_after.wav"), error_after, config.fs)
        write_audio(os.path.join(config.output_dir, "anti_noise.wav"), anti_noise, config.fs)
        write_audio(os.path.join(config.output_dir, f"error_after_{mode}.wav"), error_after, config.fs)
        write_audio(os.path.join(config.output_dir, f"anti_noise_{mode}.wav"), anti_noise, config.fs)
        if alpha is not None and band_energy is not None:
            save_gate_traces(alpha, band_energy, config.output_dir)
        if w_gfanc_history is not None and w_adapt_history is not None and w_final_history is not None:
            np.save(os.path.join(config.output_dir, "W_gfanc.npy"), w_gfanc_history)
            np.save(os.path.join(config.output_dir, "W_adapt.npy"), w_adapt_history)
            np.save(os.path.join(config.output_dir, "W_final.npy"), w_final_history)
        plot_results(error_before, error_after, config.fs, config.output_dir, config.control_band)

    return result


def main() -> None:
    """Run one mode or a default mode comparison from the command line.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Offline ANC simulation")
    parser.add_argument(
        "--mode",
        default="all",
        choices=(
            "all",
            "fxnlms_only",
            "band_gated",
            "band_gated_fxnlms",
            "rule_gfanc",
            "oracle_gfanc",
            "gfanc_kalman",
            "gfanc_kalman_fxnlms",
            "full_system",
        ),
    )
    args = parser.parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    results = [run_simulation(mode, save_outputs=True) for mode in modes]

    print("Mode comparison completed.")
    print("mode                    broadband_dB   50-300Hz_dB   max_abs_y   status")
    for result in results:
        print(
            f"{result['mode']:<24} "
            f"{result['broadband_reduction_db']:>11.2f} "
            f"{result['band_reduction_db']:>13.2f} "
            f"{result['max_abs_y']:>11.3f} "
            f"{result['status']}"
        )
    print(f"Saved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()
