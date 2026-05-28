"""End-to-end sanity checks for the band-gated fixed-filter ANC system."""

from __future__ import annotations

import os

import config
import numpy as np
from filter_bank import load_filter_bank
from fxnlms import FxNLMSResidualAdapter
from main_simulation import run_simulation
from mock_gfanc import BandEnergyGate
from path_model import (
    create_default_primary_path,
    create_default_secondary_path,
    create_estimated_secondary_path,
)


def status_line(name: str, status: str, detail: str = "") -> str:
    """Print and return a status line.

    Args:
        name: Check name.
        status: PASS, WARNING, SKIP, or FAIL.
        detail: Optional detail.

    Returns:
        status: The input status string.
    """
    suffix = f" - {detail}" if detail else ""
    print(f"{name}: {status}{suffix}")
    return status


def create_directories() -> str:
    """Create required project directories.

    Args:
        None.

    Returns:
        status: PASS if directories exist after creation.
    """
    for path in ("data", "data/noise", "data/paths", "data/filters", "checkpoints", "outputs"):
        os.makedirs(path, exist_ok=True)
    return status_line("directories", "PASS")


def check_config() -> str:
    """Check config.py numeric constraints.

    Args:
        None.

    Returns:
        status: PASS or FAIL.
    """
    checks = [
        config.fs > 0,
        config.filter_len > 0,
        config.secondary_path_len > 0,
        config.frame_len >= config.block_size,
        config.control_band[0] > 0,
        config.control_band[1] < config.fs / 2,
        config.y_limit > 0,
        config.rho_adapt >= 0,
        config.band_gate_top_k > 0,
        0.0 <= config.band_energy_smoothing < 1.0,
        config.w_gfanc_norm_limit > 0,
    ]
    return status_line("config", "PASS" if all(checks) else "FAIL")


def make_test_noise() -> np.ndarray:
    """Generate normalized 5-second low-frequency test noise.

    Args:
        None.

    Returns:
        x: Test signal with shape (num_samples,), normalized to [-0.5, 0.5].
    """
    num_samples = int(5.0 * config.fs)
    time = np.arange(num_samples) / config.fs
    x = (
        np.sin(2.0 * np.pi * 80.0 * time)
        + 0.7 * np.sin(2.0 * np.pi * 160.0 * time + 0.2)
        + 0.5 * np.sin(2.0 * np.pi * 240.0 * time + 0.5)
    )
    x += 0.02 * np.random.default_rng(42).standard_normal(num_samples)
    x = 0.5 * x / max(np.max(np.abs(x)), 1e-12)
    return x.astype(float)


def check_paths() -> tuple[str, object, object, object, object]:
    """Create and check P(z), S(z), S_hat(z), and perturbed S_hat(z).

    Args:
        None.

    Returns:
        result: Tuple (status, primary_path, secondary_path, secondary_path_hat,
            perturbed_secondary_path_hat).
    """
    primary = create_default_primary_path(config.primary_path_len)
    secondary = create_default_secondary_path(config.secondary_path_len)
    secondary_hat = create_estimated_secondary_path(secondary, perturbation=0.0)
    perturbed = create_estimated_secondary_path(secondary, perturbation=0.05)
    passed = (
        primary.impulse_response.ndim == 1
        and primary.impulse_response.shape[0] == config.primary_path_len
        and secondary.impulse_response.shape == (config.secondary_path_len,)
        and secondary_hat.impulse_response.shape == (config.secondary_path_len,)
        and perturbed.impulse_response.shape == (config.secondary_path_len,)
        and all(np.all(np.isfinite(path.impulse_response)) for path in (primary, secondary, secondary_hat, perturbed))
    )
    return status_line("paths", "PASS" if passed else "FAIL"), primary, secondary, secondary_hat, perturbed


def check_filter_bank() -> tuple[str, object | None]:
    """Check saved filter-bank file.

    Args:
        None.

    Returns:
        result: Tuple (status, filter_bank_or_none).
    """
    path = os.path.join("data", "filters", "filter_bank.npy")
    if not os.path.exists(path):
        return status_line("filter bank", "SKIP", "missing; run python train_filter_bank.py"), None
    bank = load_filter_bank(path)
    norms = np.linalg.norm(bank.filters, axis=1)
    passed = (
        bank.shape == (config.num_filters, config.filter_len)
        and np.all(np.isfinite(bank.filters))
        and np.all(norms > 0.0)
        and np.all(norms < 100.0)
    )
    detail = f"shape={bank.shape}, norm range=({norms.min():.3f}, {norms.max():.3f})"
    return status_line("filter bank", "PASS" if passed else "FAIL", detail), bank


def check_band_gate(bank, x: np.ndarray) -> str:
    """Check binary alpha and W_GFANC from band-energy gate.

    Args:
        bank: FixedFilterBank with filters shape (num_filters, filter_len).
        x: Test signal with shape (num_samples,).

    Returns:
        status: PASS, SKIP, or FAIL.
    """
    if bank is None:
        return status_line("band gate", "SKIP", "filter bank missing")
    gate = BandEnergyGate(
        config.fs,
        config.num_filters,
        top_k=config.band_gate_top_k,
        threshold_ratio=config.band_gate_threshold_ratio,
        smoothing=config.band_energy_smoothing,
        min_hold_frames=config.band_gate_min_hold_frames,
    )
    alpha, raw_energy, smooth_energy = gate.update(x[: config.frame_len])
    selected_count = max(int(np.sum(alpha)), 1)
    combine_alpha = alpha / selected_count if config.normalize_selected_filters else alpha
    w_gfanc = bank.combine(combine_alpha)
    passed = (
        alpha.shape == (config.num_filters,)
        and raw_energy.shape == (config.num_filters,)
        and smooth_energy.shape == (config.num_filters,)
        and w_gfanc.shape == (config.filter_len,)
        and set(np.unique(alpha)).issubset({0.0, 1.0})
        and np.sum(alpha) <= config.band_gate_top_k
        and np.all(np.isfinite(raw_energy))
        and np.all(np.isfinite(smooth_energy))
        and np.all(np.isfinite(w_gfanc))
    )
    return status_line("band gate", "PASS" if passed else "FAIL", f"alpha={alpha.astype(int).tolist()}")


def check_fxnlms(primary, secondary, secondary_hat, x: np.ndarray) -> str:
    """Run a minimal FxNLMS closed-loop check.

    Args:
        primary: Primary FIRPath P(z).
        secondary: True secondary FIRPath S(z).
        secondary_hat: Estimated secondary FIRPath S_hat(z).
        x: Test signal with shape (num_samples,).

    Returns:
        status: PASS or FAIL.
    """
    x = x[: 2 * config.frame_len]
    d = primary.process(x)
    adapter = FxNLMSResidualAdapter(
        config.filter_len,
        config.step_size,
        config.regularization,
        config.leakage,
        config.rho_adapt,
        config.adapt_norm_limit,
        config.adapt_tap_clip,
    )
    x_buffer = np.zeros(config.filter_len)
    filtered_x_buffer = np.zeros(config.filter_len)
    x_hat_buffer = np.zeros(config.secondary_path_len)
    y_buffer = np.zeros(config.secondary_path_len)
    w_gfanc = np.zeros(config.filter_len)
    y_out = np.zeros_like(x)
    e = np.zeros_like(x)
    for n, sample in enumerate(x):
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = sample
        x_hat_buffer[1:] = x_hat_buffer[:-1]
        x_hat_buffer[0] = sample
        filtered_x_buffer[1:] = filtered_x_buffer[:-1]
        filtered_x_buffer[0] = secondary_hat.sample(x_hat_buffer)
        w_final = adapter.final_filter(w_gfanc)
        y = float(np.clip(np.dot(w_final, x_buffer), -config.y_limit, config.y_limit))
        y_out[n] = y
        y_buffer[1:] = y_buffer[:-1]
        y_buffer[0] = y
        e[n] = d[n] + secondary.sample(y_buffer)
        adapter.update(e[n], filtered_x_buffer, w_gfanc)
    w_adapt = adapter.get_filter()
    passed = (
        np.max(np.abs(y_out)) <= config.y_limit + 1e-9
        and np.all(np.isfinite(e))
        and np.linalg.norm(w_adapt) <= config.adapt_norm_limit + 1e-6
        and adapter.final_filter(w_gfanc).shape == (config.filter_len,)
    )
    return status_line("fxnlms", "PASS" if passed else "FAIL")


def check_end_to_end() -> tuple[str, list[dict[str, object]]]:
    """Run required end-to-end modes and classify results.

    Args:
        None.

    Returns:
        result: Tuple (overall_status, result_dicts).
    """
    modes = ["fxnlms_only", "band_gated", "band_gated_fxnlms"]
    results = []
    overall = "PASS"
    for mode in modes:
        try:
            result = run_simulation(mode, save_outputs=False)
        except FileNotFoundError as exc:
            print(f"{mode}: SKIP - {exc}")
            continue
        results.append(result)
        if result["status"] == "FAIL":
            overall = "FAIL"
        elif result["status"] == "WARNING" and overall != "FAIL":
            overall = "WARNING"
        if mode == "fxnlms_only" and result["band_reduction_db"] < -0.5:
            overall = "FAIL"
            result["status"] = "FAIL"
        print(
            f"{mode}: {result['status']} | broadband={result['broadband_reduction_db']:.2f} dB, "
            f"50-300Hz={result['band_reduction_db']:.2f} dB, "
            f"max|y|={result['max_abs_y']:.3f}, max|e|={result['max_abs_e']:.3f}, "
            f"Wg={result['W_gfanc_norm']:.3f}, Wa={result['W_adapt_norm']:.3f}, "
            f"nan/inf={result['nan_or_inf']}, diverged={result['diverged']}"
        )
    return status_line("end-to-end", overall), results


def main() -> None:
    """Run the complete sanity check suite.

    Args:
        None.

    Returns:
        None.
    """
    report = {}
    report["directories"] = create_directories()
    report["config"] = check_config()
    x = make_test_noise()
    print(f"test noise: PASS - shape={x.shape}, range=({x.min():.3f}, {x.max():.3f})")
    paths_status, primary, secondary, secondary_hat, _ = check_paths()
    report["paths"] = paths_status
    report["filter bank"], bank = check_filter_bank()
    report["band gate"] = check_band_gate(bank, x)
    report["fxnlms"] = check_fxnlms(primary, secondary, secondary_hat, x)
    report["end-to-end"], _ = check_end_to_end()

    print("========================")
    print("SANITY CHECK REPORT")
    for name in ("config", "paths", "filter bank", "band gate", "fxnlms", "end-to-end"):
        print(f"{name}: {report[name]}")
    print("========================")


if __name__ == "__main__":
    main()
