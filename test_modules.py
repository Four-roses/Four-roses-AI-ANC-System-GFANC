"""Module-level stability and shape tests for the ANC project."""

from __future__ import annotations

import config
import numpy as np
from evaluate import bandpass_signal, compute_band_reduction_db, compute_power_db
from filter_bank import FixedFilterBank
from fxnlms import FxNLMSResidualAdapter
from mock_gfanc import BandEnergyGate
from path_model import create_default_primary_path, create_default_secondary_path


def _print_result(name: str, passed: bool, detail: str = "") -> bool:
    """Print one test result.

    Args:
        name: Test name.
        passed: True when the test passed.
        detail: Optional diagnostic detail.

    Returns:
        passed: Same boolean passed to the function.
    """
    suffix = f" - {detail}" if detail else ""
    print(f"{name}: {'PASS' if passed else 'FAIL'}{suffix}")
    return passed


def test_bandpass_filter() -> bool:
    """Test band-pass filtering shape and finite output.

    Args:
        None.

    Returns:
        passed: True if the test passes.
    """
    x = np.random.default_rng(1).standard_normal(config.fs)
    y = bandpass_signal(x, config.fs, config.control_band)
    return _print_result("test_bandpass_filter", y.shape == x.shape and np.all(np.isfinite(y)))


def test_path_generation() -> bool:
    """Test synthetic path generation.

    Args:
        None.

    Returns:
        passed: True if path shapes are valid.
    """
    primary = create_default_primary_path(config.primary_path_len)
    secondary = create_default_secondary_path(config.secondary_path_len)
    passed = (
        primary.impulse_response.ndim == 1
        and primary.impulse_response.shape[0] == config.primary_path_len
        and secondary.impulse_response.shape == (config.secondary_path_len,)
        and np.all(np.isfinite(primary.impulse_response))
        and np.all(np.isfinite(secondary.impulse_response))
    )
    return _print_result("test_path_generation", passed)


def test_filter_bank_combine() -> bool:
    """Test fixed filter-bank weighted combination.

    Args:
        None.

    Returns:
        passed: True if combination shape is valid and finite.
    """
    filters = np.ones((config.num_filters, config.filter_len), dtype=float) * 0.01
    bank = FixedFilterBank(filters)
    alpha = np.zeros(config.num_filters)
    alpha[:2] = 1.0
    combined = bank.combine(alpha / max(np.sum(alpha), 1.0))
    passed = combined.shape == (config.filter_len,) and np.all(np.isfinite(combined))
    return _print_result("test_filter_bank_combine", passed)


def test_band_energy_gate_output() -> bool:
    """Test binary band-energy alpha output.

    Args:
        None.

    Returns:
        passed: True if alpha and energy are finite with correct shape.
    """
    gate = BandEnergyGate(
        fs=config.fs,
        num_filters=config.num_filters,
        top_k=config.band_gate_top_k,
        threshold_ratio=config.band_gate_threshold_ratio,
        smoothing=config.band_energy_smoothing,
        min_hold_frames=config.band_gate_min_hold_frames,
    )
    frame = np.sin(2.0 * np.pi * 80.0 * np.arange(config.frame_len) / config.fs)
    alpha, raw_energy, smooth_energy = gate.update(frame)
    passed = (
        alpha.shape == (config.num_filters,)
        and raw_energy.shape == (config.num_filters,)
        and smooth_energy.shape == (config.num_filters,)
        and np.all(np.isfinite(raw_energy))
        and np.all(np.isfinite(smooth_energy))
        and set(np.unique(alpha)).issubset({0.0, 1.0})
        and np.sum(alpha) <= config.band_gate_top_k
    )
    return _print_result("test_band_energy_gate_output", passed)


def test_fxnlms_one_block() -> bool:
    """Test one block of FxNLMS residual updates.

    Args:
        None.

    Returns:
        passed: True if W_adapt remains finite and constrained.
    """
    adapter = FxNLMSResidualAdapter(
        config.filter_len,
        config.step_size,
        config.regularization,
        config.leakage,
        config.rho_adapt,
        config.adapt_norm_limit,
        config.adapt_tap_clip,
    )
    w_gfanc = np.ones(config.filter_len) * 0.01
    filtered_x = np.ones(config.filter_len) * 0.05
    for _ in range(config.block_size):
        adapter.update(0.1, filtered_x, w_gfanc)
    w_adapt = adapter.get_filter()
    limit = config.rho_adapt * max(np.linalg.norm(w_gfanc), 1e-6)
    passed = (
        w_adapt.shape == (config.filter_len,)
        and np.all(np.isfinite(w_adapt))
        and np.linalg.norm(w_adapt) <= limit + 1e-6
    )
    return _print_result("test_fxnlms_one_block", passed)


def test_evaluate_metrics() -> bool:
    """Test evaluation metric functions.

    Args:
        None.

    Returns:
        passed: True if metrics are finite.
    """
    before = np.ones(config.fs) * 0.1
    after = before * 0.5
    power_db = compute_power_db(before)
    reduction = compute_band_reduction_db(
        before + 0.01 * np.random.randn(config.fs),
        after,
        config.fs,
        config.control_band,
    )
    passed = np.isfinite(power_db) and np.isfinite(reduction)
    return _print_result("test_evaluate_metrics", passed)


def main() -> None:
    """Run all module-level tests.

    Args:
        None.

    Returns:
        None.
    """
    tests = [
        test_bandpass_filter,
        test_path_generation,
        test_filter_bank_combine,
        test_band_energy_gate_output,
        test_fxnlms_one_block,
        test_evaluate_metrics,
    ]
    results = [test() for test in tests]
    print(f"MODULE TESTS: {'PASS' if all(results) else 'FAIL'}")


if __name__ == "__main__":
    main()
