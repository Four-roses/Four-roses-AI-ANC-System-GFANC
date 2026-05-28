"""Primary and secondary acoustic path models for offline ANC simulation."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class FIRPath:
    """FIR acoustic path container."""

    def __init__(self, impulse_response: np.ndarray) -> None:
        """Create an FIR path.

        Args:
            impulse_response: FIR coefficients with shape (path_len,).

        Returns:
            None.
        """
        self.impulse_response = np.asarray(impulse_response, dtype=float)

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process a signal through the FIR path.

        Args:
            signal: Input signal with shape (num_samples,).

        Returns:
            output: Path output with shape (num_samples,).
        """
        return lfilter(self.impulse_response, [1.0], signal)

    def sample(self, input_buffer: np.ndarray) -> float:
        """Process one sample using a newest-first input buffer.

        Args:
            input_buffer: Newest-first signal buffer with shape (path_len,).

        Returns:
            output_sample: Scalar FIR output sample.
        """
        return float(np.dot(self.impulse_response, input_buffer))


def _decaying_random_fir(path_len: int, seed: int, direct_gain: float, scale: float) -> np.ndarray:
    """Create a stable synthetic FIR path.

    Args:
        path_len: Number of FIR taps.
        seed: Random seed.
        direct_gain: Gain of the first tap.
        scale: Random tail scale.

    Returns:
        impulse_response: FIR coefficients with shape (path_len,).
    """
    rng = np.random.default_rng(seed)
    decay = np.exp(-np.arange(path_len) / max(path_len / 6.0, 1.0))
    impulse_response = scale * rng.standard_normal(path_len) * decay
    impulse_response[0] += direct_gain
    norm = np.linalg.norm(impulse_response)
    if norm > 0.0:
        impulse_response = impulse_response / norm * abs(direct_gain)
    return impulse_response


def create_default_primary_path(path_len: int) -> FIRPath:
    """Create a synthetic primary path P(z).

    Args:
        path_len: Number of FIR taps.

    Returns:
        path: FIRPath with impulse response shape (path_len,).
    """
    impulse_response = _decaying_random_fir(
        path_len=path_len,
        seed=11,
        direct_gain=0.8,
        scale=0.25,
    )
    return FIRPath(impulse_response)


def create_default_secondary_path(path_len: int) -> FIRPath:
    """Create a synthetic true secondary path S(z).

    Args:
        path_len: Number of FIR taps.

    Returns:
        path: FIRPath with impulse response shape (path_len,).
    """
    impulse_response = _decaying_random_fir(
        path_len=path_len,
        seed=23,
        direct_gain=0.6,
        scale=0.18,
    )
    return FIRPath(impulse_response)


def create_estimated_secondary_path(true_secondary_path: FIRPath, perturbation: float = 0.0) -> FIRPath:
    """Create an estimated secondary path S_hat(z).

    Args:
        true_secondary_path: True secondary path with impulse response shape
            (secondary_path_len,).
        perturbation: Relative random perturbation level. A value of 0.0 makes
            S_hat identical to S.

    Returns:
        path: Estimated FIRPath with impulse response shape (secondary_path_len,).
    """
    impulse_response = true_secondary_path.impulse_response.copy()
    if perturbation > 0.0:
        rng = np.random.default_rng(37)
        impulse_response += perturbation * np.std(impulse_response) * rng.standard_normal(
            impulse_response.shape
        )
    return FIRPath(impulse_response)
