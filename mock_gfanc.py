"""Band-energy gated fixed-filter selector.

This module intentionally avoids logits, sigmoid, soft alpha, Kalman, and
predictive weighting. It selects a small number of fixed filters from measured
band energy and returns binary alpha values.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


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


class BandEnergyGate:
    """Select fixed filters from smoothed band-energy estimates."""

    def __init__(
        self,
        fs: int,
        num_filters: int,
        bands: list[tuple[int, int]] | None = None,
        top_k: int = 3,
        threshold_ratio: float = 0.2,
        smoothing: float = 0.8,
        min_hold_frames: int = 2,
    ) -> None:
        """Initialize the band-energy gate.

        Args:
            fs: Sample rate as an integer.
            num_filters: Number of filters in the fixed filter bank.
            bands: Frequency bands in Hz with length num_filters.
            top_k: Maximum number of filters to enable.
            threshold_ratio: Enable bands whose smoothed energy is at least
                this ratio of the strongest band energy.
            smoothing: Exponential smoothing factor for band energy.
            min_hold_frames: Minimum number of frames to keep a selected alpha
                pattern before switching.

        Returns:
            None.
        """
        self.fs = fs
        self.num_filters = num_filters
        self.bands = bands or FILTER_BANDS
        self.top_k = max(1, min(top_k, num_filters))
        self.threshold_ratio = threshold_ratio
        self.smoothing = smoothing
        self.min_hold_frames = max(0, min_hold_frames)
        self.smoothed_energy = np.zeros(num_filters, dtype=float)
        self.current_alpha = np.zeros(num_filters, dtype=float)
        self.hold_count = 0
        if len(self.bands) != num_filters:
            raise ValueError("bands length must match num_filters.")

    def compute_energy(self, x_segment: np.ndarray) -> np.ndarray:
        """Compute raw energy in each configured frequency band.

        Args:
            x_segment: Reference frame with shape (frame_len,).

        Returns:
            energy: Raw band energy with shape (num_filters,).
        """
        nperseg = min(256, x_segment.shape[0])
        frequencies, psd = welch(x_segment, fs=self.fs, nperseg=nperseg)
        energy = np.asarray(
            [self._band_energy(frequencies, psd, band) for band in self.bands],
            dtype=float,
        )
        return energy

    def update(self, x_segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update the gate and return binary alpha.

        Args:
            x_segment: Reference frame with shape (frame_len,).

        Returns:
            alpha: Binary filter selection with shape (num_filters,).
            raw_energy: Raw band energy with shape (num_filters,).
            smooth_energy: Smoothed band energy with shape (num_filters,).
        """
        raw_energy = self.compute_energy(x_segment)
        self.smoothed_energy = (
            self.smoothing * self.smoothed_energy + (1.0 - self.smoothing) * raw_energy
        )
        proposed_alpha = self._select_alpha(self.smoothed_energy)
        if self.hold_count > 0:
            self.hold_count -= 1
            alpha = self.current_alpha.copy()
        else:
            alpha = proposed_alpha
            if not np.array_equal(alpha, self.current_alpha):
                self.hold_count = self.min_hold_frames
            self.current_alpha = alpha.copy()
        return alpha, raw_energy, self.smoothed_energy.copy()

    def _select_alpha(self, energy: np.ndarray) -> np.ndarray:
        """Select top-k high-energy bands as binary alpha.

        Args:
            energy: Smoothed band energy with shape (num_filters,).

        Returns:
            alpha: Binary filter selection with shape (num_filters,).
        """
        alpha = np.zeros(self.num_filters, dtype=float)
        max_energy = float(np.max(energy))
        if max_energy <= 1e-12:
            return alpha
        candidate_indices = np.where(energy >= self.threshold_ratio * max_energy)[0]
        if candidate_indices.size == 0:
            candidate_indices = np.asarray([int(np.argmax(energy))])
        ranked = candidate_indices[np.argsort(energy[candidate_indices])[::-1]]
        selected = ranked[: self.top_k]
        alpha[selected] = 1.0
        return alpha

    @staticmethod
    def _band_energy(frequencies: np.ndarray, psd: np.ndarray, band: tuple[int, int]) -> float:
        """Compute PSD energy in one frequency band.

        Args:
            frequencies: Frequency bins with shape (num_bins,).
            psd: Power spectral density with shape (num_bins,).
            band: Frequency band as (low_hz, high_hz).

        Returns:
            energy: Scalar band energy.
        """
        mask = (frequencies >= band[0]) & (frequencies < band[1])
        return float(np.sum(psd[mask]))
