"""Offline audio input/output helpers.

Realtime sounddevice I/O is intentionally excluded from the first stage.
"""

from __future__ import annotations

import numpy as np


def load_audio(path: str, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file placeholder.

    Args:
        path: Audio file path.
        target_fs: Optional target sample rate.

    Returns:
        audio: Array with shape (num_samples,) for mono audio.
        fs: Sample rate as an integer.
    """
    raise NotImplementedError("Audio loading will be implemented in a later stage.")


def save_audio(path: str, audio: np.ndarray, fs: int) -> None:
    """Save a mono audio signal placeholder.

    Args:
        path: Output audio file path.
        audio: Array with shape (num_samples,).
        fs: Sample rate as an integer.

    Returns:
        None.
    """
    raise NotImplementedError("Audio saving will be implemented in a later stage.")


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono placeholder.

    Args:
        audio: Array with shape (num_samples,) or (num_samples, num_channels).

    Returns:
        mono_audio: Array with shape (num_samples,).
    """
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError("audio must have shape (num_samples,) or (num_samples, num_channels).")
