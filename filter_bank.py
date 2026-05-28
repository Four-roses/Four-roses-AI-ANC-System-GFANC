"""Fixed control filter bank utilities."""

from __future__ import annotations

import numpy as np


class FixedFilterBank:
    """Container for fixed ANC control filters."""

    def __init__(self, filters: np.ndarray) -> None:
        """Create a fixed filter bank.

        Args:
            filters: Fixed filters with shape (num_filters, filter_len).

        Returns:
            None.
        """
        if filters.ndim != 2:
            raise ValueError("filters must have shape (num_filters, filter_len).")
        self.filters = np.asarray(filters, dtype=float)

    def combine(self, alpha: np.ndarray) -> np.ndarray:
        """Combine fixed filters using activation weights.

        Args:
            alpha: Filter weights with shape (num_filters,).

        Returns:
            w_gfanc: Combined control filter with shape (filter_len,).
        """
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.filters.shape[0],):
            raise ValueError("alpha must have shape (num_filters,).")
        return alpha @ self.filters

    @property
    def shape(self) -> tuple[int, int]:
        """Return filter-bank shape.

        Args:
            None.

        Returns:
            shape: Tuple (num_filters, filter_len).
        """
        return self.filters.shape


def create_placeholder_filter_bank(num_filters: int, filter_len: int) -> FixedFilterBank:
    """Create a zero-valued placeholder filter bank.

    Args:
        num_filters: Number of fixed filters.
        filter_len: FIR length for each filter.

    Returns:
        bank: FixedFilterBank containing filters with shape (num_filters, filter_len).
    """
    return FixedFilterBank(np.zeros((num_filters, filter_len), dtype=float))


def save_filter_bank(path: str, bank: FixedFilterBank) -> None:
    """Save a fixed filter bank to a NumPy file.

    Args:
        path: Output ``.npy`` file path.
        bank: FixedFilterBank with filters shape (num_filters, filter_len).

    Returns:
        None.
    """
    np.save(path, bank.filters)


def load_filter_bank(path: str) -> FixedFilterBank:
    """Load a fixed filter bank from a NumPy file.

    Args:
        path: Input ``.npy`` file path.

    Returns:
        bank: FixedFilterBank with filters shape (num_filters, filter_len).
    """
    return FixedFilterBank(np.load(path))
