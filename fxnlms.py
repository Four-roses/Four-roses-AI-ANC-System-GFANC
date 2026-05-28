"""FxNLMS residual adaptation for offline ANC simulation."""

from __future__ import annotations

import numpy as np


class FxNLMSResidualAdapter:
    """Residual adaptive controller for W_adapt."""

    def __init__(
        self,
        filter_len: int,
        step_size: float,
        regularization: float,
        leakage: float,
        rho_adapt: float,
        adapt_norm_limit: float,
        adapt_tap_clip: float | None = None,
    ) -> None:
        """Initialize the residual adapter.

        Args:
            filter_len: Adaptive FIR length.
            step_size: FxNLMS step size.
            regularization: Positive denominator regularization.
            leakage: Leakage factor applied to W_adapt.
            rho_adapt: Norm ratio limit relative to W_GFANC.
            adapt_norm_limit: Absolute norm limit used when W_GFANC is zero.
            adapt_tap_clip: Optional tap-wise absolute clipping value.

        Returns:
            None.
        """
        self.filter_len = filter_len
        self.step_size = step_size
        self.regularization = regularization
        self.leakage = leakage
        self.rho_adapt = rho_adapt
        self.adapt_norm_limit = adapt_norm_limit
        self.adapt_tap_clip = adapt_tap_clip
        self.w_adapt = np.zeros(filter_len, dtype=float)

    def update(self, error: float, filtered_x: np.ndarray, w_gfanc: np.ndarray) -> np.ndarray:
        """Update residual adaptive weights.

        Args:
            error: Scalar error sample e(n).
            filtered_x: Filtered reference vector with shape (filter_len,).
            w_gfanc: Base control filter with shape (filter_len,).

        Returns:
            w_adapt: Updated residual filter with shape (filter_len,).
        """
        if filtered_x.shape[0] != self.filter_len:
            raise ValueError("filtered_x must have shape (filter_len,).")
        if w_gfanc.shape[0] != self.filter_len:
            raise ValueError("w_gfanc must have shape (filter_len,).")

        denominator = self.regularization + float(np.dot(filtered_x, filtered_x))
        self.w_adapt = self.leakage * self.w_adapt
        self.w_adapt -= self.step_size * error * filtered_x / denominator
        self._apply_limits(w_gfanc)
        return self.w_adapt.copy()

    def final_filter(self, w_gfanc: np.ndarray) -> np.ndarray:
        """Return W_final = W_GFANC + W_adapt.

        Args:
            w_gfanc: Base control filter with shape (filter_len,).

        Returns:
            w_final: Final control filter with shape (filter_len,).
        """
        if w_gfanc.shape[0] != self.filter_len:
            raise ValueError("w_gfanc must have shape (filter_len,).")
        return w_gfanc + self.w_adapt

    def constrain(self, w_gfanc: np.ndarray) -> np.ndarray:
        """Apply residual limits relative to the current W_GFANC.

        Args:
            w_gfanc: Base control filter with shape (filter_len,).

        Returns:
            w_adapt: Constrained residual filter with shape (filter_len,).
        """
        if w_gfanc.shape[0] != self.filter_len:
            raise ValueError("w_gfanc must have shape (filter_len,).")
        self._apply_limits(w_gfanc)
        return self.w_adapt.copy()

    def _apply_limits(self, w_gfanc: np.ndarray) -> None:
        """Apply norm and tap-wise limits to W_adapt.

        Args:
            w_gfanc: Base control filter with shape (filter_len,).

        Returns:
            None.
        """
        if self.adapt_tap_clip is not None:
            self.w_adapt = np.clip(self.w_adapt, -self.adapt_tap_clip, self.adapt_tap_clip)

        w_gfanc_norm = float(np.linalg.norm(w_gfanc))
        if w_gfanc_norm > self.regularization:
            norm_limit = self.rho_adapt * w_gfanc_norm
        else:
            norm_limit = self.adapt_norm_limit

        adapt_norm = float(np.linalg.norm(self.w_adapt))
        if adapt_norm > norm_limit > 0.0:
            self.w_adapt *= (0.999 * norm_limit) / adapt_norm

    def get_filter(self) -> np.ndarray:
        """Return current residual filter.

        Args:
            None.

        Returns:
            w_adapt: Residual filter with shape (filter_len,).
        """
        return self.w_adapt.copy()
