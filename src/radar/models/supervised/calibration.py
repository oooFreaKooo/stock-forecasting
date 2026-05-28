from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Isotonic regression calibrator for predicted probabilities."""

    def __init__(self) -> None:
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        self._calibrator.fit(y_prob, y_true)
        self._fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")
        return self._calibrator.transform(y_prob)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)


def calibrate_probabilities(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, ProbabilityCalibrator]:
    calibrator = ProbabilityCalibrator()
    calibrated = calibrator.fit_transform(raw_probs, y_true)
    return calibrated, calibrator
