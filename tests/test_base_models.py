from __future__ import annotations

import warnings

import numpy as np
import pytest

from radar.ensemble.base_models import predict_proba, train_lightgbm, train_logistic


def test_train_logistic_large_scale_no_runtime_warnings():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 24))
    X[:, 0] *= 1_000_000
    X[:, 1] *= 500_000
    y = (rng.random(300) > 0.48).astype(int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = train_logistic(X, y, feature_names=[f"feat_{idx}" for idx in range(X.shape[1])])
        probs = predict_proba(model, X[:10])

    runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_lightgbm_predict_no_feature_name_warning():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (120, 8))
    y = (rng.random(120) > 0.5).astype(int)
    feature_names = [f"feature_{idx}" for idx in range(X.shape[1])]

    model = train_lightgbm(X, y, feature_names=feature_names)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probs = predict_proba(model, X[:5])

    name_warnings = [
        item for item in caught
        if "feature names" in str(item.message).lower()
    ]
    assert len(name_warnings) == 0
    assert np.isfinite(probs).all()
