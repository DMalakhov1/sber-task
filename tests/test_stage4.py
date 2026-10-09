import numpy as np

from task1_brent.brent_lib.neural import fit_neural


def test_stage4_model_set_keeps_fixed_seed_and_shapes():
    values = np.exp(3 + 0.08 * np.sin(np.arange(240) / 14))
    for kind in ("mlp", "rnn", "lstm", "gru", "qrnn"):
        pred, _, meta = fit_neural(values, 210, kind, epochs=2, seeds=(11,))
        assert pred.shape == ((21, 19) if kind == "qrnn" else (21,))
        assert np.isfinite(pred).all()
        assert meta["selected_epoch"] >= 1
        assert "validation_loss" in meta
        assert meta["model"] == kind


def test_stage4_quantile_order_and_no_test_leakage():
    values = np.exp(2.5 + 0.1 * np.sin(np.arange(240) / 16))
    pred, _, meta = fit_neural(values, 210, "qrnn", epochs=2, seeds=(11,))
    assert np.diff(pred, axis=-1).min() >= -1e-8
    assert meta["max_training_target"] <= 210
    assert meta["inner_train_max_target"] < meta["inner_validation_first_origin"]
