import importlib
import os
import pytest

# Mark TF-dependent tests to be safely skipped if TF not installed
try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from stretch_detector.models.cnn_lstm import build_cnn_lstm
from stretch_detector.models.cnn3d import build_cnn3d


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_build_cnn_lstm():
    model = build_cnn_lstm((10, 250, 250, 1))
    assert model.output_shape[-1] == 1


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_build_cnn3d():
    model = build_cnn3d((10, 250, 250, 1))
    assert model.output_shape[-1] == 1
