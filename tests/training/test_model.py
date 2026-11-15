"""The CNN keeps its architecture, and building it changes no global state (P3.7)."""

import pytest

tf = pytest.importorskip("tensorflow")

from gwd.config import INPUT_SHAPE, LEARNING_RATE  # noqa: E402
from gwd.training.model import build_model  # noqa: E402

# DATA-26 (97.6 % of the parameters in one Dense layer) is left alone on purpose.
# A new architecture changes this number in a commit of its own.
PARAMETERS = 1_074_609


@pytest.fixture(scope="module")
def model():
    return build_model((16384, 1))


def test_parameter_count_is_unchanged(model):
    assert model.count_params() == PARAMETERS


def test_input_and_output_shapes(model):
    assert INPUT_SHAPE == (16384, 1)
    assert model.input_shape == (None, 16384, 1)
    assert model.output_shape == (None, 1)


def test_output_layer_is_float32(model):
    assert model.layers[-1].dtype_policy.compute_dtype == "float32"


def test_building_leaves_the_precision_policy_alone(model):
    assert tf.keras.mixed_precision.global_policy().name == "float32"


def test_learning_rate_defaults_to_config_and_can_be_set():
    assert float(build_model((64, 1)).optimizer.learning_rate) == pytest.approx(
        LEARNING_RATE
    )
    assert float(build_model((64, 1), 1e-3).optimizer.learning_rate) == pytest.approx(
        1e-3
    )
