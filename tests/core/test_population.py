"""gwd.core.population.random_bbh_params: bounds, laws, and the draws it makes (CORE-16)."""

import numpy as np
import pytest

from gwd.core.population import MAX_DISTANCE_MPC, random_bbh_params


@pytest.fixture
def drawn(rng):
    params = [random_bbh_params(rng) for _ in range(20000)]
    return {key: np.array([p[key] for p in params]) for key in params[0]}


def test_only_masses_and_distance_are_drawn(rng):
    """V1 is spinless and face-on, so no spin or inclination is drawn and dropped."""
    assert set(random_bbh_params(rng)) == {"mass1", "mass2", "distance"}


def test_the_parameters_stay_in_their_bounds(drawn):
    assert 5.0 <= drawn["mass2"].min()
    assert np.all(drawn["mass2"] <= drawn["mass1"])
    assert drawn["mass1"].max() <= 100.0
    assert (
        0.0 <= drawn["distance"].min() and drawn["distance"].max() <= MAX_DISTANCE_MPC
    )


def test_mass1_is_log_uniform(drawn):
    log_m1 = np.log10(drawn["mass1"])
    edges = np.linspace(np.log10(5.0), np.log10(100.0), 6)
    counts, _ = np.histogram(log_m1, edges)
    assert counts / counts.sum() == pytest.approx([0.2] * 5, abs=0.02)


def test_the_distance_is_uniform_in_volume(drawn):
    # Uniform in volume: P(d < x) = (x / d_max)^3, so (d / d_max)^3 is uniform.
    u = (drawn["distance"] / MAX_DISTANCE_MPC) ** 3
    assert np.mean(u < 0.5) == pytest.approx(0.5, abs=0.02)
    assert np.median(drawn["distance"]) == pytest.approx(
        MAX_DISTANCE_MPC * 0.5 ** (1 / 3), rel=0.03
    )


def test_a_mass_range_and_distance_can_be_given(rng):
    params = random_bbh_params(rng, masses=(30.0, 30.0), max_distance_mpc=10.0)
    assert params["mass1"] == pytest.approx(30.0)
    assert params["mass2"] == pytest.approx(30.0)
    assert params["distance"] <= 10.0


def test_three_numbers_are_drawn_in_this_order():
    rng = np.random.RandomState(4)
    params = random_bbh_params(rng)
    reference = np.random.RandomState(4)
    mass1 = 10 ** reference.uniform(np.log10(5.0), np.log10(100.0))
    mass2 = 10 ** reference.uniform(np.log10(5.0), np.log10(mass1))
    distance = reference.uniform(0, 1) ** (1 / 3) * MAX_DISTANCE_MPC
    assert params == {"mass1": mass1, "mass2": mass2, "distance": distance}
    assert rng.rand() == reference.rand()
