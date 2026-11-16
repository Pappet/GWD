"""gwd.core draws only from the rng it is given, never from the global numpy RNG (P5.3).

core:MISS-3 and DATA-5: before P5.3 gwd.core called np.random directly, so a result
was reproducible only by seeding the global RNG, and any other user of it changed the
draws.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from gwd.core.noise import colored_noise_from_psd, pink_psd, white_noise
from gwd.core.population import random_bbh_params
from gwd.core.simulation import TimeDomainSimulator

CORE = Path(__file__).resolve().parents[2] / "src" / "gwd" / "core"


def draw_everything(rng, backend="newton"):
    sim = TimeDomainSimulator(backend=backend, rng=rng, noise_psd=pink_psd)
    signal, _ = sim.generate_chirp(30, 30, 400)
    return [
        white_noise(4096, 1.0, rng=rng),
        colored_noise_from_psd(4096, 4096, pink_psd, rng=rng),
        np.array(list(random_bbh_params(rng).values())),
        sim.add_detector_noise(signal, 1.0),
    ]


@pytest.mark.parametrize(
    "make_rng",
    [np.random.default_rng, np.random.RandomState],
    ids=["Generator", "RandomState"],
)
def test_the_same_seed_gives_bit_identical_draws(make_rng):
    first = draw_everything(make_rng(7))
    second = draw_everything(make_rng(7))
    for a, b in zip(first, second, strict=True):
        assert a.tobytes() == b.tobytes()
    other = draw_everything(make_rng(8))
    assert first[0].tobytes() != other[0].tobytes()


def test_the_global_numpy_rng_is_neither_used_nor_changed():
    np.random.seed(0)
    state = np.random.get_state()
    draw_everything(np.random.default_rng(1))
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    assert np.array_equal(state[1], after[1])


def test_the_core_sources_do_not_call_the_global_numpy_rng():
    """The check of the P5.3 plan, as a test: no np.random.<draw>( in gwd.core."""
    pattern = re.compile(
        r"np[.]random[.](rand|randn|uniform|normal|randint|choice|seed)[(]"
    )
    hits = [
        f"{path.name}:{number}"
        for path in sorted(CORE.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []


def test_the_rng_is_required():
    with pytest.raises(TypeError, match="rng"):
        white_noise(10)
    with pytest.raises(TypeError, match="rng"):
        TimeDomainSimulator(backend="newton", noise_psd=pink_psd)
