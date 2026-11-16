"""The signal constants in gwd.config keep the values that were hard-coded before (P3.1)."""

from gwd import config


def test_sample_constants_match_the_old_literals():
    assert config.SAMPLE_RATE == 4096
    assert config.DURATION == 4.0
    assert config.N_SAMPLES == 16384
    assert isinstance(config.N_SAMPLES, int)
    assert config.INPUT_SHAPE == (16384, 1)


def test_conditioning_constants_match_the_old_filter_chain():
    assert config.NOTCH_FREQS == (60, 120, 180, 240)
    assert config.BANDPASS == (20, 300)
    assert config.NOISE_FETCH_SECONDS == 4096
