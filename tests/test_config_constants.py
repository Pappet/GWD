"""The constants in gwd.config keep the values that were hard-coded before (P3.1, P3.7)."""

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


def test_training_defaults_match_the_old_literals():
    assert config.EPOCHS == 40
    assert config.BATCH_SIZE == 128
    assert config.LEARNING_RATE == 0.0005
    assert config.EARLY_STOPPING_PATIENCE == 5
    assert config.REDUCE_LR_PATIENCE == 3
    assert config.TEST_SPLIT == 0.2
