"""GWOSC access: one checked download function and the shared event list (P3.4).

Next to gwd.data.preprocessing, this is the only module of the package that touches
gwpy, and it imports gwpy only inside fetch_strain, so importing it needs no gwpy.
"""

from dataclasses import dataclass

import numpy as np


class StrainError(ValueError):
    """Downloaded strain that cannot be used: wrong length or NaN values."""


def fetch_strain(detector, start, end, pad_s=0, cache=True, verbose=False):
    """Download GWOSC strain for [start - pad_s, end + pad_s) and check it.

    Returns the gwpy TimeSeries as downloaded, without conditioning. pad_s=0 gives
    exactly [start, end). With cache=True, gwpy keeps a local copy of the downloaded
    file and reads it from there next time.

    Raises StrainError if the number of samples does not match the requested span at
    the series' sample rate, or if the data contain NaN. Nothing is resized or padded.
    """
    from gwpy.timeseries import TimeSeries

    if pad_s < 0:
        raise ValueError(f"pad_s must not be negative, got {pad_s}")
    first, last = start - pad_s, end + pad_s
    strain = TimeSeries.fetch_open_data(
        detector, first, last, cache=cache, verbose=verbose
    )

    expected = round((last - first) * strain.sample_rate.value)
    if len(strain) != expected:
        raise StrainError(
            f"{detector} {first}-{last}: got {len(strain)} samples, expected {expected}"
        )
    bad = int(np.count_nonzero(np.isnan(strain.value)))
    if bad:
        raise StrainError(f"{detector} {first}-{last}: {bad} NaN samples")
    return strain


@dataclass(frozen=True)
class Event:
    """A detected event. Masses are rough source-frame values in solar masses."""

    name: str
    gps: float
    kind: str  # "BBH", "BNS", or "NSBH"
    mass1: float
    mass2: float
    # Both masses inside the 5-100 solar masses of the training set
    # (gwd.core.waveforms.generate_random_bbh_parameters).
    in_training_range: bool


# Merger GPS times as used by the field test and the Glitch Hunter before P3.4.
# The masses are rounded median source-frame values from the GWTC catalogs
# (GWTC-1 for O1/O2, GWTC-2 for O3a), written down from the literature and not
# verified offline against GWOSC.
EVENTS = (
    Event("GW150914", 1126259462.4, "BBH", 35.6, 30.6, True),
    Event("GW151226", 1135136350.6, "BBH", 13.7, 7.7, True),
    Event("GW170104", 1167559936.6, "BBH", 30.8, 20.0, True),
    Event("GW170608", 1180922494.5, "BBH", 11.0, 7.6, True),
    Event("GW170814", 1186741861.5, "BBH", 30.6, 25.2, True),
    # Binary neutron star, far below the training masses: out of distribution.
    Event("GW170817", 1187008882.4, "BNS", 1.46, 1.27, False),
    Event("GW190412", 1239082262.1, "BBH", 30.1, 8.3, True),
    Event("GW190521", 1242442967.4, "BBH", 85.0, 66.0, True),
    # The 2.6 solar mass secondary is either the heaviest neutron star or the
    # lightest black hole known; listed as NSBH here. Out of distribution.
    Event("GW190814", 1249852257.0, "NSBH", 23.2, 2.6, False),
)


def event(name):
    """Return the event with this name from EVENTS."""
    for candidate in EVENTS:
        if candidate.name == name:
            return candidate
    raise KeyError(f"Unknown event {name!r}; known: {[e.name for e in EVENTS]}")
