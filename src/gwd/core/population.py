"""Random binary black hole parameters (P5.3, CORE-16).

The distributions are simple choices, not fits to an observed population such as the
O3 catalogue, which the old docstring of generate_random_bbh_parameters claimed.
"""

import numpy as np

# Default mass range in solar masses, and the largest distance in Mpc.
MASS_RANGE = (5.0, 100.0)
MAX_DISTANCE_MPC = 2000.0


def random_bbh_params(rng, masses=MASS_RANGE, max_distance_mpc=MAX_DISTANCE_MPC):
    """Draw mass1, mass2, and distance of one binary from rng, in this order.

    - mass1: log-uniform in masses, so small masses are drawn more often;
    - mass2: log-uniform between masses[0] and mass1, so mass2 <= mass1;
    - distance: uniform in volume up to max_distance_mpc, so far binaries are drawn
      more often. It is a metadata column only: the pipeline normalizes the chirp to
      peak 1 and scales it to an SNR, so the distance does not change a sample
      (DATA-18).

    Spins and the inclination are not drawn: V1 chirps are spinless and face-on. Before
    P5.3 they were drawn and then dropped by every caller.

    rng is a numpy Generator or RandomState; the same seed gives the same parameters.
    Returns a dict with mass1 and mass2 in solar masses and distance in Mpc.
    """
    low, high = masses
    mass1 = 10 ** rng.uniform(np.log10(low), np.log10(high))
    mass2 = 10 ** rng.uniform(np.log10(low), np.log10(mass1))
    distance = rng.uniform(0, 1) ** (1 / 3) * max_distance_mpc
    return {"mass1": mass1, "mass2": mass2, "distance": distance}
