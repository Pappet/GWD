"""Physical constants of gwd.core in SI units, in one place (P5.3, CORE-18).

The values are those of LAL (lal.C_SI, lal.G_SI, lal.MSUN_SI, lal.PC_SI * 1e6), which
PyCBC uses, so the Newtonian approximation and PyCBC share their units. Before P5.3
the Newtonian chirp used rounded values (c = 2.998e8, G = 6.674e-11,
M_SUN = 1.989e30, Mpc = 3.086e22), and c and the laser wavelength were repeated in
the interferometer model and the time domain simulator.
"""

# Speed of light in vacuum, m/s (exact).
C = 299792458.0
# Newtonian constant of gravitation, m^3 / (kg s^2) (CODATA 2018).
G = 6.67430e-11
# Solar mass, kg (IAU 2015 nominal GM_sun divided by G).
M_SUN = 1.9884098706980507e30
# Megaparsec, m (IAU 2015 parsec times 1e6).
MPC = 3.085677581491367e22
# Wavelength of the Nd:YAG laser of LIGO-like detectors, m.
LASER_WAVELENGTH = 1064e-9
