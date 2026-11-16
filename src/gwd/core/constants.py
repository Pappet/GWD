"""Physical constants of gwd.core in SI units, in one place (P5.3, CORE-18).

The speed of light is LAL's (lal.C_SI), which PyCBC uses. Before P5.3 c and the laser
wavelength were repeated in the interferometer model and the time domain simulator.
Since P5.4 the interferometer model takes the Planck and the Boltzmann constant from
here too (CORE-18). G, the solar mass, and the megaparsec were here for the numpy-only
chirp backend until P6.1 deleted it (D1 = A); PyCBC brings its own.
"""

# Speed of light in vacuum, m/s (exact).
C = 299792458.0
# Wavelength of the Nd:YAG laser of LIGO-like detectors, m.
LASER_WAVELENGTH = 1064e-9
# Planck constant, J s (exact since the 2019 SI).
H = 6.62607015e-34
# Boltzmann constant, J / K (exact since the 2019 SI).
K_B = 1.380649e-23
# Arm length of the LIGO detectors, m: the default of the interferometer model and of
# the time domain simulator.
LIGO_ARM_LENGTH = 4000.0
