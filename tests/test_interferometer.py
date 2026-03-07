import numpy as np
from src.gwd_core.interferometer import InterferometerModel

def test_interferometer_default_sensitivity():
    model = InterferometerModel()
    sensitivity = model.calculate_total_sensitivity()
    
    assert len(sensitivity) == len(model.freq)
    assert np.all(sensitivity > 0)
    
def test_interferometer_features():
    model = InterferometerModel()
    model.power_recycling = True
    model.signal_recycling = True
    model.squeezed_light = True
    
    sensitivity_advanced = model.calculate_total_sensitivity()
    sensitivity_base = InterferometerModel().calculate_total_sensitivity()
    
    assert len(sensitivity_advanced) == len(model.freq)
    assert np.all(sensitivity_advanced > 0)
    
    # Advanced features generally lower the noise loop or change its shape
    # We just ensure it runs without errors and produces valid output
    assert np.any(sensitivity_advanced != sensitivity_base)
