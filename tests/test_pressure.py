import numpy as np
from pressure import suggest_pressure_delta

def test_pressure_delta_direction_and_clamp():
    band = (90.0, 106.0)
    # too hot -> suggest +psi (negative ΔT per +psi), but we return psi delta (positive to reduce heat)
    carc_hot = np.array([110.0, 111.0, 112.0])
    dpsi_hot = suggest_pressure_delta(carc_hot, band)
    assert dpsi_hot > 0
    # too cold -> suggest -psi
    carc_cold = np.array([80.0, 82.0, 83.0])
    dpsi_cold = suggest_pressure_delta(carc_cold, band)
    assert dpsi_cold < 0
    # clamp
    carc_very_hot = np.array([140.0, 142.0, 141.0])
    assert suggest_pressure_delta(carc_very_hot, band) <= 0.6
    carc_very_cold = np.array([10.0, 12.0, 11.0])
    assert suggest_pressure_delta(carc_very_cold, band) >= -0.6

def test_pressure_delta_empty_safe():
    dpsi = suggest_pressure_delta(np.array([]), (90.0, 106.0))
    assert dpsi == 0.0
