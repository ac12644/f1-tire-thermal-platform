import numpy as np
from thermal import ThermalModel, ThermalParams

def test_thermal_step_shapes_and_types():
    p = ThermalParams()
    m = ThermalModel(p)
    x = np.array([95.0, 90.0, 85.0], dtype=float)
    u = dict(slip=0.06, slip_ang=3.0, load=4200.0, speed_kmh=180.0, brake=0.3, Ta=25.0, Ttrack=35.0)
    y = m.step(x, u, dt=0.2)
    assert isinstance(y, np.ndarray)
    assert y.shape == (3,)
    assert np.isfinite(y).all()

def test_cooling_factor_scales_airflow():
    p = ThermalParams()
    m = ThermalModel(p)
    x = np.array([100.0, 95.0, 90.0], dtype=float)
    base = m.step(x, dict(speed_kmh=200.0, Ta=25.0, Ttrack=35.0, slip=0.05, slip_ang=3.0, load=4000.0, brake=0.2, cooling_factor=1.0), 0.2)
    less_cooling = m.step(x, dict(speed_kmh=200.0, Ta=25.0, Ttrack=35.0, slip=0.05, slip_ang=3.0, load=4000.0, brake=0.2, cooling_factor=0.5), 0.2)
    # With less cooling, tread should increase more (or drop less)
    assert less_cooling[0] >= base[0] - 1e-9

