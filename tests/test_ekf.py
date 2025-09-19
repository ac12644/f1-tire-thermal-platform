import numpy as np
from thermal import ThermalModel, ThermalParams
from ekf import CornerEKF

def test_ekf_converges_towards_measurements():
    m = ThermalModel(ThermalParams())
    ekf = CornerEKF(lambda x,u,dt: m.step(x,u,dt))
    # True state (unknown to EKF)
    true = np.array([100.0, 95.0, 90.0])
    # Measurements are Tt (tpms proxy) and Tr (hub proxy)
    u = dict(slip=0.05, slip_ang=3.0, load=4200.0, speed_kmh=180.0, brake=0.2, Ta=25.0, Ttrack=35.0)
    for _ in range(10):
        z = np.array([true[0], true[2]])  # tpms, hub
        x = ekf.step(z, u, dt=0.2)
    # After updates, estimate should be close in the measured channels
    assert abs(x[0] - true[0]) < 5.0
    assert abs(x[2] - true[2]) < 5.0
