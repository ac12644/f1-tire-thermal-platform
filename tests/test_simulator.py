import numpy as np
from simulator import TelemetrySim, CORNER_KEYS

def test_snapshot_controls_pure_and_bounds():
    sim = TelemetrySim(seed=123)
    t_before = sim.t
    u1, loads1 = sim.snapshot_controls(t_before + 1.0)
    u2, loads2 = sim.snapshot_controls(t_before + 1.0)  # same time -> same values
    assert sim.t == t_before  # must not mutate internal time
    # deterministic at same t
    assert u1 == u2
    assert loads1.keys() == set(CORNER_KEYS)
    # basic bounds
    assert u1["speed_kmh"] >= 100
    assert 0.0 <= u1["brake"] <= 1.0
    for v in loads1.values():
        assert v > 0.0

def test_step_mutates_time_and_returns_sensors():
    sim = TelemetrySim(seed=7)
    t0 = sim.t
    u_common, loads, sensors = sim.step(0.2)
    assert sim.t > t0
    assert isinstance(sensors, dict) and set(sensors.keys()) == set(CORNER_KEYS)
    # sensors have tpms and hub
    k0 = CORNER_KEYS[0]
    assert "tpms" in sensors[k0] and "hub" in sensors[k0]
