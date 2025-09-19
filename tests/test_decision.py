import numpy as np
from decision import DecisionEngine

def test_engine_band_mapping():
    assert DecisionEngine("soft").band == (95.0, 110.0)
    assert DecisionEngine("medium").band == (90.0, 106.0)
    assert DecisionEngine("hard").band == (88.0, 104.0)
    # default fallback
    assert DecisionEngine("unknown").band == (90.0, 106.0)

def test_actions_generate_hot_and_cold_recs():
    eng = DecisionEngine("medium")
    # HOT tread
    recs_hot = eng.actions({"FL": np.array([120.0, 100.0, 90.0])})
    assert any("Tread" in m for _, m in recs_hot)
    # COLD tread
    recs_cold = eng.actions({"FR": np.array([80.0, 85.0, 90.0])})
    assert any("Tread" in m for _, m in recs_cold)
