from modelpack import ModelPack
from thermal import ThermalParams

def test_modelpack_roundtrip_defaults():
    mp = ModelPack()
    y = mp.to_yaml()
    mp2 = ModelPack.from_yaml(y)
    assert mp2.compound == mp.compound
    assert mp2.ambient_c == mp.ambient_c
    assert mp2.a1 == mp.a1 and mp2.c2 == mp.c2

def test_apply_modelpack_to_params():
    mp = ModelPack(compound="soft", ambient_c=30.0, track_c=45.0, a1=0.002, a2=0.0009, a3=0.11, a4=0.1, a5=0.05, b1=0.07, b2=0.095, c1=0.013, c2=0.03)
    p = ThermalParams(mp.a1, mp.a2, mp.a3, mp.a4, mp.a5, mp.b1, mp.b2, mp.c1, mp.c2)
    assert p.a1 == 0.002 and p.a5 == 0.05 and p.c2 == 0.03
