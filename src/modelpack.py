from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml

@dataclass
class ModelPack:
    name: str = "default"
    compound: str = "medium"
    ambient_c: float = 27.0
    track_c: float = 39.0
    a1: float = 1.8e-3
    a2: float = 8.5e-4
    a3: float = 0.12
    a4: float = 0.08
    a5: float = 0.04
    b1: float = 0.07
    b2: float = 0.09
    c1: float = 0.012
    c2: float = 0.03

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True)

    @staticmethod
    def from_yaml(s: str) -> "ModelPack":
        base = asdict(ModelPack())
        d = yaml.safe_load(s) or {}
        base.update(d)
        return ModelPack(**base)
