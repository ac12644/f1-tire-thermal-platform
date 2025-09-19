# F1 Tire Temperature Management Prototype

This document explains the architecture, modules, and testing approach for the **F1-style Tire Temperature Management** prototype.

---

## Screenshot

![UI Screenshot](img/screenshot.png)

---

## 1. Overview

The prototype simulates and monitors tire thermal behavior in a Formula 1 context.  
It provides:

- **Live telemetry simulation** with tread, carcass, and rim temperatures.
- **Kalman-filter-based estimation (EKF)** fusing noisy sensor signals (TPMS & hub).
- **Decision engine** that issues recommendations (e.g., adjust brake bias, driving style).
- **Pit stop and modelpack system** to switch compounds and apply calibration presets.
- **Streamlit UI** for interactive charts, What-If simulation, and data export.

---

## 2. Modules

### `thermal.py`

- Implements a **3-node thermal model** (Tread–Carcass–Rim).
- Uses simplified differential equations driven by slip ratio, slip angle, load, and ambient/track temps.
- Tuned with coefficients (`ThermalParams`).

### `ekf.py`

- Extended Kalman Filter for each tire corner.
- State vector: `[Tt, Tc, Tr]` = Tread, Carcass, Rim.
- Measurements:
  - `m=1`: Tread only (`Tt`).
  - `m=2`: Tread + Rim (`Tt`, `Tr`) → matches TPMS & hub sensors.
  - `m=3`: All states observed.
- Adjusted default measurement covariance **R = 0.3** for faster convergence.

### `simulator.py`

- Generates synthetic telemetry (`TelemetrySim`).
- `step(dt)`: Advances time, produces loads & noisy sensors.
- `snapshot_controls(t)`: Pure snapshot of control inputs at time `t` (used in What-If, no side-effects).

### `decision.py`

- Maps compound → operating band (Soft: 95–110 °C, Medium: 90–106 °C, Hard: 88–104 °C).
- Provides **actions** if tires are above/below bands.

### `modelpack.py`

- Defines `ModelPack` (YAML-based parameter presets).
- Used to apply **track-specific calibration** (e.g., Monza soft/medium packs).

### `pressure.py`

- Suggests pressure adjustments (`dpsi`) based on carcass temps.
- Clamped to ±0.6 psi for realism.

### `events.py`

- Detects lockups (brake + slip spike) and slides (high slip angle vs. lateral g).

### `app_streamlit.py`

- Frontend using **Streamlit + Plotly**.
- Views:
  - **Live:** telemetry charts, recommendations, track status, metrics table.
  - **What-If:** forward simulation with adjustable controls.
  - **Session:** event log (pit stops, lockups, slides).
  - **Export:** download CSV of recent history.
- Modelpack expander: load preset/upload YAML/download snapshot.

---

## 3. Key Fixes

- **Scroll persistence**: small JS snippet saves/loads scroll position.
- **Run toggle bug**: prevented auto-pause from mutating `session_state`.
- **Compound sync logic**: fixed to avoid dropdown overriding pit stop or modelpack changes.
- **What-If purity**: uses `snapshot_controls()` instead of advancing live sim.
- **EKF measurement mapping**: explicitly maps `[Tt, Tr]` for m=2.
- **Measurement noise (R)**: tuned from 2.5 → 0.3 to ensure tests pass (<5 °C error).

---

## 4. Testing

Located under `tests/` with **pytest**.

- `test_thermal.py`: verifies thermal step produces valid numbers and cooling scaling.
- `test_ekf.py`: checks EKF converges toward Tt & Tr with repeated updates.
- `test_simulator.py`: ensures `step()` mutates time but `snapshot_controls()` does not.
- `test_pressure.py`: validates pressure delta direction (hot → +psi, cold → -psi).
- `test_modelpack.py`: roundtrip YAML serialization/deserialization.
- `test_decision.py`: band mapping + action triggers.

Run tests:

```bash
pytest -q
```
