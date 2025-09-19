from __future__ import annotations
import time
import io
import csv
from pathlib import Path
from typing import Dict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit.components.v1 import html

from thermal import ThermalModel, ThermalParams
from ekf import CornerEKF
from simulator import TelemetrySim, CORNER_KEYS
from decision import DecisionEngine
from modelpack import ModelPack
from pressure import suggest_pressure_delta
from events import detect_lockup, detect_slide

# -------------------------------------------------------------------
# Page config MUST be first Streamlit command
# -------------------------------------------------------------------
st.set_page_config(page_title="🏎️ F1-Style Tire Temperature Management – Prototype", layout="wide")
st.title("🏎️ F1-Style Tire Temperature Management – Prototype")

# Restore scroll position after refresh
html(
    """
    <script>
      const KEY = "scrollY";
      document.addEventListener("scroll", () => {
        try { localStorage.setItem(KEY, String(window.scrollY)); } catch(e){}
      }, {passive:true});
      setTimeout(() => {
        try {
          const y = parseFloat(localStorage.getItem(KEY) || "0");
          if (!isNaN(y)) window.scrollTo(0, y);
        } catch(e){}
      }, 50);
    </script>
    """,
    height=0,
)

# -------------------------------------------------------------------
# Session helpers
# -------------------------------------------------------------------
def init_session():
    params = ThermalParams()
    model = ThermalModel(params)
    st.session_state.model = model
    st.session_state.sim = TelemetrySim()
    st.session_state.ekf = {k: CornerEKF(lambda x,u,dt: model.step(x,u,dt)) for k in CORNER_KEYS}
    st.session_state.hist = {k: [] for k in CORNER_KEYS}   # list of np.array([Tt,Tc,Tr])
    st.session_state.events = []                           # (tick, type, details)
    st.session_state.tick = 0
    st.session_state.last_slip = 0.05
    st.session_state.last_sa = 2.0
    # track active compound independent from widget
    st.session_state.compound_active = st.session_state.get("compound", "medium")

def ensure_session():
    if "model" not in st.session_state or "sim" not in st.session_state or "ekf" not in st.session_state:
        init_session()

ensure_session()

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Controls")
view = st.sidebar.radio("View", ["Live", "What-If", "Session", "Export"], index=0, key="view")
compound = st.sidebar.selectbox("Compound", ["soft", "medium", "hard"], index=1, key="compound")
dt = st.sidebar.slider("Tick interval (s)", 0.1, 1.0, 0.2, 0.1, key="dt")
run_toggle = st.sidebar.toggle("Run (auto-update only in Live)", value=True, key="run")
freeze = st.sidebar.toggle("Freeze updates (view details)", value=False, key="freeze")
reset = st.sidebar.button("Reset session", use_container_width=True)
step_once = st.sidebar.button("Step once", use_container_width=True)
pit_now = st.sidebar.button("Pit stop (switch compound)", use_container_width=True)
st.sidebar.markdown("---")
wake = st.sidebar.slider("Wake (downforce loss %)", 0, 40, 0, 5)
wet = st.sidebar.slider("Wet track %", 0, 100, 0, 10)

# Effective Run (do not mutate the widget state); pause when frozen
run = bool(st.session_state.get("run", False)) and (view == "Live") and not st.session_state.get("freeze", False)

# Reset session if requested
if reset:
    init_session()

# Keep compound_active in sync with dropdown (only when user changes it)
if "_compound_widget_last" not in st.session_state:
    st.session_state._compound_widget_last = st.session_state.get("compound", "medium")
if "compound_active" not in st.session_state:
    st.session_state.compound_active = st.session_state.get("compound", "medium")

compound_widget = st.session_state.get("compound", "medium")
if compound_widget != st.session_state._compound_widget_last:
    st.session_state.compound_active = compound_widget
    st.session_state._compound_widget_last = compound_widget

compound_active = st.session_state.compound_active

# -------------------------------------------------------------------
# Modelpack expander (presets + upload + download)
# -------------------------------------------------------------------
with st.sidebar.expander("Modelpack"):
    mp_dir = (Path(__file__).resolve().parent.parent / "modelpacks")
    mp_dir.mkdir(parents=True, exist_ok=True)

    presets = sorted([p.name for p in mp_dir.glob("*.yml")] + [p.name for p in mp_dir.glob("*.yaml")])
    default_idx = presets.index("monza-medium.yaml") if "monza-medium.yaml" in presets else (0 if presets else None)
    sel = st.selectbox("Select preset", presets if presets else ["<none>"], index=default_idx if default_idx is not None else 0, disabled=len(presets)==0, key="preset_select")

    if len(presets) and st.button("Apply selected preset", use_container_width=True, key="apply_preset_btn"):
        try:
            mp_path = mp_dir / sel
            mp = ModelPack.from_yaml(mp_path.read_text(encoding="utf-8"))
            # apply without touching the 'compound' widget key
            st.session_state.compound_active = mp.compound
            st.session_state._compound_widget_last = st.session_state.get("compound", "medium")  # prevent immediate resync
            compound_active = mp.compound
            st.session_state.sim.ambient = mp.ambient_c
            st.session_state.sim.track = mp.track_c
            st.session_state.model.p = ThermalParams(mp.a1, mp.a2, mp.a3, mp.a4, mp.a5, mp.b1, mp.b2, mp.c1, mp.c2)
            st.success(f"Modelpack applied: {mp.name}")
        except Exception as e:
            st.error(f"Error loading preset: {e}")

    uploaded = st.file_uploader("Upload custom YAML", type=["yml","yaml"], key="mp_upload")
    if uploaded is not None:
        try:
            mp = ModelPack.from_yaml(uploaded.read().decode("utf-8"))
            st.session_state.compound_active = mp.compound
            st.session_state._compound_widget_last = st.session_state.get("compound", "medium")  # prevent immediate resync
            compound_active = mp.compound
            st.session_state.sim.ambient = mp.ambient_c
            st.session_state.sim.track = mp.track_c
            st.session_state.model.p = ThermalParams(mp.a1, mp.a2, mp.a3, mp.a4, mp.a5, mp.b1, mp.b2, mp.c1, mp.c2)
            st.success(f"Modelpack applied: {mp.name}")
        except Exception as e:
            st.error(f"YAML parse error: {e}")

    st.markdown("---")
    # Download current snapshot
    p = st.session_state.model.p
    snap = ModelPack(
        name="snapshot",
        compound=st.session_state.compound_active,
        ambient_c=st.session_state.sim.ambient,
        track_c=st.session_state.sim.track,
        a1=p.a1,a2=p.a2,a3=p.a3,a4=p.a4,a5=p.a5,b1=p.b1,b2=p.b2,c1=p.c1,c2=p.c2,
    )
    default_name = (sel.split(".")[0] + "-snapshot.yaml") if len(presets) else "modelpack_snapshot.yaml"
    st.download_button("Download current modelpack", data=snap.to_yaml().encode("utf-8"),
                       file_name=default_name, mime="text/yaml", key="dl_mp_btn")

# -------------------------------------------------------------------
# Engine uses active compound
# -------------------------------------------------------------------
engine = DecisionEngine(compound_active)

# -------------------------------------------------------------------
# Pit stop (cycle compound_active only)
# -------------------------------------------------------------------
def apply_pit_stop():
    order = ["soft", "medium", "hard"]
    current = st.session_state.compound_active
    nxt = order[(order.index(current) + 1) % len(order)]
    st.session_state.compound_active = nxt
    st.session_state._compound_widget_last = st.session_state.get("compound", "medium")
    st.session_state.events.append((st.session_state.tick, "PIT", f"Compound -> {nxt}"))
    return DecisionEngine(nxt)

if pit_now:
    engine = apply_pit_stop()
    compound_active = st.session_state.compound_active

# -------------------------------------------------------------------
# One simulation tick with wake/wet & event detection
# -------------------------------------------------------------------
def tick(dt: float):
    u_common, loads, sensors = st.session_state.sim.step(dt)

    # Wake & Wet effects
    wake_k_cooling = 1.0 - 0.01*wake * 0.6   # reduces convective cooling
    wake_k_downforce = 1.0 - 0.01*wake * 0.5 # reduces normal load
    wet_k_cooling = 1.0 + 0.01*wet*0.8       # more cooling when wet
    wet_k_heating = 1.0 - 0.01*wet*0.9       # less frictional heating

    u_common["slip"] *= wet_k_heating
    u_common["slip_ang"] *= wet_k_heating
    u_common["cooling_factor"] = max(0.2, wake_k_cooling * wet_k_cooling)

    # Event detection
    if detect_lockup(st.session_state.last_slip, u_common["slip"], u_common["brake"]):
        st.session_state.events.append((st.session_state.tick, "LOCKUP", "Possible front lockup"))
    if detect_slide(st.session_state.last_sa, u_common["slip_ang"], u_common.get("latg", 3.0)):
        st.session_state.events.append((st.session_state.tick, "SLIDE", "Corner entry/exit slide"))

    st.session_state.last_slip = u_common["slip"]
    st.session_state.last_sa = u_common["slip_ang"]

    est_now = {}
    for corner in CORNER_KEYS:
        u = dict(**u_common, load=loads[corner]*wake_k_downforce)
        z = np.array([sensors[corner]["tpms"], sensors[corner]["hub"]])
        ekf = st.session_state.ekf[corner]
        x_est = ekf.step(z=z, u=u, dt=dt)
        est_now[corner] = x_est
        st.session_state.hist[corner].append(x_est.copy())
    st.session_state.tick += 1
    return est_now

# Drive sim if needed
est_now = {}
if (view == "Live" and run) or step_once:
    est_now = tick(st.session_state.dt)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def last_hist(n=240):
    out = {}
    for c in CORNER_KEYS:
        h = np.array(st.session_state.hist[c][-n:])
        out[c] = h if len(h) else np.zeros((0,3))
    return out

# -------------------------------------------------------------------
# Views
# -------------------------------------------------------------------
if view == "Live":
    # Wider left column for charts; right column uses tabs to avoid scrolling
    left, right = st.columns([3, 2])

    chart_tread = left.empty()
    chart_carcass = left.empty()
    tab_status, tab_recs, tab_metrics = right.tabs(["Status", "Recommendations", "Metrics"])

    lo, hi = engine.band
    H = last_hist(240)

    # Tread chart (shorter height to fit above the fold)
    fig_tread = go.Figure()
    for c in CORNER_KEYS:
        h = H[c]
        if len(h):
            fig_tread.add_trace(go.Scatter(y=h[:,0], mode="lines", name=f"{c} Tread"))
    band_label = f"{compound_active.title()} band"
    fig_tread.add_hrect(y0=lo, y1=hi, fillcolor="LightGreen", opacity=0.2, line_width=0)
    fig_tread.update_layout(height=260, title=f"Tread Temperature (last ~240 ticks) • {band_label}", xaxis_title="Tick", yaxis_title="°C")
    chart_tread.plotly_chart(fig_tread, use_container_width=True)

    # Carcass chart
    fig_carc = go.Figure()
    for c in CORNER_KEYS:
        h = H[c]
        if len(h):
            fig_carc.add_trace(go.Scatter(y=h[:,1], mode="lines", name=f"{c} Carcass"))
    fig_carc.update_layout(height=220, title="Carcass Temperature (last ~240 ticks)", xaxis_title="Tick", yaxis_title="°C")
    chart_carcass.plotly_chart(fig_carc, use_container_width=True)

    # Track status (tab)
    with tab_status:
        corner_xy = {"FL": (0.2, 0.8), "FR": (0.8, 0.8), "RL": (0.2, 0.2), "RR": (0.8, 0.2)}
        fig_track = go.Figure()
        xs, ys, texts, colors = [], [], [], []
        for c in CORNER_KEYS:
            h = H[c]
            if len(h):
                t = h[-1, 0]
                status = "HOT" if t > hi else ("COLD" if t < lo else "OK")
            else:
                status = "OK"
            color = {"HOT":"red", "COLD":"blue", "OK":"green"}[status]
            xs.append(corner_xy[c][0]); ys.append(corner_xy[c][1])
            texts.append(f"{c}: {status}")
            colors.append(color)
        fig_track.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
                                       marker=dict(size=18, color=colors)))
        fig_track.update_xaxes(visible=False); fig_track.update_yaxes(visible=False)
        fig_track.update_layout(height=240, title="Corner Status (demo track)", showlegend=False)
        st.plotly_chart(fig_track, use_container_width=True)

    # Recommendations (tab)
    actions = engine.actions(est_now) if est_now else []
    with tab_recs:
        st.subheader("Recommendations")
        if actions:
            for c, msg in actions[:12]:
                st.write(f"**{c}** — {msg}")
        else:
            st.write("_No recommendations yet. Start the simulation or step once._")

    # Metrics table (tab)
    with tab_metrics:
        def temp_stats(h):
            if not len(h): return (np.nan, np.nan, 0, 0)
            t = h[:,0]
            return (float(np.nanmax(t)), float(np.nanmean(t)), int(np.sum(t>hi)), int(np.sum(t<lo)))
        rows = []
        for c in CORNER_KEYS:
            mx, av, over, under = temp_stats(H[c]); rows.append((c, mx, av, over, under))
        cols = ["Corner","Max T(°C)","Avg T(°C)","Ticks > band","Ticks < band"]
        df = pd.DataFrame([(c, round(mx,1), round(av,1), over, under) for (c, mx, av, over, under) in rows], columns=cols)
        st.dataframe(df.set_index(cols[0]), use_container_width=True, height=240)

elif view == "What-If":
    st.subheader("What-If Simulator (1 lap forward)")
    wcol = st.columns([1,1,1,1])
    with wcol[0]:
        bias = st.slider("Brake bias (± clicks)", -3, 3, 0, 1)
    with wcol[1]:
        diff_entry = st.slider("Diff Entry (±)", -3, 3, 0, 1)
    with wcol[2]:
        push = st.slider("Aggressiveness (0=lift&coast, 10=push)", 0, 10, 5, 1)
    with wcol[3]:
        horizon_ticks = st.slider("Ticks to simulate", 60, 360, 180, 30)

    def control_to_mods(bias:int, diff:int, push:int):
        bias_k = 1.0 + 0.03*bias
        diff_k = 1.0 + 0.02*diff
        push_k = 0.8 + 0.04*push
        return bias_k, diff_k, push_k

    def simulate_forward(horizon:int):
        if not any(len(st.session_state.hist[c]) for c in CORNER_KEYS):
            return {c: np.zeros((0,3)) for c in CORNER_KEYS}
        bias_k, diff_k, push_k = control_to_mods(bias, diff_entry, push)
        model = st.session_state.model
        sim = st.session_state.sim
        dt_sim = st.session_state.dt

        # seed with last estimated state
        est0 = {c: (st.session_state.hist[c][-1].copy() if len(st.session_state.hist[c]) else np.array([95.,90.,85.])) for c in CORNER_KEYS}
        traj = {c: [est0[c].copy()] for c in CORNER_KEYS}

        t0 = sim.t  # current live sim time
        for k in range(1, horizon+1):
            u_common, loads = sim.snapshot_controls(t0 + k*dt_sim)
            for c in CORNER_KEYS:
                u = dict(**u_common, load=loads[c])
                u["brake"] = u["brake"]*bias_k
                u["slip"] = max(0.0, u["slip"]*diff_k)
                u["speed_kmh"] = u["speed_kmh"]*push_k
                u["cooling_factor"] = u.get("cooling_factor", 1.0)
                x = model.step(traj[c][-1], u, dt_sim)
                traj[c].append(x)
        return {c: np.vstack(traj[c]) for c in CORNER_KEYS}

    # Pressure planner (carcass temps across corners)
    N = 120
    carc_all = []
    for c in CORNER_KEYS:
        h = np.array(st.session_state.hist[c][-N:])
        if len(h): carc_all.append(h[:,1])
    carc_all = np.concatenate(carc_all) if carc_all else np.array([])
    dpsi = suggest_pressure_delta(carcass_t=carc_all, target_band=engine.band)
    st.info(f"Box pressure suggestion: **{dpsi:+.1f} psi** (toward band center)")

    if st.button("Run What-If", use_container_width=True):
        W = simulate_forward(horizon_ticks)
        lo, hi = engine.band
        colw1, colw2 = st.columns(2)
        with colw1:
            figw_t = go.Figure()
            for c in CORNER_KEYS:
                h = np.array(st.session_state.hist[c][-120:])
                if len(h): figw_t.add_trace(go.Scatter(y=h[:,0], mode="lines", name=f"{c} Tread (hist)", line=dict(dash="dot")))
                if len(W[c]): figw_t.add_trace(go.Scatter(y=W[c][:,0], mode="lines", name=f"{c} Tread (what-if)"))
            figw_t.add_hrect(y0=lo, y1=hi, fillcolor="LightGreen", opacity=0.2, line_width=0)
            figw_t.update_layout(height=360, title="Tread: history vs what-if", xaxis_title="Tick", yaxis_title="°C")
            st.plotly_chart(figw_t, use_container_width=True)
        with colw2:
            figw_c = go.Figure()
            for c in CORNER_KEYS:
                h = np.array(st.session_state.hist[c][-120:])
                if len(h): figw_c.add_trace(go.Scatter(y=h[:,1], mode="lines", name=f"{c} Carcass (hist)", line=dict(dash="dot")))
                if len(W[c]): figw_c.add_trace(go.Scatter(y=W[c][:,1], mode="lines", name=f"{c} Carcass (what-if)"))
            figw_c.update_layout(height=360, title="Carcass: history vs what-if", xaxis_title="Tick", yaxis_title="°C")
            st.plotly_chart(figw_c, use_container_width=True)

elif view == "Session":
    st.subheader("Events & Logs")
    if len(st.session_state.events) == 0:
        st.write("_No events yet._")
    else:
        for (ts, kind, details) in reversed(st.session_state.events[-50:]):
            st.write(f"• **{kind}** @ tick {ts}: {details}")

elif view == "Export":
    st.subheader("Export data")
    export_ticks = st.slider("Last N ticks", 120, 3000, 600, 60)
    def make_csv(n:int)->bytes:
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["tick","corner","tread_c","carcass_c","rim_c"])
        for c in CORNER_KEYS:
            hist = st.session_state.hist[c][-(n):]
            base = max(0, st.session_state.tick - len(hist))
            for i, x in enumerate(hist):
                w.writerow([base+i, c, float(x[0]), float(x[1]), float(x[2])])
        return out.getvalue().encode("utf-8")
    csv_bytes = make_csv(export_ticks)
    st.download_button("Download CSV", data=csv_bytes, file_name="tire_temps.csv", mime="text/csv")

# -------------------------------------------------------------------
# Auto-rerun only in Live and when Run is enabled (and not frozen)
# -------------------------------------------------------------------
if view == "Live" and run and not step_once:
    time.sleep(st.session_state.dt)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
