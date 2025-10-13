from __future__ import annotations
import time
import io
import csv
from pathlib import Path
from typing import Dict
from datetime import datetime
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
from wear import TireWearModel, WearParams
from weather import WeatherModel, WeatherParams, SessionType
from driver_profiles import DriverProfiles

# Advanced Features Integration
from simulation_engine import RaceSimulation, SimulationParams
from strategy_optimization import StrategyOptimizer, StrategyOptimizationParams
from big_data import BigDataAnalytics
from predictive_analytics import PredictiveAnalytics
from advanced_visualization import AdvancedVisualization
from data_insights import DataDrivenInsights
from real_time_collaboration import RealTimeCollaboration
from advanced_reporting import ReportGenerator

# -------------------------------------------------------------------
# Page config MUST be first Streamlit command
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🏎️ F1 Tire Management System", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏎️"
)

# Professional F1 Dashboard Styling
st.markdown("""
<style>
    /* F1 Theme Colors */
    :root {
        --f1-red: #e10600;
        --f1-black: #15151e;
        --f1-silver: #c0c0c0;
        --f1-gold: #ffd700;
        --f1-blue: #0066cc;
        --f1-green: #00ff00;
        --f1-orange: #ff6600;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, var(--f1-red) 0%, var(--f1-black) 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: var(--f1-silver);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid var(--f1-silver);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Status indicators */
    .status-good {
        color: var(--f1-green);
        font-weight: bold;
    }
    
    .status-warning {
        color: var(--f1-orange);
        font-weight: bold;
    }
    
    .status-critical {
        color: var(--f1-red);
        font-weight: bold;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #2d2d44 0%, #1e1e2e 100%);
        color: var(--f1-silver);
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--f1-silver);
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--f1-red) 0%, #8b0000 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--f1-red) 0%, #8b0000 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a1a 0%, var(--f1-red) 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--f1-green) 0%, #00cc00 100%);
        color: white;
        border-radius: 6px;
        padding: 1rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, var(--f1-red) 0%, #cc0000 100%);
        color: white;
        border-radius: 6px;
        padding: 1rem;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--f1-blue) 0%, #0066cc 100%);
        color: white;
        border-radius: 6px;
        padding: 1rem;
        border: none;
    }
    
    /* Chart containers */
    .chart-container {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid var(--f1-silver);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Data table styling */
    .stDataFrame {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid var(--f1-silver);
        border-radius: 8px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--f1-red) 0%, var(--f1-orange) 100%);
    }
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="main-header">
    <h1>🏎️ F1 TIRE MANAGEMENT SYSTEM</h1>
    <p>Professional Grade Tire Temperature & Strategy Optimization Platform</p>
</div>
""", unsafe_allow_html=True)

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
    wear_model = TireWearModel(WearParams())
    weather_model = WeatherModel(WeatherParams())
    driver_profiles = DriverProfiles()
    model = ThermalModel(params, wear_model, weather_model)
    sim = TelemetrySim(wear_model=wear_model, weather_model=weather_model)
    st.session_state.model = model
    st.session_state.sim = sim
    st.session_state.wear_model = wear_model
    st.session_state.weather_model = weather_model
    st.session_state.driver_profiles = driver_profiles
    st.session_state.active_driver = driver_profiles.active_driver
    st.session_state.ekf = {k: CornerEKF(lambda x,u,dt: model.step(x,u,dt,k,sim.current_compound)) for k in CORNER_KEYS}
    st.session_state.hist = {k: [] for k in CORNER_KEYS}   # list of np.array([Tt,Tc,Tr])
    st.session_state.events = []                           # (tick, type, details)
    st.session_state.tick = 0
    st.session_state.last_slip = 0.05
    st.session_state.last_sa = 2.0
    # track active compound independent from widget
    st.session_state.compound_active = st.session_state.get("compound", "medium")
    
    # Initialize advanced features
    st.session_state.big_data_analytics = BigDataAnalytics()
    st.session_state.predictive_analytics = PredictiveAnalytics()
    st.session_state.advanced_visualization = AdvancedVisualization()
    st.session_state.data_insights = DataDrivenInsights()
    st.session_state.report_generator = ReportGenerator()
    
    # Initialize simulation and strategy components
    st.session_state.race_simulation = RaceSimulation()
    st.session_state.strategy_optimizer = StrategyOptimizer(StrategyOptimizationParams())

def ensure_session():
    if "model" not in st.session_state or "sim" not in st.session_state or "ekf" not in st.session_state:
        init_session()

ensure_session()

# -------------------------------------------------------------------
# Professional Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, var(--f1-red) 0%, var(--f1-black) 100%); 
            padding: 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center;">
    <h2 style="color: white; margin: 0; font-size: 1.5rem;">🎛️ CONTROL CENTER</h2>
</div>
""", unsafe_allow_html=True)

# Main Controls Section
st.sidebar.markdown("### 🎮 Main Controls")
view = st.sidebar.radio("📊 Dashboard View", ["Live", "What-If", "Session", "Export"], index=0, key="view")
compound = st.sidebar.selectbox("🏎️ Tire Compound", ["soft", "medium", "hard"], index=1, key="compound")
dt = st.sidebar.slider("⏱️ Update Interval (s)", 0.1, 1.0, 0.2, 0.1, key="dt")

# Control Buttons with Professional Styling
st.sidebar.markdown("### 🎯 Session Control")
run_toggle = st.sidebar.toggle("▶️ Auto-Run", value=True, key="run")
freeze = st.sidebar.toggle("⏸️ Freeze Updates", value=False, key="freeze")

# Action Buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    reset = st.button("🔄 Reset", width='stretch', help="Reset entire session")
    step_once = st.button("⏭️ Step", width='stretch', help="Single simulation step")
with col2:
    pit_now = st.button("🏁 Pit Stop", width='stretch', help="Switch tire compound")

st.sidebar.markdown("---")

# Track Conditions
st.sidebar.markdown("### 🏁 Track Conditions")
wake = st.sidebar.slider("🌪️ Wake Effect (%)", 0, 40, 0, 5, help="Downforce loss from following")
wet = st.sidebar.slider("💧 Wet Track (%)", 0, 100, 0, 10, help="Track wetness level")

# Weather & Environment Section
st.sidebar.markdown("### 🌤️ Weather & Environment")
session_type = st.sidebar.selectbox("🏁 Session Type", ["fp1", "fp2", "fp3", "qualifying", "race"], index=0, key="session_type")
rain_probability = st.sidebar.slider("🌧️ Rain Probability (%)", 0, 100, 0, 5, key="rain_prob", help="Chance of rain")
wind_speed = st.sidebar.slider("💨 Wind Speed (km/h)", 0, 50, 0, 5, key="wind_speed", help="Wind speed affecting cooling")
humidity = st.sidebar.slider("💧 Humidity (%)", 0, 100, 50, 5, key="humidity", help="Air humidity level")

# Update weather model with user controls
if hasattr(st.session_state, 'weather_model'):
    st.session_state.weather_model.params.rain_probability = rain_probability / 100.0
    st.session_state.weather_model.params.wind_speed = wind_speed
    st.session_state.weather_model.params.humidity = humidity / 100.0
    st.session_state.sim.set_session_type(session_type)

# Driver Selection Section
st.sidebar.markdown("### 👤 Driver Selection")
if hasattr(st.session_state, 'driver_profiles'):
    driver_names = st.session_state.driver_profiles.get_driver_names()
    current_driver_idx = driver_names.index(st.session_state.active_driver) if st.session_state.active_driver in driver_names else 0
    selected_driver = st.sidebar.selectbox("🏎️ Active Driver", driver_names, index=current_driver_idx, key="driver_select")
    
    # Update active driver
    if selected_driver != st.session_state.active_driver:
        st.session_state.active_driver = selected_driver
        st.session_state.driver_profiles.set_active_driver(selected_driver)
    
    # Show driver info with professional styling
    driver = st.session_state.driver_profiles.get_driver(selected_driver)
    if driver:
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, #2d2d44 0%, #1e1e2e 100%); 
                    padding: 1rem; border-radius: 8px; border: 1px solid var(--f1-silver);">
            <h4 style="color: var(--f1-silver); margin: 0 0 0.5rem 0;">Driver Profile</h4>
            <p style="color: white; margin: 0.25rem 0;"><strong>Style:</strong> {driver.style.value.title()}</p>
            <p style="color: white; margin: 0.25rem 0;"><strong>Experience:</strong> {driver.experience.value.title()}</p>
            <p style="color: white; margin: 0.25rem 0;"><strong>Thermal Aggression:</strong> {driver.params.thermal_aggression:.2f}</p>
            <p style="color: white; margin: 0.25rem 0;"><strong>Tire Awareness:</strong> {driver.params.tire_awareness:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

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

    if len(presets) and st.button("Apply selected preset", width='stretch', key="apply_preset_btn"):
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
    st.download_button(
        "📥 Download Modelpack", 
        data=snap.to_yaml().encode("utf-8"),
        file_name=f"f1_modelpack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml", 
        mime="text/yaml", 
        key="dl_mp_btn"
    )

# -------------------------------------------------------------------
# Engine uses active compound
# -------------------------------------------------------------------
engine = DecisionEngine(compound_active, st.session_state.get('wear_model'), 
                       st.session_state.get('driver_profiles'), st.session_state.get('active_driver'))

# -------------------------------------------------------------------
# Pit stop (cycle compound_active only)
# -------------------------------------------------------------------
def apply_pit_stop():
    order = ["soft", "medium", "hard"]
    current = st.session_state.compound_active
    nxt = order[(order.index(current) + 1) % len(order)]
    st.session_state.compound_active = nxt
    st.session_state._compound_widget_last = st.session_state.get("compound", "medium")
    
    # Reset wear levels for new tires
    if hasattr(st.session_state, 'wear_model'):
        st.session_state.wear_model.reset_wear()
    
    # Update simulator compound and reset wear
    st.session_state.sim.pit_stop(nxt)
    
    st.session_state.events.append((st.session_state.tick, "PIT", f"Compound -> {nxt}, Wear reset"))
    return DecisionEngine(nxt, st.session_state.wear_model)

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
    # Professional Dashboard Layout
    st.markdown("### 📊 LIVE DASHBOARD")
    
    # Status Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏎️ Lap", f"{st.session_state.sim.lap_count}", "Active")
    with col2:
        st.metric("🌡️ Track Temp", f"{st.session_state.sim.track:.1f}°C", f"{st.session_state.sim.ambient:.1f}°C")
    with col3:
        st.metric("🏁 Session", session_type.title(), "Running")
    with col4:
        st.metric("👤 Driver", st.session_state.active_driver, "Active")
    
    st.markdown("---")
    
    # Main Dashboard Layout
    left, right = st.columns([3, 2])

    chart_tread = left.empty()
    chart_carcass = left.empty()
    tab_status, tab_recs, tab_metrics, tab_wear, tab_weather, tab_drivers, tab_advanced = right.tabs([
        "📊 Status", "💡 Recommendations", "📈 Metrics", "🛞 Wear", "🌤️ Weather", "👤 Drivers", "🚀 Advanced"
    ])

    lo, hi = engine.band
    H = last_hist(240)

    # Professional Tread Temperature Chart
    fig_tread = go.Figure()
    colors = ['#e10600', '#0066cc', '#00ff00', '#ff6600']  # F1 team colors
    
    for i, c in enumerate(CORNER_KEYS):
        h = H[c]
        if len(h):
            fig_tread.add_trace(go.Scatter(
                y=h[:,0], 
                mode="lines+markers", 
                name=f"{c} Tread",
                line=dict(color=colors[i], width=3),
                marker=dict(size=4)
            ))
    
    band_label = f"{compound_active.title()} Optimal Band"
    fig_tread.add_hrect(y0=lo, y1=hi, fillcolor="rgba(0,255,0,0.1)", opacity=0.3, line_width=0, 
                       annotation_text=band_label, annotation_position="top right")
    
    fig_tread.update_layout(
        height=300,
        title=dict(
            text=f"🌡️ TREAD TEMPERATURE MONITORING<br><sub>{band_label}</sub>",
            font=dict(size=16, color="white"),
            x=0.5
        ),
        xaxis=dict(title="Time (ticks)", color="white", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Temperature (°C)", color="white", gridcolor="rgba(255,255,255,0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
    )
    chart_tread.plotly_chart(fig_tread, config={"displayModeBar": True, "displaylogo": False})

    # Professional Carcass Temperature Chart
    fig_carc = go.Figure()
    for i, c in enumerate(CORNER_KEYS):
        h = H[c]
        if len(h):
            fig_carc.add_trace(go.Scatter(
                y=h[:,1], 
                mode="lines+markers", 
                name=f"{c} Carcass",
                line=dict(color=colors[i], width=3),
                marker=dict(size=4)
            ))
    
    fig_carc.update_layout(
        height=280,
        title=dict(
            text="🛞 CARCASS TEMPERATURE MONITORING",
            font=dict(size=16, color="white"),
            x=0.5
        ),
        xaxis=dict(title="Time (ticks)", color="white", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Temperature (°C)", color="white", gridcolor="rgba(255,255,255,0.1)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
    )
    chart_carcass.plotly_chart(fig_carc, config={"displayModeBar": True, "displaylogo": False})

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
        st.plotly_chart(fig_track, config={"displayModeBar": True, "displaylogo": False})

    # Recommendations (tab)
    wear_summary = st.session_state.wear_model.get_wear_summary() if hasattr(st.session_state, 'wear_model') else None
    weather_summary = st.session_state.weather_model.get_weather_summary() if hasattr(st.session_state, 'weather_model') else None
    actions = engine.actions(est_now, wear_summary) if est_now else []
    
    # Add weather recommendations
    if weather_summary:
        weather_recs = st.session_state.weather_model.get_weather_recommendations()
        actions.extend([("WEATHER", rec) for _, rec in weather_recs])
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
        st.dataframe(df.set_index(cols[0]), width='stretch', height=240)

    # Wear analysis (tab)
    with tab_wear:
        st.subheader("Tire Wear Analysis")
        
        if hasattr(st.session_state, 'wear_model'):
            wear_summary = st.session_state.wear_model.get_wear_summary()
            
            # Wear level chart
            corners = list(wear_summary.keys())
            wear_levels = [wear_summary[c]['wear_level'] for c in corners]
            grip_factors = [wear_summary[c]['grip_factor'] for c in corners]
            
            # Create wear visualization
            fig_wear = go.Figure()
            
            # Wear level bars
            fig_wear.add_trace(go.Bar(
                x=corners,
                y=wear_levels,
                name='Wear Level',
                marker_color=['red' if w > 0.6 else 'orange' if w > 0.3 else 'green' for w in wear_levels],
                text=[f"{w:.1%}" for w in wear_levels],
                textposition='auto'
            ))
            
            fig_wear.update_layout(
                title="Tire Wear Levels",
                yaxis_title="Wear Level (%)",
                height=200,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_wear, config={"displayModeBar": True, "displaylogo": False})
            
            # Grip degradation chart
            fig_grip = go.Figure()
            fig_grip.add_trace(go.Bar(
                x=corners,
                y=[gf * 100 for gf in grip_factors],
                name='Grip Factor',
                marker_color=['red' if gf < 0.8 else 'orange' if gf < 0.9 else 'green' for gf in grip_factors],
                text=[f"{gf:.0f}%" for gf in [gf * 100 for gf in grip_factors]],
                textposition='auto'
            ))
            
            fig_grip.update_layout(
                title="Grip Degradation",
                yaxis_title="Grip Factor (%)",
                height=200,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_grip, config={"displayModeBar": True, "displaylogo": False})
            
            # Wear recommendations
            wear_recs = engine.get_wear_recommendations(wear_summary)
            if wear_recs:
                st.subheader("Wear Recommendations")
                for corner, rec in wear_recs:
                    st.write(f"**{corner}** — {rec}")
            
            # Pit window prediction
            if st.session_state.sim.lap_count > 0:
                pit_predictions = engine.predict_pit_window(wear_summary, st.session_state.sim.lap_count, 50)
                st.subheader("Pit Window Predictions")
                for corner, pred in pit_predictions.items():
                    urgency_color = "🔴" if pred['urgency'] == "HIGH" else "🟡" if pred['urgency'] == "MEDIUM" else "🟢"
                    st.write(f"**{corner}** {urgency_color} Lap {pred['recommended_pit_lap']:.0f} (urgency: {pred['urgency']})")
        else:
            st.write("_Wear modeling not available. Reset session to enable._")

    # Weather analysis (tab)
    with tab_weather:
        st.subheader("Weather & Environment")
        
        if hasattr(st.session_state, 'weather_model'):
            weather_summary = st.session_state.weather_model.get_weather_summary()
            
            # Weather condition indicator
            condition = weather_summary['current_condition']
            condition_colors = {
                'dry': '🟢',
                'damp': '🟡', 
                'wet': '🟠',
                'heavy_rain': '🔴'
            }
            condition_emoji = condition_colors.get(condition, '⚪')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Condition", f"{condition_emoji} {condition.title()}")
            with col2:
                st.metric("Rain Probability", f"{weather_summary['rain_probability']:.1%}")
            with col3:
                st.metric("Session Progress", f"{weather_summary['session_progress']:.1%}")
            
            # Temperature charts
            fig_temp = go.Figure()
            
            # Track temperature
            fig_temp.add_trace(go.Scatter(
                y=st.session_state.weather_model.track_temp_history[-100:],
                mode='lines',
                name='Track Temperature',
                line=dict(color='red', width=2)
            ))
            
            # Ambient temperature
            fig_temp.add_trace(go.Scatter(
                y=st.session_state.weather_model.ambient_temp_history[-100:],
                mode='lines',
                name='Ambient Temperature',
                line=dict(color='blue', width=2)
            ))
            
            fig_temp.update_layout(
                title="Temperature Evolution",
                yaxis_title="Temperature (°C)",
                height=200,
                xaxis_title="Time Steps"
            )
            st.plotly_chart(fig_temp, config={"displayModeBar": True, "displaylogo": False})
            
            # Environmental factors
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Track Temperature", f"{weather_summary['track_temperature']:.1f}°C")
                st.metric("Ambient Temperature", f"{weather_summary['ambient_temperature']:.1f}°C")
                st.metric("Rubber Buildup", f"{weather_summary['rubber_buildup']:.1%}")
            
            with col2:
                st.metric("Cooling Factor", f"{weather_summary['cooling_factor']:.2f}x")
                st.metric("Grip Factor", f"{weather_summary['grip_factor']:.2f}x")
                st.metric("Thermal Factor", f"{weather_summary['thermal_factor']:.2f}x")
            
            # Weather recommendations
            weather_recs = st.session_state.weather_model.get_weather_recommendations()
            if weather_recs:
                st.subheader("Weather Recommendations")
                for category, rec in weather_recs:
                    if category == "WEATHER":
                        st.warning(f"🌧️ {rec}")
                    elif category == "TRACK":
                        st.info(f"🏁 {rec}")
                    elif category == "WIND":
                        st.info(f"💨 {rec}")
                    else:
                        st.write(f"**{category}** — {rec}")
            
            # Weather history table
            if len(st.session_state.weather_model.weather_history) > 0:
                st.subheader("Weather History")
                weather_df = pd.DataFrame(st.session_state.weather_model.weather_history[-20:])
                weather_df['time'] = weather_df['time'].round(1)
                weather_df['rain_probability'] = weather_df['rain_probability'].apply(lambda x: f"{x:.1%}")
                weather_df['rain_intensity'] = weather_df['rain_intensity'].apply(lambda x: f"{x:.1%}")
                st.dataframe(weather_df, width='stretch', height=200)
        else:
            st.write("_Weather modeling not available. Reset session to enable._")

    # Driver analysis (tab)
    with tab_drivers:
        st.subheader("Driver Analysis & Comparison")
        
        if hasattr(st.session_state, 'driver_profiles'):
            driver_profiles = st.session_state.driver_profiles
            
            # Active driver summary
            active_driver = driver_profiles.get_active_driver()
            if active_driver:
                st.subheader(f"Active Driver: {active_driver.name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Style", active_driver.style.value.title())
                    st.metric("Experience", active_driver.experience.value.title())
                with col2:
                    st.metric("Thermal Aggression", f"{active_driver.params.thermal_aggression:.2f}")
                    st.metric("Tire Awareness", f"{active_driver.params.tire_awareness:.2f}")
                with col3:
                    st.metric("Wet Weather Skill", f"{active_driver.params.wet_weather_skill:.2f}")
                    st.metric("Pressure Management", f"{active_driver.params.pressure_management:.2f}")
                
                # Driver thermal signature
                thermal_sig = active_driver.thermal_signature
                st.subheader("Thermal Signature")
                
                fig_thermal = go.Figure()
                fig_thermal.add_trace(go.Bar(
                    x=list(thermal_sig.keys()),
                    y=list(thermal_sig.values()),
                    marker_color=['red', 'blue', 'green', 'orange', 'purple']
                ))
                fig_thermal.update_layout(
                    title="Driver Thermal Characteristics",
                    yaxis_title="Multiplier",
                    height=200
                )
                st.plotly_chart(fig_thermal, config={"displayModeBar": True, "displaylogo": False})
            
            # Driver comparison
            st.subheader("Driver Comparison")
            
            # Get comparison data
            comparison_data = driver_profiles.compare_drivers()
            
            if comparison_data:
                # Create comparison chart
                drivers = list(comparison_data.keys())
                metrics = ['thermal_consistency', 'recommendation_follow_rate']
                
                fig_comparison = go.Figure()
                for metric in metrics:
                    values = [comparison_data[driver].get(metric, 0.0) for driver in drivers]
                    fig_comparison.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=drivers,
                        y=values
                    ))
                
                fig_comparison.update_layout(
                    title="Driver Performance Comparison",
                    yaxis_title="Score",
                    height=200,
                    barmode='group'
                )
                st.plotly_chart(fig_comparison, config={"displayModeBar": True, "displaylogo": False})
                
                # Driver rankings
                st.subheader("Driver Rankings")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Thermal Consistency Ranking**")
                    thermal_rankings = driver_profiles.get_driver_rankings('thermal_consistency')
                    for i, (driver_name, score) in enumerate(thermal_rankings):
                        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                        st.write(f"{medal} {driver_name}: {score:.3f}")
                
                with col2:
                    st.write("**Recommendation Follow Rate**")
                    follow_rankings = driver_profiles.get_driver_rankings('recommendation_follow_rate')
                    for i, (driver_name, score) in enumerate(follow_rankings):
                        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                        st.write(f"{medal} {driver_name}: {score:.3f}")
            
            # Driver development insights
            if active_driver:
                st.subheader("Development Insights")
                insights = driver_profiles.get_driver_development_insights(active_driver.name)
                if insights:
                    for category, insight in insights:
                        if category == "DEVELOPMENT":
                            st.info(f"📈 {insight}")
                        elif category == "COACHING":
                            st.warning(f"🎯 {insight}")
                        else:
                            st.write(f"**{category}** — {insight}")
                else:
                    st.write("_No specific development insights available._")
            
            # Multi-driver simulation
            st.subheader("Multi-Driver Race Simulation")
            if st.button("Run Race Simulation", width='stretch'):
                with st.spinner("Simulating multi-driver race..."):
                    # Get current conditions
                    weather_summary = st.session_state.weather_model.get_weather_summary() if hasattr(st.session_state, 'weather_model') else {}
                    
                    # Run simulation
                    simulation_results = driver_profiles.simulate_multi_driver_race(weather_summary, laps=5)
                    
                    # Display results
                    st.success("Race simulation completed!")
                    
                    # Lap times comparison
                    fig_lap_times = go.Figure()
                    for driver_name, results in simulation_results.items():
                        fig_lap_times.add_trace(go.Scatter(
                            x=list(range(1, len(results['lap_times']) + 1)),
                            y=results['lap_times'],
                            mode='lines+markers',
                            name=driver_name
                        ))
                    
                    fig_lap_times.update_layout(
                        title="Lap Times Comparison",
                        xaxis_title="Lap",
                        yaxis_title="Lap Time (seconds)",
                        height=200
                    )
                    st.plotly_chart(fig_lap_times, config={"displayModeBar": True, "displaylogo": False})
                    
                    # Average lap times
                    st.write("**Average Lap Times:**")
                    avg_times = {}
                    for driver_name, results in simulation_results.items():
                        avg_time = np.mean(results['lap_times'])
                        avg_times[driver_name] = avg_time
                    
                    sorted_times = sorted(avg_times.items(), key=lambda x: x[1])
                    for i, (driver_name, avg_time) in enumerate(sorted_times):
                        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                        st.write(f"{medal} {driver_name}: {avg_time:.2f}s")
        else:
            st.write("_Driver profiles not available. Reset session to enable._")

    # Advanced Features (tab)
    with tab_advanced:
        st.subheader("🚀 Advanced Features")
        
        # Big Data Analytics
        st.subheader("📊 Big Data Analytics")
        if hasattr(st.session_state, 'big_data_analytics'):
            analytics = st.session_state.big_data_analytics
            
            # Store current telemetry data
            if est_now:
                telemetry_data = {
                    'timestamp': datetime.now(),
                    'tread_temps': [est_now[c][0] for c in CORNER_KEYS],
                    'carcass_temps': [est_now[c][1] for c in CORNER_KEYS],
                    'rim_temps': [est_now[c][2] for c in CORNER_KEYS],
                    'compound': st.session_state.compound_active,
                    'lap_count': st.session_state.sim.lap_count
                }
                analytics.store_telemetry_data(telemetry_data)
            
            # Show analytics summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Points Stored", len(analytics.telemetry_cache))
                st.metric("Performance Records", len(analytics.analysis_cache))
            with col2:
                st.metric("Weather Records", len(analytics.weather_cache))
                st.metric("Driver Records", len(analytics.driver_cache))
            
            if st.button("Generate Analytics Report", width='stretch'):
                with st.spinner("Generating analytics report..."):
                    report = analytics.generate_performance_report()
                    st.session_state.analytics_report = report
                    st.success("Analytics report generated successfully!")
            
            # Display the report if it exists
            if hasattr(st.session_state, 'analytics_report'):
                st.subheader("📊 Analytics Report")
                st.json(st.session_state.analytics_report)
                
                # Add a button to clear the report
                if st.button("Clear Report", width='stretch'):
                    del st.session_state.analytics_report
                    st.rerun()
        
        # Predictive Analytics
        st.subheader("🔮 Predictive Analytics")
        if hasattr(st.session_state, 'predictive_analytics'):
            pred_analytics = st.session_state.predictive_analytics
            
            if st.button("Run Predictive Analysis", width='stretch'):
                with st.spinner("Running predictive analysis..."):
                    # Prepare thermal state (average across corners)
                    thermal_state = np.array([
                        np.mean([est_now[c][0] for c in CORNER_KEYS]) if est_now else 100.0,  # tread temp
                        np.mean([est_now[c][1] for c in CORNER_KEYS]) if est_now else 95.0,   # carcass temp
                        np.mean([est_now[c][2] for c in CORNER_KEYS]) if est_now else 90.0    # rim temp
                    ])
                    
                    # Prepare wear summary
                    wear_summary = st.session_state.wear_model.get_wear_summary() if hasattr(st.session_state, 'wear_model') else {}
                    
                    # Prepare weather summary
                    weather_summary = st.session_state.weather_model.get_weather_summary() if hasattr(st.session_state, 'weather_model') else {
                        'track_temperature': st.session_state.sim.track,
                        'ambient_temperature': st.session_state.sim.ambient,
                        'humidity': 50.0,
                        'wind_speed': 5.0
                    }
                    
                    # Prepare race context
                    race_context = {
                        'lap_number': st.session_state.sim.lap_count,
                        'compound': st.session_state.compound_active,
                        'session_type': 'race',
                        'track_position': 1
                    }
                    
                    prediction = pred_analytics.predict_lap_time(thermal_state, wear_summary, weather_summary, race_context)
                    st.session_state.prediction_result = prediction
                    st.success("Predictive analysis completed successfully!")
            
            # Display prediction results if they exist
            if hasattr(st.session_state, 'prediction_result'):
                st.subheader("🔮 Prediction Results")
                prediction = st.session_state.prediction_result
                st.success(f"Predicted next lap time: {prediction.get('prediction', 'N/A')}s")
                st.info(f"Confidence: {prediction.get('confidence', 0):.2%}")
                
                # Add a button to clear the prediction
                if st.button("Clear Prediction", width='stretch'):
                    del st.session_state.prediction_result
                    st.rerun()
        
        # Strategy Optimization
        st.subheader("🎯 Strategy Optimization")
        if hasattr(st.session_state, 'strategy_optimizer'):
            optimizer = st.session_state.strategy_optimizer
            
            if st.button("Optimize Race Strategy", width='stretch'):
                with st.spinner("Optimizing race strategy..."):
                    # Create race context
                    race_context = {
                        'current_lap': st.session_state.sim.lap_count,
                        'total_laps': 58,
                        'current_compound': st.session_state.compound_active,
                        'tire_age': st.session_state.sim.lap_count,
                        'track_conditions': 'dry',
                        'weather_forecast': 'stable'
                    }
                    
                    result = optimizer.optimize_strategy(race_context)
                    st.session_state.strategy_result = result
                    st.success("Race strategy optimization completed successfully!")
            
            # Display strategy results if they exist
            if hasattr(st.session_state, 'strategy_result'):
                st.subheader("🎯 Optimized Strategy Results")
                result = st.session_state.strategy_result
                st.success(f"Best strategy fitness: {result.get('best_fitness', 'N/A'):.2f}")
                
                if 'best_strategy' in result:
                    strategy = result['best_strategy']
                    st.write("**Optimized Strategy:**")
                    st.write(f"- Pit windows: {strategy.get('pit_windows', 'N/A')}")
                    st.write(f"- Tire pressure: {strategy.get('tire_pressure', 'N/A')} bar")
                    st.write(f"- Driving style: {strategy.get('driving_style', 'N/A')}")
                
                # Add a button to clear the strategy results
                if st.button("Clear Strategy Results", width='stretch'):
                    del st.session_state.strategy_result
                    st.rerun()
        
        # Data-Driven Insights
        st.subheader("💡 Data-Driven Insights")
        if hasattr(st.session_state, 'data_insights'):
            insights = st.session_state.data_insights
            
            if st.button("Generate Insights", width='stretch'):
                with st.spinner("Generating insights..."):
                    # Prepare performance data
                    performance_data = []
                    for corner in CORNER_KEYS:
                        if corner in st.session_state.hist and len(st.session_state.hist[corner]) > 0:
                            hist = st.session_state.hist[corner]
                            performance_data.append({
                                'corner': corner,
                                'tread_temp': hist[-1][0] if len(hist) > 0 else 100.0,
                                'carcass_temp': hist[-1][1] if len(hist) > 0 else 95.0,
                                'rim_temp': hist[-1][2] if len(hist) > 0 else 90.0,
                                'timestamp': datetime.now()
                            })
                    
                    if performance_data:
                        trend_insights = insights.generate_performance_trend_insights()
                        optimization_insights = insights.generate_optimization_insights(performance_data)
                        
                        st.session_state.insights_result = {
                            'trend_insights': trend_insights,
                            'optimization_insights': optimization_insights,
                            'performance_data': performance_data
                        }
                        st.success("Data-driven insights generated successfully!")
                    else:
                        st.warning("No performance data available for insights generation.")
            
            # Display insights results if they exist
            if hasattr(st.session_state, 'insights_result'):
                st.subheader("💡 Generated Insights")
                insights_result = st.session_state.insights_result
                
                st.write("**Trend Insights:**")
                trend_insights = insights_result.get('trend_insights', [])
                if trend_insights and len(trend_insights) > 0:
                    for insight in trend_insights[:3]:  # Show top 3
                        if hasattr(insight, 'title') and hasattr(insight, 'description'):
                            st.write(f"• {insight.title}: {insight.description}")
                        else:
                            st.write(f"• {insight}")
                else:
                    st.write("• No trend insights available yet")
                
                st.write("**Optimization Insights:**")
                optimization_insights = insights_result.get('optimization_insights', [])
                if optimization_insights and len(optimization_insights) > 0:
                    for insight in optimization_insights[:3]:  # Show top 3
                        if hasattr(insight, 'title') and hasattr(insight, 'description'):
                            st.write(f"• {insight.title}: {insight.description}")
                        else:
                            st.write(f"• {insight}")
                else:
                    st.write("• No optimization insights available yet")
                
                # Add a button to clear the insights
                if st.button("Clear Insights", width='stretch'):
                    del st.session_state.insights_result
                    st.rerun()
        
        # Advanced Reporting
        st.subheader("📋 Advanced Reporting")
        if hasattr(st.session_state, 'report_generator'):
            report_gen = st.session_state.report_generator
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Race Summary", width='stretch'):
                    with st.spinner("Generating race summary..."):
                        report_content = report_gen.generate_race_summary()
                        st.success("Race summary report generated!")
                        
                        # Create structured CSV data for race summary
                        csv_lines = [
                            "Metric,Value,Unit,Timestamp",
                            f"Report Type,Race Summary,,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            f"Generated By,F1 Tire Management System,,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            f"Report Format,CSV Export,,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        ]
                        
                        # Parse report content and add structured data
                        if hasattr(st.session_state, 'sim') and st.session_state.sim:
                            sim = st.session_state.sim
                            csv_lines.extend([
                                f"Current Lap,{sim.lap_count},laps,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Current Compound,{sim.current_compound},type,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Ambient Temperature,{sim.ambient:.1f},°C,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Track Temperature,{sim.track:.1f},°C,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            ])
                        
                        if hasattr(st.session_state, 'wear_model') and st.session_state.wear_model:
                            wear = st.session_state.wear_model
                            csv_lines.extend([
                                f"Front Left Wear,{wear.wear_levels['FL']:.3f},%,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Front Right Wear,{wear.wear_levels['FR']:.3f},%,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Rear Left Wear,{wear.wear_levels['RL']:.3f},%,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Rear Right Wear,{wear.wear_levels['RR']:.3f},%,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            ])
                        
                        csv_data = "\n".join(csv_lines)
                        
                        st.download_button(
                            label="📊 Download Race Summary CSV",
                            data=csv_data.encode('utf-8'),
                            file_name=f"f1_race_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if st.button("Generate Performance Analysis", width='stretch'):
                    with st.spinner("Generating performance analysis..."):
                        report_content = report_gen.generate_performance_analysis()
                        st.success("Performance analysis report generated!")
                        
                        # Create structured CSV data for performance analysis
                        csv_lines = [
                            "Metric,Value,Unit,Timestamp",
                            f"Report Type,Performance Analysis,,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            f"Generated By,F1 Tire Management System,,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            f"Report Format,CSV Export,,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        ]
                        
                        # Add performance metrics
                        if hasattr(st.session_state, 'sim') and st.session_state.sim:
                            sim = st.session_state.sim
                            csv_lines.extend([
                                f"Current Speed,{sim.speed:.1f},km/h,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Lap Distance,{sim.lap_dist:.0f},meters,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"Lap Time,{sim.lap_time_s:.2f},seconds,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            ])
                        
                        # Add thermal data if available
                        if hasattr(st.session_state, 'hist') and st.session_state.hist:
                            hist = st.session_state.hist
                            for corner in ['FL', 'FR', 'RL', 'RR']:
                                if corner in hist and len(hist[corner]) > 0:
                                    latest_temp = hist[corner][-1]
                                    csv_lines.extend([
                                        f"{corner} Tread Temp,{latest_temp[0]:.1f},°C,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                        f"{corner} Carcass Temp,{latest_temp[1]:.1f},°C,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                        f"{corner} Rim Temp,{latest_temp[2]:.1f},°C,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                    ])
                        
                        csv_data = "\n".join(csv_lines)
                        
                        st.download_button(
                            label="📊 Download Performance Analysis CSV",
                            data=csv_data.encode('utf-8'),
                            file_name=f"f1_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

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

    if st.button("Run What-If", width='stretch'):
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
            st.plotly_chart(figw_t, config={"displayModeBar": True, "displaylogo": False})
        with colw2:
            figw_c = go.Figure()
            for c in CORNER_KEYS:
                h = np.array(st.session_state.hist[c][-120:])
                if len(h): figw_c.add_trace(go.Scatter(y=h[:,1], mode="lines", name=f"{c} Carcass (hist)", line=dict(dash="dot")))
                if len(W[c]): figw_c.add_trace(go.Scatter(y=W[c][:,1], mode="lines", name=f"{c} Carcass (what-if)"))
            figw_c.update_layout(height=360, title="Carcass: history vs what-if", xaxis_title="Tick", yaxis_title="°C")
            st.plotly_chart(figw_c, config={"displayModeBar": True, "displaylogo": False})

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
    st.download_button(
        "📊 Download Tire Data CSV", 
        data=csv_bytes, 
        file_name=f"f1_tire_temperatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
        mime="text/csv"
    )

# -------------------------------------------------------------------
# Auto-rerun only in Live and when Run is enabled (and not frozen)
# -------------------------------------------------------------------
if view == "Live" and run and not step_once:
    time.sleep(st.session_state.dt)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
