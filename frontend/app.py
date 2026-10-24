import os
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import numpy as np
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "location_aware_gis_leakage_dataset.csv"

# =====================================================================
# BACKEND: models, env, Gemini client
# =====================================================================
# Load Models
models = {
    "Random Forest": joblib.load(ROOT_DIR / "random_forest_model.pkl"),
    "Decision Tree": joblib.load(ROOT_DIR / "decision_tree_model.pkl"),
    "Logistic Regression": joblib.load(ROOT_DIR / "logistic_regression_model.pkl"),
    "SVM": joblib.load(ROOT_DIR / "svm_model.pkl")
}

load_dotenv(ROOT_DIR / ".env")

gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.7-flash")

# Initialize Gemini client
client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

# Page Config
st.set_page_config(page_title="Water Leak Detection", page_icon="💧", layout="wide")

# How long (seconds) the loader stays visible on quick actions so it doesn't just flash.
LOADER_MIN_SECONDS = 0.7


# =====================================================================
# FRONTEND: theme, CSS, diagrams, loaders
# =====================================================================
STATIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');
@property --num { syntax:'<integer>'; initial-value:0; inherits:false; }

html, body, .stApp, [class*="css"] { font-family:'Manrope', sans-serif; }
h1,h2,h3,h4,.hero-title,.kpi-num,.sect,.brand,.res-title,.svg-title { font-family:'Fraunces', serif !important; letter-spacing:-0.01em; }

.stApp {
  background:
    radial-gradient(900px 420px at 90% -8%, var(--glow), transparent 62%),
    radial-gradient(700px 420px at -8% 108%, var(--glow), transparent 62%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  color:var(--text);
}
[data-testid="stHeader"] { background:transparent; }
[data-testid="stStatusWidget"] { display:none; }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stMarkdown, [data-testid="stWidgetLabel"] * { color:var(--text); }
.stCaption, small { color:var(--muted) !important; }

/* thin loading bar while the app is running */
.stApp[data-test-script-state="running"]::after {
  content:""; position:fixed; top:0; left:0; width:100%; height:3px; z-index:99999;
  background:linear-gradient(90deg, transparent, var(--aqua), transparent);
  background-size:40% 100%; background-repeat:no-repeat; animation:sweep 1.1s linear infinite;
}

/* Sidebar */
[data-testid="stSidebar"] { background:var(--surface-solid); border-right:1px solid var(--border); }
.brand { font-size:1.5rem; font-weight:600; color:var(--text); margin:4px 0 2px; display:flex; align-items:center; gap:10px; }
.brand svg { width:22px; height:22px; }
[data-testid="stSidebar"] .stRadio label { padding:8px 12px; border-radius:12px; transition:background .25s ease, transform .25s ease; width:100%; }
[data-testid="stSidebar"] .stRadio label:hover { background:var(--glow); transform:translateX(5px); }
[data-testid="stSidebar"] .stRadio label:has(input:checked) { background:var(--glow); box-shadow:inset 3px 0 0 var(--aqua); }

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"],
.stNumberInput input, .stTextInput input {
  background:var(--input-bg) !important; color:var(--text) !important;
  border-color:var(--border) !important; border-radius:12px !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
  box-shadow:0 0 0 3px var(--glow) !important; border-color:var(--aqua) !important; transition:box-shadow .25s ease;
}
.stNumberInput button { background:transparent !important; color:var(--text) !important; }
div[data-baseweb="popover"] ul { background:var(--surface-solid) !important; }
div[data-baseweb="popover"] li { color:var(--text) !important; }
.stSlider [role="slider"] { background:var(--aqua) !important; box-shadow:0 0 0 6px var(--glow); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background:var(--aqua); color:var(--on-accent) !important; font-weight:700; border:none; border-radius:12px;
  padding:.55rem 1.4rem; position:relative; overflow:hidden;
  transition:transform .2s ease, box-shadow .25s ease; box-shadow:0 4px 14px var(--glow);
}
.stButton > button p, .stDownloadButton > button p { color:var(--on-accent) !important; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 22px var(--glow); }
.stButton > button:active { transform:scale(.97); }
.stButton > button::after {
  content:""; position:absolute; inset:0;
  background:linear-gradient(110deg, transparent 30%, rgba(255,255,255,.4) 50%, transparent 70%);
  transform:translateX(-120%); transition:transform .6s ease;
}
.stButton > button:hover::after { transform:translateX(120%); }

/* Frames */
[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border); border-radius:16px; }
[data-testid="stPlotlyChart"] {
  background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:6px;
  box-shadow:var(--shadow); animation:rise .7s ease both;
}
iframe { border-radius:16px; }
[data-testid="stChatMessage"] { background:var(--surface); border:1px solid var(--border); border-radius:16px; animation:rise .5s ease both; }

/* Headings */
.sect { font-size:1.3rem; font-weight:600; margin:14px 0 8px; color:var(--text) !important; }
.sect::after { content:""; display:block; height:2px; width:46px; margin-top:6px; background:linear-gradient(90deg, var(--brass), transparent); animation:grow .9s ease both; }

/* Hero */
.hero { position:relative; overflow:hidden; border-radius:22px; padding:34px 40px 66px;
  background:linear-gradient(120deg, #07202f, #0c3a54 70%, #0f5a6e); border:1px solid var(--border);
  box-shadow:var(--shadow); margin-bottom:24px; animation:rise .8s ease both; }
.hero-in { display:flex; align-items:center; justify-content:space-between; gap:28px; position:relative; z-index:3; }
.hero-title { color:#fff !important; font-size:2.15rem; font-weight:600; line-height:1.15; margin:0; max-width:520px; }
.hero-sub { color:#bfd9e4 !important; margin-top:10px; font-size:1.02rem; max-width:480px; line-height:1.55; }
.hero-chip { display:inline-flex; align-items:center; gap:9px; margin-top:16px; padding:6px 14px; border-radius:999px;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18); color:#e7f3f7 !important; font-size:.85rem; }
.hero-art { flex:0 0 390px; }
.net { width:100%; height:auto; display:block; }
.dot { width:9px; height:9px; border-radius:50%; background:#4cc38a; animation:pulse 1.8s infinite; display:inline-block; }
.waves { position:absolute; left:0; bottom:0; width:100%; height:44px; z-index:2; pointer-events:none; }
.waves svg { position:absolute; bottom:0; left:0; width:200%; height:100%; }
.waves .w1 { animation:drift 14s linear infinite; fill:rgba(98,210,204,.10); }
.waves .w2 { animation:drift 9s linear infinite reverse; fill:rgba(98,210,204,.16); }

/* Network (hero) diagram */
.netbase { fill:none; stroke:rgba(255,255,255,.16); stroke-width:7; stroke-linecap:round; stroke-linejoin:round; }
.netflow { fill:none; stroke:#62d2cc; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:5 11; animation:dash2 1.6s linear infinite; }
.netnode { fill:#e8f6f8; }
.ring { fill:none; stroke:#ee6a70; stroke-width:2; transform-box:fill-box; transform-origin:center; animation:ring 2s ease-out infinite; }
.ring.r2 { animation-delay:1s; }
.leakdot { fill:#ee6a70; animation:blink 1.4s ease-in-out infinite; }
.net-label { fill:#cfe6ee; font-size:12px; font-family:'Manrope',sans-serif; }
.net-leak { fill:#ffb3b7; font-size:12px; font-weight:700; font-family:'Manrope',sans-serif; }

/* Pipe diagram */
.diagram-card { background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:14px 16px 6px;
  box-shadow:var(--shadow); margin:14px 0; animation:rise .6s ease both; }
.diagram { width:100%; height:auto; display:block; }
.pipe-wall { fill:var(--surface-solid); stroke:var(--border); stroke-width:2; }
.water { fill:var(--aqua); opacity:.14; }
.flowline { stroke:var(--aqua); stroke-linecap:round; stroke-dasharray:22 20; animation:dash linear infinite; opacity:.85; fill:none; }
.flowline.thin { opacity:.45; }
.flange { fill:var(--border); }
.ground { stroke:var(--muted); stroke-dasharray:3 6; opacity:.5; }
.svg-text { fill:var(--text); font-family:'Manrope',sans-serif; font-size:13px; font-weight:600; }
.svg-muted { fill:var(--muted); font-family:'Manrope',sans-serif; font-size:12px; }
.svg-title { fill:var(--text); font-size:16px; font-weight:600; }
.crack { stroke:var(--danger); stroke-width:3; fill:none; stroke-linecap:round; stroke-linejoin:round; }
.spurt { fill:var(--danger); opacity:0; animation:spurt 1.6s ease-in infinite; }
.puddle { fill:none; stroke:var(--danger); stroke-width:1.5; transform-box:fill-box; transform-origin:center; animation:puddle 1.6s ease-out infinite; }

/* Architecture diagram */
.box { fill:var(--surface-solid); stroke:var(--border); stroke-width:1.5; }
.archflow { stroke:var(--aqua); stroke-width:2; stroke-dasharray:6 8; animation:dash2 1.2s linear infinite; }
.arrowhead { fill:var(--aqua); }
.box-hi { animation:boxglow 4s ease-in-out infinite; }

/* Loader */
.ldr { display:flex; align-items:center; gap:14px; padding:14px 20px; border-radius:16px; background:var(--surface);
  border:1px solid var(--border); width:fit-content; box-shadow:var(--shadow); animation:rise .4s ease both; margin:10px 0; }
.ldr-orb { position:relative; width:46px; height:46px; border-radius:50%; border:2px solid var(--aqua); overflow:hidden; flex:0 0 auto; }
.ldr-fill { position:absolute; left:0; right:0; bottom:0; height:85%; animation:swell 2s ease-in-out infinite alternate; }
.ldr-fill svg { position:absolute; left:0; bottom:0; width:200%; height:100%; animation:drift 1.4s linear infinite; }
.ldr-text { color:var(--text) !important; font-weight:600; }
.dots i { display:inline-block; width:4px; height:4px; margin-left:3px; border-radius:50%; background:var(--aqua); animation:blink 1.2s infinite; }
.dots i:nth-child(2) { animation-delay:.2s; } .dots i:nth-child(3) { animation-delay:.4s; }

/* Pipe-fill progress */
.pp { margin:10px 0 16px; }
.pp-head { display:flex; justify-content:space-between; color:var(--muted) !important; font-size:.9rem; margin-bottom:6px; }
.pp-track { height:18px; border-radius:99px; background:var(--surface-solid); border:1px solid var(--border); overflow:hidden; }
.pp-fill { height:100%; border-radius:99px; width:var(--w); transition:width .25s ease;
  background:repeating-linear-gradient(110deg, var(--aqua) 0 14px, var(--sky) 14px 28px); background-size:56px 100%; animation:stripes 1s linear infinite; }

/* Result card */
.res { position:relative; border-radius:18px; padding:20px 26px; margin:12px 0; border:1px solid var(--border);
  background:var(--surface); box-shadow:var(--shadow); animation:pop .55s cubic-bezier(.2,1.3,.4,1) both; overflow:hidden; }
.res::before { content:""; position:absolute; left:0; top:0; bottom:0; width:5px; }
.res-title { font-size:1.4rem; font-weight:600; margin:0; }
.res-sub { color:var(--muted) !important; margin-top:2px; }
.res-leak { border-color:var(--danger); animation:pop .55s both, alarm 2.2s ease-in-out .6s infinite; }
.res-leak::before { background:var(--danger); } .res-leak .res-title { color:var(--danger) !important; }
.res-ok { border-color:var(--ok); }
.res-ok::before { background:var(--ok); } .res-ok .res-title { color:var(--ok) !important; }
.bar { height:8px; border-radius:99px; background:var(--border); margin-top:14px; overflow:hidden; }
.bar > i { display:block; height:100%; border-radius:99px; width:var(--w); animation:fill 1.1s ease-out both; }
.res-leak .bar > i { background:linear-gradient(90deg, var(--brass), var(--danger)); }
.res-ok .bar > i { background:linear-gradient(90deg, var(--aqua), var(--ok)); }

/* Inline notes (replace pop-up style alerts) */
.note { display:inline-flex; align-items:center; gap:10px; padding:8px 14px; border-radius:12px; margin:8px 0;
  background:var(--surface); border:1px solid var(--border); color:var(--muted) !important; font-size:.9rem; animation:rise .4s ease both; }
.note.bad { border-color:var(--danger); color:var(--danger) !important; }
.note b { color:var(--ok); }

/* KPIs */
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:6px 0 20px; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:16px 18px; box-shadow:var(--shadow);
  transition:transform .25s ease, box-shadow .25s ease; animation:rise .6s ease both; }
.kpi:hover { transform:translateY(-4px); box-shadow:0 14px 30px var(--glow); }
.kpi-num { font-size:2rem; font-weight:600; animation:count 1.5s ease-out both; counter-reset:num var(--num); color:var(--aqua) !important; }
.kpi-num::after { content:counter(num); }
.kpi-label { color:var(--muted) !important; font-size:.9rem; }
.kpi.bad .kpi-num { color:var(--danger) !important; }
.kpi.dead .kpi-num { color:var(--sky) !important; }

/* Feature grid (About) */
.feats { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin:10px 0 18px; }
.feat { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:16px 18px;
  transition:transform .25s ease, border-color .25s ease; animation:rise .6s ease both; }
.feat:hover { transform:translateY(-4px); border-color:var(--aqua); }
.feat b { font-family:'Fraunces',serif; font-size:1.05rem; color:var(--text) !important; }
.feat div { color:var(--muted) !important; margin-top:4px; font-size:.93rem; }

.status-box { display:flex; align-items:center; gap:10px; padding:10px 14px; border-radius:12px; background:var(--surface);
  border:1px solid var(--border); margin-bottom:8px; }
.status-box span.t { color:var(--text) !important; font-size:.92rem; }

/* Keyframes */
@keyframes drift { from { transform:translateX(0); } to { transform:translateX(-50%); } }
@keyframes dash { to { stroke-dashoffset:-84; } }
@keyframes dash2 { to { stroke-dashoffset:-32; } }
@keyframes swell { from { transform:translateY(55%); } to { transform:translateY(12%); } }
@keyframes blink { 0%,100% { opacity:.25; } 50% { opacity:1; } }
@keyframes ring { 0% { transform:scale(.4); opacity:.9; } 100% { transform:scale(4); opacity:0; } }
@keyframes spurt {
  0% { transform:translate(0,0); opacity:0; } 10% { opacity:1; }
  45% { transform:translate(calc(var(--dx) * .55), -34px); }
  100% { transform:translate(var(--dx), 104px); opacity:.15; }
}
@keyframes puddle { 0% { transform:scale(.3); opacity:.9; } 100% { transform:scale(2.4); opacity:0; } }
@keyframes stripes { to { background-position:56px 0; } }
@keyframes sweep { from { background-position:-50% 0; } to { background-position:150% 0; } }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(76,195,138,.7); } 70% { box-shadow:0 0 0 10px rgba(76,195,138,0); } 100% { box-shadow:0 0 0 0 rgba(76,195,138,0); } }
@keyframes alarm { 0%,100% { box-shadow:0 0 0 0 rgba(238,106,112,0), var(--shadow); } 50% { box-shadow:0 0 0 9px rgba(238,106,112,.14), var(--shadow); } }
@keyframes boxglow { 0%,100% { stroke:var(--border); } 50% { stroke:var(--aqua); } }
@keyframes pop { from { opacity:0; transform:scale(.94) translateY(8px); } to { opacity:1; transform:none; } }
@keyframes rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
@keyframes fill { from { width:0; } to { width:var(--w); } }
@keyframes grow { from { width:0; } to { width:46px; } }
@keyframes count { from { --num:0; } }

@media (max-width:900px) { .hero-art { display:none; } .hero-title { font-size:1.6rem; } }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""


def inject_theme(dark: bool, streamlit_is_dark):
    if dark:
        v = dict(bg="#08131c", bg2="#0b1b27", surface="rgba(18,40,54,0.72)", surface_solid="#112a39",
                 border="rgba(150,190,210,0.16)", text="#e9f0f4", muted="#8ea7b5", aqua="#62d2cc",
                 sky="#5aa9e6", danger="#ee6a70", ok="#4cc38a", brass="#c9b27c", on_accent="#04202c",
                 shadow="0 10px 28px rgba(0,0,0,0.4)", input_bg="#0d2a3a", glow="rgba(98,210,204,0.16)")
    else:
        v = dict(bg="#f4f7f9", bg2="#e9f0f4", surface="rgba(255,255,255,0.9)", surface_solid="#ffffff",
                 border="rgba(16,50,70,0.12)", text="#10222e", muted="#5b7483", aqua="#0f8b95",
                 sky="#2b79b8", danger="#d44049", ok="#1c9a63", brass="#a8873c", on_accent="#ffffff",
                 shadow="0 10px 26px rgba(16,50,70,0.10)", input_bg="#ffffff", glow="rgba(15,139,149,0.14)")

    root = ":root{" + "".join(f"--{k.replace('_', '-')}:{val};" for k, val in v.items()) + "}"

    # Dataframes are drawn by Streamlit itself; invert them if our theme differs from Streamlit's.
    df_filter = ""
    if streamlit_is_dark is not None and streamlit_is_dark != dark:
        df_filter = '[data-testid="stDataFrame"]{filter:invert(1) hue-rotate(180deg);border-radius:12px;}'

    st.markdown(f"<style>{root}{df_filter}</style>", unsafe_allow_html=True)
    st.markdown(STATIC_CSS, unsafe_allow_html=True)


# ---------- Diagrams ----------
NET_PATHS = ["M30 110 H120", "M120 110 V50 H220", "M220 50 H330", "M120 110 H230 V170 H330",
             "M330 50 V110 H400", "M330 170 H400 V110"]
NET_NODES = [(120, 110), (220, 50), (330, 50), (330, 170), (400, 110)]


def network_svg():
    base = "".join(f'<path class="netbase" d="{p}"/>' for p in NET_PATHS)
    flow = "".join(f'<path class="netflow" d="{p}" style="animation-delay:{i * 0.25}s"/>' for i, p in enumerate(NET_PATHS))
    nodes = "".join(f'<circle class="netnode" cx="{x}" cy="{y}" r="5"/>' for x, y in NET_NODES)
    return (
        '<svg class="net" viewBox="0 0 430 210" role="img" aria-label="Animated pipe network with one leak">'
        f'{base}{flow}{nodes}'
        '<circle class="netnode" cx="30" cy="110" r="9"/>'
        '<text class="net-label" x="30" y="136" text-anchor="middle">Source</text>'
        '<circle class="ring" cx="275" cy="50" r="6"/><circle class="ring r2" cx="275" cy="50" r="6"/>'
        '<circle class="leakdot" cx="275" cy="50" r="5"/>'
        '<text class="net-leak" x="275" y="30" text-anchor="middle">Leak</text>'
        '</svg>'
    )


def pipe_diagram(leak, flow=100.0, pressure=50.0):
    """Animated pipe cross-section. Flow speed follows the Flow Rate input; a crack spurts when a leak is predicted."""
    try:
        flow = float(flow)
        pressure = float(pressure)
    except Exception:
        flow, pressure = 100.0, 50.0
    dur = max(0.7, min(5.0, 240.0 / max(flow, 20.0)))
    lx = 330
    if leak:
        lines = (
            f'<line class="flowline" x1="34" y1="95" x2="{lx}" y2="95" stroke-width="30" style="animation-duration:{dur:.2f}s"/>'
            f'<line class="flowline thin" x1="{lx}" y1="95" x2="606" y2="95" stroke-width="16" style="animation-duration:{dur * 1.6:.2f}s"/>'
        )
        extra = (
            f'<path class="crack" d="M{lx - 8} 60 l6 -9 l5 9 l6 -9"/>'
            + "".join(
                f'<circle class="spurt" cx="{lx}" cy="58" r="3.4" style="--dx:{dx}px;animation-delay:{d}s"/>'
                for dx, d in [(22, 0), (40, .25), (58, .5), (30, .8), (50, 1.05)]
            )
            + '<ellipse class="puddle" cx="372" cy="166" rx="12" ry="3"/>'
            + '<ellipse class="puddle" cx="372" cy="166" rx="12" ry="3" style="animation-delay:.6s"/>'
        )
        status = '<text class="svg-text" x="320" y="30" text-anchor="middle" style="fill:var(--danger)">Leak detected, downstream flow reduced</text>'
    else:
        lines = f'<line class="flowline" x1="34" y1="95" x2="606" y2="95" stroke-width="30" style="animation-duration:{dur:.2f}s"/>'
        extra = ""
        status = '<text class="svg-text" x="320" y="30" text-anchor="middle" style="fill:var(--ok)">Normal flow</text>'
    svg = (
        '<svg class="diagram" viewBox="0 0 640 195" role="img" aria-label="Pipe diagram">'
        f'{status}'
        '<rect class="pipe-wall" x="20" y="60" width="600" height="70" rx="10"/>'
        '<rect class="water" x="30" y="70" width="580" height="50" rx="6"/>'
        f'{lines}'
        '<rect class="flange" x="12" y="52" width="14" height="86" rx="4"/>'
        '<rect class="flange" x="614" y="52" width="14" height="86" rx="4"/>'
        '<line class="ground" x1="20" y1="166" x2="620" y2="166"/>'
        f'{extra}'
        '<text class="svg-muted" x="20" y="46">Inlet</text>'
        '<text class="svg-muted" x="620" y="46" text-anchor="end">Outlet</text>'
        f'<text class="svg-muted" x="20" y="188">Pressure {pressure:.0f}</text>'
        f'<text class="svg-muted" x="130" y="188">Flow {flow:.0f}</text>'
        '</svg>'
    )
    st.markdown(f'<div class="diagram-card">{svg}</div>', unsafe_allow_html=True)


def architecture_svg():
    boxes = [(10, "Sensor data", "Pressure, flow, vibration"),
             (240, "ML models", "Four trained classifiers"),
             (470, "Prediction", "Leak or no leak"),
             (700, "Dashboard", "Map, what-if, AI assistant")]
    parts = []
    for i, (x, title, sub) in enumerate(boxes):
        parts.append(f'<rect class="box box-hi" x="{x}" y="30" width="190" height="90" rx="14" style="animation-delay:{i}s"/>')
        parts.append(f'<text class="svg-title" x="{x + 95}" y="68" text-anchor="middle">{title}</text>')
        parts.append(f'<text class="svg-muted" x="{x + 95}" y="92" text-anchor="middle">{sub}</text>')
        if i < len(boxes) - 1:
            parts.append(f'<line class="archflow" x1="{x + 194}" y1="75" x2="{x + 232}" y2="75"/>')
            parts.append(f'<polygon class="arrowhead" points="{x + 228},69 {x + 238},75 {x + 228},81"/>')
    svg = ('<svg class="diagram" viewBox="0 0 900 150" role="img" aria-label="System flow from sensors to dashboard">'
           + "".join(parts) + '</svg>')
    st.markdown(f'<div class="diagram-card">{svg}</div>', unsafe_allow_html=True)


# ---------- Loaders ----------
WAVE_PATH = "M0 10 C11 0 33 20 44 10 C55 0 77 20 88 10 V40 H0Z"


def show_loader(slot, text):
    slot.markdown(
        '<div class="ldr"><div class="ldr-orb"><div class="ldr-fill">'
        f'<svg viewBox="0 0 88 40" preserveAspectRatio="none"><path d="{WAVE_PATH}" fill="var(--aqua)"/></svg>'
        f'</div></div><div class="ldr-text">{text}<span class="dots"><i></i><i></i><i></i></span></div></div>',
        unsafe_allow_html=True,
    )


def show_pipe_progress(slot, frac, text):
    pct = max(0.0, min(1.0, frac)) * 100
    slot.markdown(
        f'<div class="pp"><div class="pp-head"><span>{text}</span><span>{pct:.0f}%</span></div>'
        f'<div class="pp-track"><div class="pp-fill" style="--w:{pct:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


# ---------- Small UI helpers ----------
def render_hero(title, subtitle):
    wave = "M0 22 C150 4 350 40 600 22 C850 4 1050 40 1200 22 C1350 4 1550 40 1800 22 C2050 4 2250 40 2400 22 V44 H0Z"
    html = (
        '<div class="hero"><div class="hero-in"><div class="hero-copy">'
        f'<div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div>'
        '<div class="hero-chip"><span class="dot"></span>Live monitoring</div></div>'
        f'<div class="hero-art">{network_svg()}</div></div>'
        '<div class="waves">'
        f'<svg class="w1" viewBox="0 0 2400 44" preserveAspectRatio="none"><path d="{wave}"/></svg>'
        f'<svg class="w2" viewBox="0 0 2400 44" preserveAspectRatio="none"><path d="{wave}"/></svg>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sect">{text}</div>', unsafe_allow_html=True)


def note(text, bad=False):
    st.markdown(f'<div class="note{" bad" if bad else ""}">{text}</div>', unsafe_allow_html=True)


def result_card(prediction, prob):
    pct = max(0, min(1, float(prob))) * 100
    if prediction == 1:
        title, cls = "⚠ Leakage Detected!", "res-leak"
    else:
        title, cls = "No Leakage Detected", "res-ok"
    st.markdown(
        f'<div class="res {cls}"><div class="res-title">{title}</div>'
        f'<div class="res-sub">Confidence: {float(prob):.2%}</div>'
        f'<div class="bar"><i style="--w:{pct:.1f}%"></i></div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of (label, integer value, css class)"""
    cells = "".join(
        f'<div class="kpi {cls}"><div class="kpi-num" style="--num:{int(val)}"></div>'
        f'<div class="kpi-label">{label}</div></div>'
        for label, val, cls in items
    )
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def style_fig(fig):
    """Make Plotly charts follow the active theme."""
    text = "#e9f0f4" if IS_DARK else "#10222e"
    grid = "rgba(142,167,181,0.18)" if IS_DARK else "rgba(16,50,70,0.10)"
    colors = (["#62d2cc", "#5aa9e6", "#ee6a70", "#c9b27c", "#8f9bd6"] if IS_DARK
              else ["#0f8b95", "#2b79b8", "#d44049", "#a8873c", "#6a76b8"])
    fig.update_layout(
        template="plotly_dark" if IS_DARK else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=text),
        colorway=colors,
        margin=dict(l=20, r=20, t=55, b=20),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor=grid)
    fig.update_yaxes(gridcolor=grid)
    return fig


# ---------------------------------------------------------------------
# Sidebar: theme toggle + navigation
# ---------------------------------------------------------------------
_sys_dark = None
try:
    _sys_dark = st.context.theme.type == "dark"
except Exception:
    pass

st.sidebar.markdown(
    '<div class="brand"><svg viewBox="0 0 24 24"><path d="M12 2 C12 2 5 10 5 15 a7 7 0 0 0 14 0 C19 10 12 2 12 2Z" fill="var(--aqua)"/></svg>AquaGuard</div>',
    unsafe_allow_html=True,
)
theme_choice = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"],
    index=0 if (_sys_dark is None or _sys_dark) else 1,
    horizontal=True,
    key="theme_choice",
)
IS_DARK = theme_choice == "Dark"
inject_theme(IS_DARK, _sys_dark)

# Sidebar
menu = st.sidebar.radio("Navigation", [
    "Predict Leakage",
    "What-if Analysis",
    "Zone Map & Geo Analysis",
    "Batch Prediction",
    "Analytics & Insights",
    "AI Assistant",
    "About"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="status-box"><span class="dot"></span><span class="t">System online</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Last updated {datetime.now().strftime('%Y-%m-%d %H:%M')}")

render_hero(
    "Smart Water Leak Detection & Monitoring",
    "Spot leaks before they spread. Sensor data, ML predictions and a live pipeline map in one place.",
)


# =====================================================================
# BACKEND FUNCTIONS (unchanged logic)
# =====================================================================
# Input Function
def get_user_input():
    section("Enter Pipeline Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        pressure = st.number_input("Pressure", min_value=0.0, value=50.0)
        flow_rate = st.number_input("Flow Rate", min_value=0.0, value=100.0)
        temperature = st.number_input("Temperature", min_value=-50.0, value=25.0)
    with col2:
        vibration = st.number_input("Vibration", min_value=0.0, value=0.5)
        rpm = st.number_input("RPM", min_value=0.0, value=1500.0)
        operational_hours = st.number_input("Operational Hours", min_value=0.0, value=500.0)
    with col3:
        latitude = st.number_input("Latitude", value=25.16)
        longitude = st.number_input("Longitude", value=55.23)

    zone = st.selectbox("Zone", ["Zone_1", "Zone_2", "Zone_3", "Zone_4"])
    block = st.selectbox("Block", ["Block_1", "Block_2", "Block_3", "Block_4"])
    pipe = st.selectbox("Pipe", ["Pipe_1", "Pipe_2", "Pipe_3"])
    location_code = f"{zone}{block}{pipe}"

    input_data = pd.DataFrame(
        [
            {
                "Pressure": pressure,
                "Flow_Rate": flow_rate,
                "Temperature": temperature,
                "Vibration": vibration,
                "RPM": rpm,
                "Operational_Hours": operational_hours,
                "Latitude": latitude,
                "Longitude": longitude,
                "Zone": zone,
                "Block": block,
                "Pipe": pipe,
                "Location_Code": location_code,
            }
        ]
    )
    return input_data

# Prediction
def make_prediction(input_data, model_choice):
    model = models[model_choice]
    prediction = model.predict(input_data)[0]
    try:
        prob = model.predict_proba(input_data)[0][prediction]
    except:
        prob = 0.9
    return prediction, prob


# =====================================================================
# PAGES
# =====================================================================
# Predict Leakage Page
if menu == "Predict Leakage":
    input_data = get_user_input()
    section("Input Data Preview")
    st.dataframe(input_data)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("Predict Leakage"):
        slot = st.empty()
        try:
            show_loader(slot, "Analysing sensor readings")
            time.sleep(LOADER_MIN_SECONDS)
            prediction, prob = make_prediction(input_data, model_choice)
            slot.empty()

            result_card(prediction, prob)
            pipe_diagram(prediction == 1,
                         flow=input_data.iloc[0]["Flow_Rate"],
                         pressure=input_data.iloc[0]["Pressure"])

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(prob) * 100,
                    number={"suffix": "%"},
                    title={"text": "Model confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ee6a70" if prediction == 1 else "#4cc38a"},
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                )
            )
            gauge.update_layout(height=260)
            st.plotly_chart(style_fig(gauge), width="stretch")
        except Exception as e:
            slot.empty()
            note(f"Prediction error: {str(e)}", bad=True)

# What-if Analysis
elif menu == "What-if Analysis":
    section("What-if Analysis (Adjust sliders & see prediction instantly)")
    pressure = st.slider("Pressure", 0.0, 100.0, 50.0)
    flow_rate = st.slider("Flow Rate", 0.0, 200.0, 100.0)
    temperature = st.slider("Temperature", -50.0, 150.0, 25.0)

    input_data = pd.DataFrame(
        [
            {
                "Pressure": pressure,
                "Flow_Rate": flow_rate,
                "Temperature": temperature,
                "Vibration": 0.5,
                "RPM": 1500,
                "Operational_Hours": 500,
                "Latitude": 25.16,
                "Longitude": 55.23,
                "Zone": "Zone_1",
                "Block": "Block_1",
                "Pipe": "Pipe_1",
                "Location_Code": "Zone_1_Block_1_Pipe_1",
            }
        ]
    )

    model_choice = st.selectbox("Choose Model", list(models.keys()), key="whatif_model")
    prediction, prob = make_prediction(input_data, model_choice)

    result_card(prediction, prob)
    pipe_diagram(prediction == 1, flow=flow_rate, pressure=pressure)

    section("Pressure vs Leak Risk Curve")
    pressures = np.linspace(0, 100, 30)
    risks = [make_prediction(input_data.assign(Pressure=p), model_choice)[1] for p in pressures]
    fig = px.line(
        x=pressures,
        y=risks,
        labels={"x": "Pressure", "y": "Leak Probability"},
        title="Leak Risk vs Pressure",
    )
    fig.update_traces(line=dict(color="#62d2cc" if IS_DARK else "#0f8b95", width=3),
                      fill="tozeroy", fillcolor="rgba(98,210,204,0.14)")
    fig.add_vline(x=pressure, line_dash="dot", line_color="#ee6a70",
                  annotation_text="current", annotation_position="top")
    st.plotly_chart(style_fig(fig), width="stretch")

# Zone Map & Geo Analysis
elif menu == "Zone Map & Geo Analysis":
    section("Leakage Risk Map & Zone Analysis")
    slot = st.empty()
    show_loader(slot, "Loading pipeline map")
    time.sleep(LOADER_MIN_SECONDS)
    try:
        df = pd.read_csv(DATASET_PATH)
        using_sample = False
    except:
        using_sample = True
        df = pd.DataFrame(
            {
                "Zone": ["Zone_1", "Zone_2", "Zone_3"],
                "Block": ["Block_1", "Block_2", "Block_3"],
                "Pipe": ["Pipe_1", "Pipe_2", "Pipe_3"],
                "Latitude": [25.16, 25.09, 25.2],
                "Longitude": [55.23, 55.16, 55.18],
                "Leakage_Flag": [0, 1, 0],
                "Flow_Rate": [100, 0, 20],
            }
        )

    df["Pipe_Status"] = df["Flow_Rate"].apply(lambda x: "Dead" if x < 5 else "Active")
    slot.empty()

    if using_sample:
        note("Dataset file not found, showing sample data.")

    kpi_row([
        ("Pipes monitored", len(df), ""),
        ("Active leaks", int((df["Leakage_Flag"] == 1).sum()), "bad"),
        ("Dead pipes", int((df["Pipe_Status"] == "Dead").sum()), "dead"),
        ("Zones", df["Zone"].nunique(), ""),
    ])

    # Advanced Interactive Map
    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=12,
        tiles="CartoDB dark_matter" if IS_DARK else "OpenStreetMap",
    )
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        if row["Leakage_Flag"] == 1:
            color, icon = "red", "tint"
            status = "Leak Detected"
        elif row["Pipe_Status"] == "Dead":
            color, icon = "blue", "times-circle"
            status = "Dead Pipe"
        else:
            color, icon = "green", "check-circle"
            status = "Active"

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"<b>Zone:</b> {row['Zone']}<br><b>Block:</b> {row['Block']}<br><b>Pipe:</b> {row['Pipe']}<br><b>Status:</b> {status}",
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(marker_cluster)

    # Add legend
    legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 120px;
            border:2px solid grey; z-index:9999; font-size:14px; background:white; padding:5px; color:black;">
<b>Legend</b><br>
<i class="fa fa-tint" style="color:red"></i> <span style="color:red;">Leak</span><br>
<i class="fa fa-times-circle" style="color:blue"></i> <span style="color:blue;">Dead</span><br>
<i class="fa fa-check-circle" style="color:green"></i> <span style="color:green;">Active</span>
</div>
"""
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=700, height=500)

    # Zone-wise Analytics
    section("Zone-wise Leak Distribution")
    col1, col2 = st.columns(2)

    with col1:
        zone_counts = df.groupby("Zone")["Leakage_Flag"].sum().reset_index()
        fig_bar = px.bar(
            zone_counts,
            x="Zone",
            y="Leakage_Flag",
            title="Leak Counts per Zone",
            color="Leakage_Flag",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(style_fig(fig_bar), width="stretch")

    with col2:
        zone_pie = df.groupby("Zone")["Leakage_Flag"].sum().reset_index()
        fig_pie = px.pie(
            zone_pie,
            names="Zone",
            values="Leakage_Flag",
            title="Leak Proportion by Zone",
            hole=0.4,
        )
        st.plotly_chart(style_fig(fig_pie), width="stretch")

# Batch Prediction
elif menu == "Batch Prediction":
    section("Upload CSV for Batch Predictions")

    uploaded_file = st.file_uploader("Upload pipeline dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Dataset Summary
        with st.expander("Dataset Info"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing Values", df.isnull().sum().sum())

        # Preview
        section("Preview of Uploaded Data")
        st.dataframe(df.head(10), width="stretch")

        # Model Selection
        section("Select Model for Prediction")
        model_choice = st.selectbox("Choose Model", list(models.keys()), key="batch_model")

        # Run Predictions Button
        if st.button("Run Predictions"):
            preds = []
            pslot = st.empty()
            total_rows = max(len(df), 1)
            for i in range(len(df)):
                input_row = df.iloc[[i]]
                pred, _ = make_prediction(input_row, model_choice)
                preds.append(pred)
                if i % 10 == 0 or i == len(df) - 1:
                    show_pipe_progress(pslot, (i + 1) / total_rows, f"Scanning pipes, {i + 1} of {len(df)}")
            pslot.empty()
            df["Prediction"] = preds

            # Display Results
            note(f"<b>✓</b> {len(df)} predictions completed")
            section("Sample Predictions")
            st.dataframe(df.head(15), width="stretch")

            # Prediction Distribution
            section("Prediction Distribution")
            pred_counts = df["Prediction"].value_counts()
            st.bar_chart(pred_counts)

            # Accuracy Calculation (if ground truth exists)
            if "Leakage_Flag" in df.columns:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(df["Leakage_Flag"], df["Prediction"]) * 100
                st.metric("Accuracy", f"{acc:.2f}%")

            # Download Button
            st.download_button(
                label="Download Predictions",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

# Analytics Page
elif menu == "Analytics & Insights":
    section("Model & System Insights")

    feat_imp = {
        "Pressure": 0.3,
        "Flow_Rate": 0.25,
        "Temperature": 0.15,
        "Vibration": 0.1,
        "RPM": 0.1,
        "Operational_Hours": 0.1,
    }
    fig = px.bar(
        x=list(feat_imp.values()),
        y=list(feat_imp.keys()),
        orientation="h",
        title="Feature Importance",
    )
    fig.update_traces(marker_color="#62d2cc" if IS_DARK else "#0f8b95")
    st.plotly_chart(style_fig(fig), width="stretch")

    section("Model Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    rf_scores = [0.99, 0.98, 0.97, 0.98]
    dt_scores = [0.95, 0.94, 0.92, 0.93]
    log_scores = [0.93, 0.92, 0.9, 0.91]
    svm_scores = [0.94, 0.93, 0.91, 0.92]

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatterpolar(r=rf_scores, theta=metrics, fill="toself", name="Random Forest")
    )
    fig2.add_trace(
        go.Scatterpolar(r=dt_scores, theta=metrics, fill="toself", name="Decision Tree")
    )
    fig2.add_trace(
        go.Scatterpolar(r=log_scores, theta=metrics, fill="toself", name="Logistic Regression")
    )
    fig2.add_trace(go.Scatterpolar(r=svm_scores, theta=metrics, fill="toself", name="SVM"))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    st.plotly_chart(style_fig(fig2), width="stretch")

# About
elif menu == "About":
    section("About Smart Water Leak Detection")
    st.write("This system provides *real-time water leak detection* using ML models on IoT sensor data.")
    architecture_svg()
    feats = [
        ("Predict", "Check a single pipeline and see the result on an animated pipe."),
        ("What-if", "Move the sliders and watch the prediction and flow change live."),
        ("Zone map", "See leaks, active and dead pipes on an interactive map."),
        ("Batch", "Upload a CSV, scan every pipe, download the results."),
        ("Analytics", "Compare models and see which sensors matter most."),
        ("AI assistant", "Ask about trends and request plots in plain language."),
    ]
    cells = "".join(
        f'<div class="feat" style="animation-delay:{i * 0.07}s"><b>{t}</b><div>{d}</div></div>'
        for i, (t, d) in enumerate(feats)
    )
    st.markdown(f'<div class="feats">{cells}</div>', unsafe_allow_html=True)

# AI Assistant
elif menu == "AI Assistant":
    section("Smart AI Assistant (Data-Aware)")

    st.markdown("""
    Ask AI to analyze your pipeline dataset and suggest insights:
    - Leak trends across zones  
    - Pipe condition analysis  
    - Operational suggestions  
    - Predictive maintenance insights  
    - Ask for plots: e.g. "plot: leaks by zone"
    """)

    # Maintain chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load dataset for context
    try:
        df = pd.read_csv(DATASET_PATH)
        # Basic stats
        leak_counts = df["Leakage_Flag"].sum()
        total_pipes = len(df)
        zone_summary = df.groupby("Zone")["Leakage_Flag"].sum().to_dict()
        block_summary = df.groupby("Block")["Leakage_Flag"].sum().to_dict()

        dataset_context = f"""
        Dataset Summary:
        - Total pipes: {total_pipes}
        - Total leaks: {leak_counts}
        - Zone-wise leaks: {zone_summary}
        - Block-wise leaks: {block_summary}
        """
    except:
        df = None
        dataset_context = "No dataset available. Running in general mode."

    # User input (chat-like input box)
    user_input = st.chat_input("Ask something about leaks, trends, or plots...")

    if user_input:
        slot = st.empty()
        try:
            if client is None:
                note("AI Assistant requires GEMINI_API_KEY in the project .env file.", bad=True)
                st.stop()

            # Save user message
            st.session_state.chat_history.append(("User", user_input))

            # AI response
            show_loader(slot, "Reading your pipeline data")
            response = client.models.generate_content(
                model=AI_MODEL,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a water leakage detection expert AI. "
                        "Use dataset insights provided to answer with professional suggestions.\n"
                        + dataset_context  # inject dataset context
                    ),
                    max_output_tokens=2048,
                ),
            )
            answer = response.text or "The model returned an empty response. Try rephrasing your question."
            st.session_state.chat_history.append(("AI", answer))
            slot.empty()

            #  Ask-to-Plot feature
            if df is not None and "plot:" in user_input.lower():
                if "zone" in user_input.lower():
                    fig = px.bar(
                        df.groupby("Zone")["Leakage_Flag"].sum().reset_index(),
                        x="Zone", y="Leakage_Flag", title="Leak Count per Zone",
                        color="Leakage_Flag", color_continuous_scale="Reds"
                    )
                    st.plotly_chart(style_fig(fig), width="stretch")

                elif "block" in user_input.lower():
                    fig = px.bar(
                        df.groupby("Block")["Leakage_Flag"].sum().reset_index(),
                        x="Block", y="Leakage_Flag", title="Leak Count per Block",
                        color="Leakage_Flag", color_continuous_scale="Blues"
                    )
                    st.plotly_chart(style_fig(fig), width="stretch")

                elif "temperature" in user_input.lower():
                    fig = px.scatter(
                        df, x="Temperature", y="Pressure", color="Leakage_Flag",
                        title="Temperature vs Pressure (Leak Highlighted)",
                        symbol="Leakage_Flag"
                    )
                    st.plotly_chart(style_fig(fig), width="stretch")

        except Exception as e:
            slot.empty()
            note(f"AI API Error: {str(e)}", bad=True)

    # Chat UI
    if st.session_state.chat_history:
        for role, msg in st.session_state.chat_history:
            if role == "User":
                with st.chat_message("user"):
                    st.write(msg)
            else:
                with st.chat_message("assistant"):
                    st.write(msg)