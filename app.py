import os
from pathlib import Path
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2, av, time, tempfile, numpy as np
import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
from typing import Optional  # ✅ untuk Optional[str]

# webrtc
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# YOLO
from ultralytics import YOLO

# (opsional) autorefresh bila lib tersedia
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ======== Konfigurasi ========
MODEL_PATH = "best.pt"     # path model internal
DEFAULT_IMGSZ = 800
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Google Sheet (fixed)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ssnVf_JS_KrlNYKfSHlxHwttwtntqTY3NdB8KbYrgrQ/edit?usp=sharing"
AUTO_REFRESH_SEC = 10  # 0 = nonaktif
TZ_ID = "Asia/Jakarta"
# =============================

# -------------------- Styling --------------------
st.set_page_config(page_title="UHTP Smart Egg Incubator", layout="wide", page_icon="🥚")
st.markdown("""
<style>
:root{
  --brand:#0ea5e9; --brand-2:#22c55e; --ink:#0f172a; --ink-soft:#334155;
  --text:#0b1220; --muted:#475569; --card:#ffffffcc; --border:#e5e7eb;
}
html, body, .stApp {
  background:
    radial-gradient(900px 600px at 10% 0%, rgba(56,189,248,.20), transparent 60%),
    radial-gradient(900px 600px at 90% 10%, rgba(34,197,94,.18), transparent 60%),
    linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%);
  color: var(--text);
}
header, .block-container { padding-top: .5rem; }
h1, h2, h3, h4, h5, h6 { color: var(--ink) !important; }

/* ===== Header panel (logo kiri - judul - logo kanan) ===== */
.header-wrap{
  width:100%;
  background: var(--card);
  backdrop-filter: blur(6px);
  border:1px solid var(--border);
  border-radius:18px;
  padding:14px 18px;
  box-shadow:0 10px 30px rgba(2,8,23,.08), inset 0 1px 0 rgba(255,255,255,.65);
  margin-bottom:8px;
  box-sizing:border-box;
}
.header-grid{
  display:grid;
  grid-template-columns: 190px 1fr 170px;
  align-items:center;
  gap:16px;
}
.header-logo{
  display:flex; align-items:center; justify-content:center;
  min-height:78px;
  padding:4px 0;
}
.header-logo img{
  max-height:72px;
  width:auto; height:auto; display:block; object-fit:contain;
}
/* Perbesar hanya logo kiri (logoseg) */
.header-logo.left img{ max-height:88px; }
.header-logo.left{ min-height:96px; }

.header-center{ text-align:center; }
.header-title{ margin:0; font-weight:900; letter-spacing:.2px; color:#0b1220; font-size:34px; }
.header-sub{ font-size:.95rem; color:var(--muted); margin-top:2px; }

/* Responsif */
@media (max-width: 992px){
  .header-grid{ grid-template-columns: 140px 1fr 130px; }
  .header-logo{ min-height:68px; }
  .header-logo img{ max-height:60px; }
  .header-logo.left img{ max-height:72px; }
  .header-logo.left{ min-height:80px; }
  .header-title{ font-size:28px; }
}
@media (max-width: 680px){
  .header-grid{ grid-template-columns: 110px 1fr 100px; }
  .header-logo{ min-height:56px; }
  .header-logo img{ max-height:50px; }
  .header-logo.left img{ max-height:60px; }
  .header-logo.left{ min-height:66px; }
  .header-title{ font-size:24px; }
}

/* Kartu metric & lainnya */
.metric-card{ background:var(--card); backdrop-filter:blur(6px); border:1px solid var(--border);
  border-radius:16px; padding:14px 16px; box-shadow:0 8px 24px rgba(2,8,23,.06); }
.metric-title{ font-size:.92rem; color:var(--muted); display:flex; gap:.5rem; align-items:center; }
.metric-value{ font-size:2.15rem; font-weight:800; letter-spacing:.2px;
  background:linear-gradient(90deg, var(--ink), #1d4ed8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.section-title{ font-weight:800; letter-spacing:.2px; font-size:1.05rem; color:var(--ink-soft); margin:.35rem 0 .5rem; }
div[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Utils --------------------
def sheet_url_to_csv(url: str):
    if not url: return None
    if "export?format=csv" in url or "gviz/tq" in url: return url
    if "docs.google.com/spreadsheets/d/" not in url: return None
    try: sid = url.split("/spreadsheets/d/")[1].split("/")[0]
    except Exception: return None
    gid = "0"
    if "#gid=" in url: gid = url.split("#gid=")[1].split("&")[0]
    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"

@st.cache_data(show_spinner=False, ttl=10)
def load_sheet(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    df.columns = [c.strip().lower() for c in df.columns]

    def pick(*cands):
        for c in cands:
            if c in df.columns: return c
        return None

    ts_col  = pick("timestamp","waktu","time","tanggal")
    t_col   = pick("suhu udara (°c)","suhu","temperature","temp","t")
    rh_col  = pick("kelembaban udara rh (%)","kelembapan","kelembaban","humidity","hum","rh")
    rain_col= pick("curah hujan (mm)","curah hujan","rain","precipitation")
    wind_col= pick("kecepatan angin (m/s)","kecepatan angin","wind speed")
    soil_col= pick("kelembaban tanah (%)","kelembaban tanah","soil moisture")

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        df = df.rename(columns={ts_col:"Timestamp"})
    else:
        df = df.reset_index(drop=True)
        df["Timestamp"] = pd.RangeIndex(1, len(df)+1)

    rename_map = {}
    if t_col:    rename_map[t_col]    = "Suhu Udara (°C)"
    if rh_col:   rename_map[rh_col]   = "Kelembaban Udara RH (%)"
    if rain_col: rename_map[rain_col] = "Curah Hujan (mm)"
    if wind_col: rename_map[wind_col] = "Kecepatan Angin (m/s)"
    if soil_col: rename_map[soil_col] = "Kelembaban Tanah (%)"
    df = df.rename(columns=rename_map)

    for col in ["Suhu Udara (°C)", "Kelembaban Udara RH (%)", "Curah Hujan (mm)",
                "Kecepatan Angin (m/s)", "Kelembaban Tanah (%)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def format_wib(ts) -> str:
    if ts is None or pd.isna(ts): return "-"
    tz = pytz.timezone(TZ_ID)
    if isinstance(ts, str): ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts, "tzinfo", None) is None:
        dt = tz.localize(pd.Timestamp(ts).to_pydatetime())
    else:
        dt = ts.tz_convert(tz).to_pydatetime()
    months = ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"]
    return f"{dt.day} {months[dt.month-1]} {dt.year}, jam {dt:%H:%M:%S} WIB"

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    m = YOLO(path)
    try:
        import torch
        if torch.cuda.is_available():
            m.to("cuda")
    except Exception:
        pass
    return m

def yolo_annotate(bgr_image: np.ndarray, model: YOLO, conf: float, iou: float, imgsz: int):
    results = model.predict(bgr_image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    annotated = results[0].plot()
    return annotated, results[0]

# -------- helper: gambar -> data URI --------
def img_to_data_uri(path: str) -> Optional[str]:   # ✅ diperbaiki
    p = Path(path)
    if not p.exists(): return None
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# -------------------- Header --------------------
def app_header():
    left_uri  = img_to_data_uri("logoseg.png")
    right_uri = img_to_data_uri("logosponsor.png")
    left_img_html  = f"<img src='{left_uri}' alt='logo kiri'/>" if left_uri else ""
    right_img_html = f"<img src='{right_uri}' alt='logo kanan'/>" if right_uri else ""

    st.markdown(
        f"""
        <div class="header-wrap">
          <div class="header-grid">
            <div class="header-logo left">{left_img_html}</div>
            <div class="header-center">
              <h1 class="header-title">UHTP Smart Egg Incubator</h1>
              <div class="header-sub">Real-time monitoring • Control • Analytics</div>
            </div>
            <div class="header-logo right">{right_img_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

app_header()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Pengaturan")
    conf_thres = st.slider("Confidence", 0.05, 0.95, 0.30, 0.01)
    iou_thres  = st.slider("IoU", 0.10, 0.95, 0.60, 0.01)
    imgsz      = st.select_slider("Image size", options=[640, 800, 960], value=DEFAULT_IMGSZ)
    st.caption("Mode deteksi menggunakan model bawaan.")
    st.divider()
    mode = st.radio("Mode", ["Monitoring (Suhu dan Kelembaban)", "Live Camera", "Gambar (Upload)", "Video (Upload)"], index=0)

# -------------------- MODE: Monitoring --------------------
# ... (lanjutan script Anda tetap sama, hanya bagian Optional[str] yang diubah)
