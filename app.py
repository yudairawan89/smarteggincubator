import os
from pathlib import Path
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2, av, time, tempfile, numpy as np
import streamlit as st
import pandas as pd
import pytz
from datetime import datetime

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
MODEL_PATH = "best.pt"     # dipakai internal; input user dihapus
DEFAULT_IMGSZ = 800
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Google Sheet (fixed / tanpa input sidebar)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ssnVf_JS_KrlNYKfSHlxHwttwtntqTY3NdB8KbYrgrQ/edit?usp=sharing"
AUTO_REFRESH_SEC = 10  # 0 untuk nonaktif
TZ_ID = "Asia/Jakarta"
# =============================

# -------------------- Styling Cerah & Elegan --------------------
st.set_page_config(page_title="UHTP Smart Egg Incubator", layout="wide", page_icon="🥚")
st.markdown("""
<style>
:root{
  --brand:#0ea5e9;          /* sky-500 */
  --brand-2:#22c55e;        /* green-500 */
  --ink:#0f172a;            /* slate-900 */
  --ink-soft:#334155;       /* slate-700 */
  --text:#0b1220;           /* nearly black */
  --muted:#475569;          /* slate-600 */
  --card:#ffffffcc;         /* white glass */
  --border:#e5e7eb;         /* gray-200 */
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

/* ================= HEADER PANEL ================= */
.header-wrap {
  background: var(--card);
  backdrop-filter: blur(6px);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 10px 16px;
  box-shadow: 0 10px 30px rgba(2,8,23,.08), inset 0 1px 0 rgba(255,255,255,.65);
  margin-bottom: 8px;
  overflow: visible; /* cegah crop */
}
.header-bar{
  display:flex; align-items:center; justify-content:space-between;
  gap:16px;
}
.header-logo{
  display:flex; align-items:center; justify-content:center;
  width: 170px;                 /* lebar slot kiri/kanan */
  min-height: 78px;             /* tinggi slot agar panel cukup tinggi */
  padding: 4px 0;               /* ruang atas-bawah agar tidak mentok */
}
.header-logo img{
  max-height: 72px;             /* atur ukuran logo di dalam panel */
  width:auto; height:auto;
  object-fit: contain;           /* jangan dipotong */
  display:block;
}
.header-center{
  flex:1; text-align:center;
}
.header-title{
  margin:0; font-weight:900; letter-spacing:.2px; color:#0b1220; font-size:34px;
}
.header-sub{
  font-size:.95rem; color: var(--muted); margin-top:2px;
}

/* Responsif */
@media (max-width: 992px){
  .header-logo{ width:130px; min-height:68px; }
  .header-logo img{ max-height:60px; }
  .header-title{ font-size:28px; }
}
@media (max-width: 680px){
  .header-logo{ width:96px; min-height:56px; }
  .header-logo img{ max-height:50px; }
  .header-title{ font-size:24px; }
}

/* ================= METRIC & LAINNYA ================= */
.metric-card{
  background: var(--card);
  backdrop-filter: blur(6px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(2,8,23,.06);
}
.metric-title{
  font-size: .92rem; color: var(--muted);
  display:flex; gap:.5rem; align-items:center;
}
.metric-value{
  font-size: 2.15rem; font-weight: 800; letter-spacing:.2px;
  background: linear-gradient(90deg, var(--ink), #1d4ed8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.section-title{
  font-weight:800; letter-spacing:.2px; font-size:1.05rem;
  color: var(--ink-soft); margin:.35rem 0 .5rem;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Utils --------------------
def sheet_url_to_csv(url: str):
    """Ubah URL Google Sheet biasa menjadi export CSV."""
    if not url:
        return None
    if "export?format=csv" in url or "gviz/tq" in url:
        return url
    if "docs.google.com/spreadsheets/d/" not in url:
        return None
    try:
        sid = url.split("/spreadsheets/d/")[1].split("/")[0]
    except Exception:
        return None
    gid = "0"
    if "#gid=" in url:
        gid = url.split("#gid=")[1].split("&")[0]
    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"

@st.cache_data(show_spinner=False, ttl=10)
def load_sheet(csv_url: str) -> pd.DataFrame:
    """Baca CSV publik dari Google Sheet dan normalisasi kolom."""
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
    """Format contoh: 22 Agustus 2025, jam 13:45:03 WIB."""
    if ts is None or pd.isna(ts):
        return "-"
    tz = pytz.timezone(TZ_ID)
    if isinstance(ts, str):
        ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts, "tzinfo", None) is None:
        dt = tz.localize(pd.Timestamp(ts).to_pydatetime())
    else:
        dt = ts.tz_convert(tz).to_pydatetime()
    months = ["Januari","Februari","Maret","April","Mei","Juni",
              "Juli","Agustus","September","Oktober","November","Desember"]
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

# ---------- helper: gambar -> data URI untuk dipakai di HTML ----------
def img_to_data_uri(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# -------------------- Header: logo dan judul DI DALAM panel --------------------
def app_header():
    left_uri  = img_to_data_uri("logoseg.png")
    right_uri = img_to_data_uri("logosponsor.png")

    left_img_html  = f"<img src='{left_uri}' alt='logo kiri'/>" if left_uri else ""
    right_img_html = f"<img src='{right_uri}' alt='logo kanan'/>" if right_uri else ""

    st.markdown(
        f"""
        <div class="header-wrap">
          <div class="header-bar">
            <div class="header-logo">{left_img_html}</div>
            <div class="header-center">
              <h1 class="header-title">UHTP Smart Egg Incubator</h1>
              <div class="header-sub">Real-time monitoring • Control • Analytics</div>
            </div>
            <div class="header-logo">{right_img_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

app_header()

# -------------------- Sidebar (tanpa input path model) --------------------
with st.sidebar:
    st.header("Pengaturan")
    conf_thres = st.slider("Confidence", 0.05, 0.95, 0.30, 0.01)
    iou_thres  = st.slider("IoU",        0.10, 0.95, 0.60, 0.01)
    imgsz      = st.select_slider("Image size", options=[640, 800, 960], value=DEFAULT_IMGSZ)
    st.caption("Mode deteksi menggunakan model bawaan.")
    st.divider()
    mode = st.radio("Mode", ["Monitoring (Suhu & Kelembaban)", "Live Camera", "Gambar (Upload)", "Video (Upload)"], index=0)

# -------------------- MODE: Monitoring (default) --------------------
if mode == "Monitoring (Google Sheet)":
    csv_url = sheet_url_to_csv(SHEET_URL)
    if st_autorefresh and AUTO_REFRESH_SEC and AUTO_REFRESH_SEC > 0:
        st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="gsheet_refresh")

    try:
        df = load_sheet(csv_url)
    except Exception as e:
        st.error(f"Gagal memuat Sheet: {e}")
        st.stop()

    if df.empty:
        st.warning("Sheet kosong atau belum ada data.")
    else:
        latest = df.iloc[-1]

        colA, colB, colC, colD, colE = st.columns(5)
        # Kartu metric custom
        def metric_card(col, title, value, icon=None, accent="var(--brand)"):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-left:6px solid {accent}">
                  <div class="metric-title">{icon or ''} {title}</div>
                  <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        if "Suhu Udara (°C)" in df.columns and pd.notna(latest.get("Suhu Udara (°C)")):
            metric_card(colA, "Suhu Udara (°C)", f"{latest['Suhu Udara (°C)']:.1f}", "🌡️", "#38bdf8")
        if "Kelembaban Udara RH (%)" in df.columns and pd.notna(latest.get("Kelembaban Udara RH (%)")):
            metric_card(colB, "Kelembaban Udara RH (%)", f"{latest['Kelembaban Udara RH (%)']:.1f}", "💧", "#34d399")
        if "Curah Hujan (mm)" in df.columns and pd.notna(latest.get("Curah Hujan (mm)")):
            metric_card(colC, "Curah Hujan (mm)", f"{latest['Curah Hujan (mm)']:.1f}", "☔", "#a78bfa")
        if "Kecepatan Angin (m/s)" in df.columns and pd.notna(latest.get("Kecepatan Angin (m/s)")):
            metric_card(colD, "Kecepatan Angin (m/s)", f"{latest['Kecepatan Angin (m/s)']:.1f}", "↯", "#fb7185")
        if "Kelembaban Tanah (%)" in df.columns and pd.notna(latest.get("Kelembaban Tanah (%)")):
            metric_card(colE, "Kelembaban Tanah (%)", f"{latest['Kelembaban Tanah (%)']:.1f}", "🧪", "#f59e0b")

        st.caption(f"Terakhir diperbarui: **{format_wib(latest['Timestamp'])}**")

        st.divider()
        col1, col2 = st.columns(2, gap="large")
        if "Suhu Udara (°C)" in df.columns:
            with col1:
                st.markdown('<div class="section-title">Trend Suhu Udara (°C)</div>', unsafe_allow_html=True)
                ch = df[["Timestamp","Suhu Udara (°C)"]].dropna().set_index("Timestamp").tail(200)
                st.line_chart(ch)
        if "Kelembaban Udara RH (%)" in df.columns:
            with col2:
                st.markdown('<div class="section-title">Trend Kelembaban Udara RH (%)</div>', unsafe_allow_html=True)
                ch = df[["Timestamp","Kelembaban Udara RH (%)"]].dropna().set_index("Timestamp").tail(200)
                st.line_chart(ch)

        st.markdown('<div class="section-title">Data Terbaru</div>', unsafe_allow_html=True)
        show_cols = [c for c in ["Timestamp", "Suhu Udara (°C)", "Kelembaban Udara RH (%)",
                                 "Curah Hujan (mm)", "Kecepatan Angin (m/s)", "Kelembaban Tanah (%)"] if c in df.columns]
        st.dataframe(df[show_cols].tail(100), use_container_width=True)

# -------------------- MODE: Live Camera --------------------
elif mode == "Live Camera":
    model = load_model(MODEL_PATH)

    st.subheader("📷 Live Camera")
    st.write("Pilih resolusi, lalu Start.")

    width_opt  = st.selectbox("Width",  [640, 800, 960, 1280], index=3)
    height_opt = st.selectbox("Height", [480, 600, 720, 960], index=2)

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_thres
            self.iou  = iou_thres
            self.imgsz = imgsz
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            annotated, _ = yolo_annotate(img, model, self.conf, self.iou, self.imgsz)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    constraints = {
        "video": {"width": {"ideal": width_opt}, "height": {"ideal": height_opt}},
        "audio": False
    }

    webrtc_streamer(
        key="yolov12-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=constraints,
        video_processor_factory=LiveProcessor,
    )

# -------------------- MODE: Gambar (Upload) --------------------
elif mode == "Gambar (Upload)":
    model = load_model(MODEL_PATH)

    st.subheader("🖼️ Deteksi dari Gambar")
    file = st.file_uploader("Upload gambar", type=["jpg","jpeg","png","bmp","webp","tif","tiff"])
    if file is not None:
        bytes_data = np.frombuffer(file.read(), np.uint8)
        bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Gagal membaca gambar.")
        else:
            annotated, res = yolo_annotate(bgr, model, conf_thres, iou_thres, imgsz)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Hasil deteksi", use_container_width=True)
            st.write("Deteksi per kelas:")
            names = res.names
            counts = {}
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                counts[names[cls_id]] = counts.get(names[cls_id], 0) + 1
            st.json(counts if counts else {"(tidak ada deteksi)": 0})

# -------------------- MODE: Video (Upload) --------------------
else:
    model = load_model(MODEL_PATH)

    st.subheader("🎞️ Deteksi dari Video (Upload)")
    file = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv","webm"])
    if file is not None:
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix)
        t_in.write(file.read()); t_in.flush(); t_in.close()

        st.info("Memproses video… (akan menghasilkan video dengan bounding box)")
        prog = st.progress(0)
        status = st.empty()

        cap = cv2.VideoCapture(t_in.name)
        if not cap.isOpened():
            st.error("Gagal membuka video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(t_out.name, fourcc, fps, (w, h))

            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, _ = yolo_annotate(frame, model, conf_thres, iou_thres, imgsz)
                writer.write(annotated)
                i += 1
                if total > 0:
                    prog.progress(min(i/total, 1.0))
                    status.text(f"Frame {i}/{total}")
            writer.release(); cap.release()
            prog.progress(1.0); status.text("Selesai ✅")

            st.video(t_out.name)
            with open(t_out.name, "rb") as f:
                st.download_button("Download hasil (MP4)", f, file_name="deteksi.mp4", mime="video/mp4")

