import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2, av, time, tempfile, numpy as np
import streamlit as st
import pandas as pd

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
MODEL_PATH = "best.pt"     # ganti jika perlu
DEFAULT_IMGSZ = 800        # 640–800 bagus untuk retakan halus
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Google Sheet (hardcoded, tidak perlu input user)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ssnVf_JS_KrlNYKfSHlxHwttwtntqTY3NdB8KbYrgrQ/edit?usp=sharing"
AUTO_REFRESH_SEC = 10  # 0 untuk nonaktif
# =============================

# ===================== Utils Google Sheet ======================
def sheet_url_to_csv(url: str):
    """
    Ubah URL Google Sheet biasa menjadi export CSV.
    """
    if not url:
        return None
    if "export?format=csv" in url or "gviz/tq" in url:
        return url  # sudah CSV
    if "docs.google.com/spreadsheets/d/" not in url:
        return None

    # Ambil spreadsheet ID
    try:
        sid = url.split("/spreadsheets/d/")[1].split("/")[0]
    except Exception:
        return None

    # Ambil gid jika ada, kalau tidak gunakan 0
    gid = "0"
    if "#gid=" in url:
        gid = url.split("#gid=")[1].split("&")[0]

    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"

@st.cache_data(show_spinner=False, ttl=10)
def load_sheet(csv_url: str) -> pd.DataFrame:
    """
    Baca CSV publik dari Google Sheet.
    Kolom yang didukung (pakai salah satu sinonim):
      - Timestamp: timestamp / waktu / time / tanggal
      - Suhu: suhu / suhu udara / temperature / temp / t
      - Kelembapan: kelembapan / kelembaban / kelembaban udara rh / humidity / rh
      - Curah Hujan: curah hujan / rain / precipitation
      - Kecepatan Angin: kecepatan angin / wind speed
      - Kelembaban Tanah: kelembaban tanah / soil moisture
    """
    df = pd.read_csv(csv_url)

    # Lower-case kolom untuk pencarian mudah
    orig_cols = list(df.columns)
    lower_map = {c: c.strip().lower() for c in df.columns}
    df.columns = [lower_map[c] for c in df.columns]

    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    ts_col  = pick("timestamp", "waktu", "time", "tanggal")
    t_col   = pick("suhu udara (°c)", "suhu", "temperature", "temp", "t")
    rh_col  = pick("kelembaban udara rh (%)", "kelembapan", "kelembaban", "humidity", "hum", "rh")
    rain_col= pick("curah hujan (mm)", "curah hujan", "rain", "precipitation")
    wind_col= pick("kecepatan angin (m/s)", "kecepatan angin", "wind speed")
    soil_col= pick("kelembaban tanah (%)", "kelembaban tanah", "soil moisture")

    # Parse waktu (jika ada)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        df = df.rename(columns={ts_col: "Timestamp"})
    else:
        df = df.reset_index(drop=True)
        df["Timestamp"] = pd.RangeIndex(1, len(df) + 1)

    # Rename kolom angka ke nama konsisten
    rename_map = {}
    if t_col: rename_map[t_col] = "Suhu Udara (°C)"
    if rh_col: rename_map[rh_col] = "Kelembaban Udara RH (%)"
    if rain_col: rename_map[rain_col] = "Curah Hujan (mm)"
    if wind_col: rename_map[wind_col] = "Kecepatan Angin (m/s)"
    if soil_col: rename_map[soil_col] = "Kelembaban Tanah (%)"
    df = df.rename(columns=rename_map)

    # Konversi numerik
    for col in ["Suhu Udara (°C)", "Kelembaban Udara RH (%)", "Curah Hujan (mm)",
                "Kecepatan Angin (m/s)", "Kelembaban Tanah (%)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# Cache model agar tidak reload setiap interaksi
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
    annotated = results[0].plot()  # BGR ndarray dengan bbox/label
    return annotated, results[0]

# ===================== UI ======================
st.set_page_config(page_title="Deteksi Telur • Streamlit", layout="wide")
st.title("🥚 Deteksi Telur • YOLOv12 (Streamlit)")

with st.sidebar:
    st.header("Pengaturan")
    model_path = st.text_input("Path model (.pt)", MODEL_PATH)
    conf_thres = st.slider("Confidence", 0.05, 0.95, 0.30, 0.01)
    iou_thres  = st.slider("IoU",        0.10, 0.95, 0.60, 0.01)
    imgsz      = st.select_slider("Image size", options=[640, 800, 960], value=DEFAULT_IMGSZ)
    st.caption("Tips: 640/800 cukup cepat; 960 lebih teliti untuk retakan kecil.")
    st.divider()
    mode = st.radio("Mode", ["Monitoring (Google Sheet)", "Live Camera", "Gambar (Upload)", "Video (Upload)"], index=0)
    st.markdown("> Kelas: **Telur**, **Retak**, **Anak Ayam**")

# ===================== MODE ======================

# =============== MODE 1 (default): MONITORING (GOOGLE SHEET) ===============
if mode == "Monitoring (Google Sheet)":
    st.subheader("📈 Hasil Prediksi Terkini")

    csv_url = sheet_url_to_csv(SHEET_URL)
    if not csv_url:
        st.error("URL Google Sheet tidak valid.")
        st.stop()

    # Auto-refresh jika lib tersedia
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
        # METRICS TERKINI
        latest = df.iloc[-1]
        cols = st.columns(5)
        def m(col, title, key):
            if key in df.columns and pd.notna(latest.get(key)):
                with col:
                    st.metric(title, f"{latest.get(key):.1f}")
        m(cols[0], "Suhu Udara (°C)", "Suhu Udara (°C)")
        m(cols[1], "Kelembaban Udara RH (%)", "Kelembaban Udara RH (%)")
        m(cols[2], "Curah Hujan (mm)", "Curah Hujan (mm)")
        m(cols[3], "Kecepatan Angin (m/s)", "Kecepatan Angin (m/s)")
        m(cols[4], "Kelembaban Tanah (%)", "Kelembaban Tanah (%)")

        st.caption(f"Terakhir diperbarui: **{latest['Timestamp']}**")

        st.divider()
        colA, colB = st.columns(2)
        # Grafik SUHU
        if "Suhu Udara (°C)" in df.columns:
            with colA:
                st.markdown("#### Suhu Udara (°C)")
                ch = df[["Timestamp", "Suhu Udara (°C)"]].dropna().set_index("Timestamp").tail(200)
                st.line_chart(ch)
        # Grafik RH
        if "Kelembaban Udara RH (%)" in df.columns:
            with colB:
                st.markdown("#### Kelembaban Udara RH (%)")
                ch = df[["Timestamp", "Kelembaban Udara RH (%)"]].dropna().set_index("Timestamp").tail(200)
                st.line_chart(ch)

        # Tabel ringkas
        st.markdown("#### Data Terbaru")
        show_cols = [c for c in ["Timestamp", "Suhu Udara (°C)", "Kelembaban Udara RH (%)",
                                 "Curah Hujan (mm)", "Kecepatan Angin (m/s)", "Kelembaban Tanah (%)"] if c in df.columns]
        st.dataframe(df[show_cols].tail(100), use_container_width=True)

# =============== MODE 2: LIVE CAMERA ===============
elif mode == "Live Camera":
    model = load_model(model_path)

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
        "video": {
            "width": {"ideal": width_opt},
            "height": {"ideal": height_opt}
        },
        "audio": False
    }

    webrtc_streamer(
        key="yolov12-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=constraints,
        video_processor_factory=LiveProcessor,
    )

    st.caption(
        "Jika kamera tidak tampil: pastikan izin kamera sudah diizinkan, "
        "tidak dipakai aplikasi lain, gunakan HTTPS/localhost, lalu refresh."
    )

# =============== MODE 3: GAMBAR (UPLOAD) ===============
elif mode == "Gambar (Upload)":
    model = load_model(model_path)

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

# =============== MODE 4: VIDEO (UPLOAD) ===============
else:
    model = load_model(model_path)

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
