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
# =============================

# ===================== Utils Google Sheet ======================
def sheet_url_to_csv(url: str):
    """
    Terima URL Google Sheet biasa dan ubah ke export CSV.
    Mendukung pola:
    - https://docs.google.com/spreadsheets/d/<ID>/edit#gid=<GID>
    - https://docs.google.com/spreadsheets/d/<ID>/ (tanpa gid) -> pakai gid=0
    - Jika user sudah beri URL export, kembalikan apa adanya.
    """
    if not url:
        return None
    if "export?format=csv" in url or "gviz/tq" in url:
        return url  # sudah CSV
    if "docs.google.com/spreadsheets/d/" not in url:
        return None

    # Ambil ID
    try:
        sid = url.split("/spreadsheets/d/")[1].split("/")[0]
    except Exception:
        return None

    # Ambil GID kalau ada
    gid = "0"
    if "#gid=" in url:
        gid = url.split("#gid=")[1].split("&")[0]

    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"

@st.cache_data(show_spinner=False, ttl=10)
def load_sheet(csv_url: str) -> pd.DataFrame:
    """
    Baca CSV publik dari Google Sheet.
    Harap share sheet: Anyone with the link -> Viewer.
    Kolom yang diharapkan: Timestamp, Suhu, Kelembapan
    """
    df = pd.read_csv(csv_url)
    # Normalisasi nama kolom
    df.columns = [c.strip().lower() for c in df.columns]
    # Coba deteksi kolom
    # timestamp bisa bernama 'timestamp' atau 'waktu'
    ts_col = None
    for c in ["timestamp", "waktu", "time", "tanggal"]:
        if c in df.columns:
            ts_col = c
            break
    temp_col = None
    for c in ["suhu", "temperature", "temp", "t"]:
        if c in df.columns:
            temp_col = c
            break
    hum_col = None
    for c in ["kelembapan", "kelembaban", "humidity", "hum", "rh"]:
        if c in df.columns:
            hum_col = c
            break

    # Parse kolom
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df.sort_values(ts_col)
        df = df.reset_index(drop=True)
        df = df.rename(columns={ts_col: "Timestamp"})
    else:
        # Jika tak ada timestamp, buat index berjalan saja
        df = df.reset_index(drop=True)
        df["Timestamp"] = pd.RangeIndex(start=1, stop=len(df)+1, step=1)

    if temp_col:
        df = df.rename(columns={temp_col: "Suhu"})
    if hum_col:
        df = df.rename(columns={hum_col: "Kelembapan"})

    # pastikan kolom angka bertipe float
    if "Suhu" in df.columns:
        df["Suhu"] = pd.to_numeric(df["Suhu"], errors="coerce")
    if "Kelembapan" in df.columns:
        df["Kelembapan"] = pd.to_numeric(df["Kelembapan"], errors="coerce")

    return df

# Cache model agar tidak reload setiap interaksi
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    m = YOLO(path)
    try:
        # aktifkan half precision kalau GPU mendukung (sedikit lebih cepat)
        import torch
        if torch.cuda.is_available():
            m.to("cuda")
    except Exception:
        pass
    return m

def yolo_annotate(bgr_image: np.ndarray, model: YOLO, conf: float, iou: float, imgsz: int):
    """
    Jalankan inferensi dan kembalikan frame ber-anotasi BGR.
    """
    results = model.predict(
        bgr_image, imgsz=imgsz, conf=conf, iou=iou, verbose=False
    )
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

    mode = st.radio("Mode", ["Live Camera", "Gambar (Upload)", "Video (Upload)", "Monitoring (Google Sheet)"])

    if mode == "Monitoring (Google Sheet)":
        st.markdown("### Link Google Sheet")
        sheet_url_input = st.text_input(
            "Tempel URL Google Sheet (share: Anyone with link - Viewer)",
            placeholder="https://docs.google.com/spreadsheets/d/......../edit#gid=0"
        )
        refresh_sec = st.number_input("Auto-refresh (detik)", min_value=0, max_value=120, value=10, step=5,
                                      help="0 = tanpa auto-refresh")
        st.caption("Kolom yang diharapkan: **Timestamp**, **Suhu**, **Kelembapan** (boleh pakai nama serupa).")

# ===================== MODE ======================

# =============== MODE 4: MONITORING (GOOGLE SHEET) ===============
if mode == "Monitoring (Google Sheet)":
    st.subheader("📈 Monitoring Suhu & Kelembapan (Google Sheet)")

    if sheet_url_input:
        csv_url = sheet_url_to_csv(sheet_url_input)
        if not csv_url:
            st.error("URL tidak valid. Tempel URL Google Sheet lengkap (menu Share -> Anyone with link - Viewer).")
        else:
            if st_autorefresh and refresh_sec and refresh_sec > 0:
                # jalankan autorefresh
                st_autorefresh(interval=int(refresh_sec * 1000), key="sheet_refresh")

            try:
                df = load_sheet(csv_url)
                if df.empty:
                    st.warning("Sheet kosong atau belum ada data.")
                else:
                    # pilih kolom tampil
                    cols_show = [c for c in ["Timestamp", "Suhu", "Kelembapan"] if c in df.columns]
                    if "Timestamp" in cols_show:
                        df_view = df[cols_show].copy()
                        st.dataframe(df_view.tail(100), use_container_width=True)

                        # metrics terbaru
                        latest = df.iloc[-1]
                        col1, col2, col3 = st.columns([1,1,2])
                        with col1:
                            if "Suhu" in df.columns:
                                st.metric("Suhu terakhir (°C)", f"{latest.get('Suhu', float('nan')):.2f}")
                        with col2:
                            if "Kelembapan" in df.columns:
                                st.metric("Kelembapan terakhir (%RH)", f"{latest.get('Kelembapan', float('nan')):.1f}")
                        with col3:
                            st.write(f"Terakhir: **{latest['Timestamp']}**")

                        # grafik
                        chart_df = df.tail(200).copy()
                        chart_df = chart_df.set_index("Timestamp")
                        gcols = [c for c in ["Suhu", "Kelembapan"] if c in chart_df.columns]
                        if gcols:
                            st.line_chart(chart_df[gcols])
                        else:
                            st.info("Tidak menemukan kolom angka untuk digrafikkan (Suhu/Kelembapan).")
                    else:
                        st.dataframe(df.tail(100), use_container_width=True)
                        st.info("Tidak ada kolom 'Timestamp'. Data ditampilkan apa adanya.")
            except Exception as e:
                st.error(f"Gagal memuat Sheet: {e}")
                st.stop()
    else:
        st.info("Tempel URL Google Sheet di sidebar.")

# =============== MODE 1: LIVE CAMERA ===============
elif mode == "Live Camera":
    # load model HANYA untuk mode yang butuh
    model = load_model(model_path)

    st.subheader("📷 Live Camera")
    st.write("Pilih resolusi, lalu Start.")

    # Dropdown resolusi
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

    ctx = webrtc_streamer(
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

# =============== MODE 2: GAMBAR (UPLOAD) ===============
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
            # ringkasan deteksi
            st.write("Deteksi per kelas:")
            names = res.names
            counts = {}
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                counts[names[cls_id]] = counts.get(names[cls_id], 0) + 1
            st.json(counts if counts else {"(tidak ada deteksi)": 0})

# =============== MODE 3: VIDEO (UPLOAD) ===============
else:
    model = load_model(model_path)

    st.subheader("🎞️ Deteksi dari Video (Upload)")
    file = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv","webm"])
    if file is not None:
        # simpan ke file temporer
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
