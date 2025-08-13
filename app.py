import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2, av, time, tempfile, numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# ======== Konfigurasi ========
MODEL_PATH = "best.pt"     # ganti jika perlu
DEFAULT_IMGSZ = 800        # 640–800 bagus untuk retakan halus
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# =============================

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
    mode = st.radio("Mode", ["Live Camera", "Gambar (Upload)", "Video (Upload)"])
    st.markdown("> Kelas: **Telur**, **Retak**, **Anak Ayam**")

# load model
model = load_model(model_path)

# =============== MODE 1: LIVE CAMERA ===============
if mode == "Live Camera":
    st.subheader("📷 Live Camera")
    st.write("Bounding box akan tampil langsung di stream.")

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_thres
            self.iou  = iou_thres
            self.imgsz = imgsz

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            annotated, _ = yolo_annotate(img, model, self.conf, self.iou, self.imgsz)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolov12-live",
        mode=WebRtcMode.SENDRECV,   # <-- BUKAN "SENDRECV" (string)
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=LiveProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# =============== MODE 2: GAMBAR (UPLOAD) ===============
elif mode == "Gambar (Upload)":
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

