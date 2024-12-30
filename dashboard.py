import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Fungsi untuk menjalankan deteksi
def detect_and_classify(frame, yolo_model, mobilenet_model, mobilenet_classes, mobilenet_input_size):
    yolo_results = yolo_model(frame)

    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            conf = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # ID kelas

            # Dapatkan nama kelas dari model YOLO
            class_name = yolo_model.names[class_id]

            if conf > 0.5 and class_name == "face":  # Pastikan hanya wajah yang diproses
                # Ekstrak ROI (wajah) dari frame
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    # Ubah ukuran wajah agar sesuai dengan input MobileNetV2
                    resized_face = cv2.resize(face_roi, mobilenet_input_size)
                    normalized_face = resized_face / 255.0  # Normalisasi ke [0, 1]
                    input_face = np.expand_dims(normalized_face, axis=0)

                    # Jalankan prediksi dengan MobileNetV2
                    mobilenet_preds = mobilenet_model.predict(input_face)
                    class_idx = np.argmax(mobilenet_preds[0])  # Kelas dengan probabilitas tertinggi
                    class_label = mobilenet_classes[class_idx]
                    confidence = mobilenet_preds[0][class_idx]

                    # Tambahkan bounding box dan label ke frame
                    if class_label == "Microsleep":
                        color = (0, 0, 255) # Merah
                    elif class_label == "Menguap":
                        color = (255, 0, 0) # Biru
                    else:
                        color = (0, 255, 0) # Hijau
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Streamlit setup
st.title("Deteksi Tanda Mengantuk")
st.text("YOLO dan MobileNetV2")

# Inisialisasi model
@st.cache_resource
def load_models():
    yolo_model = YOLO('model/best.pt')
    mobilenet_model = tf.keras.models.load_model('model/mobilenetv2-91.h5')
    mobilenet_classes = ["Normal", "Menguap", "Microsleep"]
    return yolo_model, mobilenet_model, mobilenet_classes

yolo_model, mobilenet_model, mobilenet_classes = load_models()

mobilenet_input_size = (224, 224)  # Sesuaikan dengan input model Anda

col1, col2 = st.columns(2)

# Akses kamera
with col1:
    run_camera = st.button("Aktifkan Kamera")

with col2:
    stop_camera = st.button("Stop Kamera")

# Menggunakan session_state untuk memantau status kamera
if "cap" not in st.session_state:
    st.session_state.cap = None

if run_camera and st.session_state.cap is None:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.running = True

# Variabel untuk menghitung FPS
frame_count = 0
start_time = time.time()

if st.session_state.cap is not None and st.session_state.running:
    st_frame = st.empty()

    while st.session_state.running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Tidak dapat membaca data dari kamera.")
            break

        # Jalankan deteksi dan klasifikasi
        frame = detect_and_classify(frame, yolo_model, mobilenet_model, mobilenet_classes, mobilenet_input_size)

        # Hitung FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # Setiap detik, hitung FPS
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Tampilkan FPS di frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Konversi frame ke format yang sesuai untuk Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame, channels="RGB")

        # Periksa apakah tombol "Stop Kamera" ditekan
        if stop_camera:
            st.session_state.running = False
            st.session_state.cap.release()
            st.session_state.cap = None
            st.success("Kamera telah dihentikan.")