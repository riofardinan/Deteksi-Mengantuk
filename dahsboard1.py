import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import cv2

# Inisialisasi model
@st.cache_resource
def load_models():
    yolo_model = YOLO('model/best.pt')  # Ganti dengan path model YOLO Anda
    mobilenet_model = tf.keras.models.load_model('model/mobilenetv2-91.h5')  # Ganti dengan model MobileNetV2
    mobilenet_classes = ["Normal", "Menguap", "Microsleep"]
    return yolo_model, mobilenet_model, mobilenet_classes

yolo_model, mobilenet_model, mobilenet_classes = load_models()

mobilenet_input_size = (224, 224)  # Sesuaikan dengan input model Anda


# Kelas untuk menangani transformasi video
class DrowsinessDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.yolo_model = yolo_model
        self.mobilenet_model = mobilenet_model
        self.mobilenet_classes = mobilenet_classes
        self.mobilenet_input_size = mobilenet_input_size

    def detect_and_classify(self, frame):
        # Jalankan deteksi YOLO
        yolo_results = self.yolo_model(frame)

        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
                conf = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # ID kelas

                # Dapatkan nama kelas dari model YOLO
                class_name = self.yolo_model.names[class_id]

                if conf > 0.5 and class_name == "face":  # Pastikan hanya wajah yang diproses
                    # Ekstrak ROI (wajah) dari frame
                    face_roi = frame[y1:y2, x1:x2]

                    if face_roi.size > 0:
                        # Ubah ukuran wajah agar sesuai dengan input MobileNetV2
                        resized_face = cv2.resize(face_roi, self.mobilenet_input_size)
                        normalized_face = resized_face / 255.0  # Normalisasi ke [0, 1]
                        input_face = np.expand_dims(normalized_face, axis=0)

                        # Jalankan prediksi dengan MobileNetV2
                        mobilenet_preds = self.mobilenet_model.predict(input_face)
                        class_idx = np.argmax(mobilenet_preds[0])  # Kelas dengan probabilitas tertinggi
                        class_label = self.mobilenet_classes[class_idx]
                        confidence = mobilenet_preds[0][class_idx]

                        # Tambahkan bounding box dan label ke frame
                        if class_label == "Microsleep":
                            color = (0, 0, 255)  # Merah
                        elif class_label == "Menguap":
                            color = (255, 0, 0)  # Biru
                        else:
                            color = (0, 255, 0)  # Hijau
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_label} {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def transform(self, frame):
        # Konversi frame dari WebRTC ke numpy array
        img = frame.to_ndarray(format="bgr24")

        # Jalankan deteksi dan klasifikasi
        processed_frame = self.detect_and_classify(img)

        # Konversi kembali ke VideoFrame
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


# Streamlit UI
st.title("Deteksi Tanda Mengantuk dengan YOLO dan MobileNetV2")
st.text("Menggunakan Streamlit WebRTC untuk akses kamera secara real-time.")

# Streamlit WebRTC dengan kelas DrowsinessDetectionTransformer
webrtc_streamer(key="drowsiness-detection", video_transformer_factory=DrowsinessDetectionTransformer)
