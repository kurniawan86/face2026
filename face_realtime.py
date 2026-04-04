import cv2
import numpy as np
from collections import deque, Counter
from mtcnn import MTCNN
import tensorflow as tf

from preprocessing import FacePreprocessing


class FaceRealtime:
    def __init__(
        self,
        model_path,
        label_map,
        img_size=(112,112),
        vote_size=5,
        detect_interval=5   # 🔥 kunci utama
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.detector = MTCNN()
        self.preprocess = FacePreprocessing(img_size)

        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()}

        self.vote_size = vote_size
        self.pred_queue = deque(maxlen=vote_size)

        # 🔥 optimasi
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.last_faces = []

    # =====================================================
    # 🔹 PREDICT
    # =====================================================
    def predict_face(self, face_img):
        img = self.preprocess.process_for_model(face_img)
        pred = self.model.predict(img, verbose=0)
        return np.argmax(pred)

    # =====================================================
    # 🔹 VOTING
    # =====================================================
    def voting(self, label):
        self.pred_queue.append(label)

        if len(self.pred_queue) == self.vote_size:
            return Counter(self.pred_queue).most_common(1)[0][0]
        return label

    def resize_with_aspect_ratio(self, frame, max_width=640):
        h, w = frame.shape[:2]

        if w <= max_width:
            return frame

        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h))

    # =====================================================
    # 🔹 MAIN LOOP
    # =====================================================
    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Kamera tidak bisa dibuka")
            return

        print("Tekan 'q' untuk keluar")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.resize_with_aspect_ratio(frame,max_width=700)

            self.frame_count += 1

            # =====================================================
            # 🔥 DETEKSI HANYA TIAP N FRAME
            # =====================================================
            if self.frame_count % self.detect_interval == 0:
                faces = self.detector.detect_faces(frame)
                self.last_faces = faces
            else:
                faces = self.last_faces

            # =====================================================
            # 🔹 LOOP WAJAH
            # =====================================================
            for face in faces:
                x, y, w, h = face['box']

                x, y = max(0, x), max(0, y)

                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                # 🔹 predict
                label = self.predict_face(face_img)

                # 🔹 voting
                final_label = self.voting(label)

                name = self.inverse_label_map.get(final_label, "Unknown")

                # 🔹 bounding box
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

                # 🔹 label
                cv2.putText(
                    frame,
                    name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            # =====================================================
            # 🔹 DISPLAY
            # =====================================================
            cv2.imshow("Face Realtime (Fast MTCNN)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
