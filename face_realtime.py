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
        vote_size=20,
        detect_interval=20,
        confidence_threshold=0.65
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.detector = MTCNN()
        self.preprocess = FacePreprocessing(img_size)

        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()}

        self.vote_size = vote_size
        self.pred_queues = {}

        self.confidence_threshold = confidence_threshold

        self.detect_interval = detect_interval
        self.frame_count = 0
        self.last_faces = []

    # =====================================================
    # 🔹 RESIZE (ANTI DISTORSI)
    # =====================================================
    def resize_with_aspect_ratio(self, frame, max_width=800):
        h, w = frame.shape[:2]

        if w <= max_width:
            return frame

        scale = max_width / w
        return cv2.resize(frame, (int(w*scale), int(h*scale)))

    # =====================================================
    # 🔹 SAFE CROP
    # =====================================================
    def safe_crop(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return frame[y1:y2, x1:x2]

    # =====================================================
    # 🔹 PREDICT
    # =====================================================
    def predict_face(self, face_img, keypoints=None):
        try:
            img = self.preprocess.process_for_model(face_img, keypoints)
            pred = self.model.predict(img, verbose=0)[0]

            label = int(np.argmax(pred))
            confidence = float(np.max(pred))

            if confidence < self.confidence_threshold:
                return "unknown", confidence

            return label, confidence

        except:
            return "unknown", 0.0

    # =====================================================
    # 🔹 VOTING
    # =====================================================
    def voting(self, face_id, label):
        if face_id not in self.pred_queues:
            self.pred_queues[face_id] = deque(maxlen=self.vote_size)

        q = self.pred_queues[face_id]
        q.append(label)

        return Counter(q).most_common(1)[0][0]

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

            frame = self.resize_with_aspect_ratio(frame)

            self.frame_count += 1

            # 🔥 DETEKSI PERIODIK
            if self.frame_count % self.detect_interval == 0:
                faces = self.detector.detect_faces(frame)
                self.last_faces = faces
            else:
                faces = self.last_faces

            # =====================================================
            # 🔹 LOOP FACE
            # =====================================================
            for idx, face in enumerate(faces):
                try:
                    x, y, w, h = face['box']
                    keypoints = face.get('keypoints', None)

                    # 🔥 margin biar tidak terlalu sempit
                    margin = int(0.2 * w)

                    x1 = x - margin
                    y1 = y - margin
                    x2 = x + w + margin
                    y2 = y + h + margin

                    face_img = self.safe_crop(frame, x1, y1, x2, y2)

                    if face_img.size == 0:
                        continue

                    # 🔥 adjust keypoints ke local crop
                    if keypoints is not None:
                        try:
                            kp = {
                                'left_eye': (
                                    keypoints['left_eye'][0] - x1,
                                    keypoints['left_eye'][1] - y1
                                ),
                                'right_eye': (
                                    keypoints['right_eye'][0] - x1,
                                    keypoints['right_eye'][1] - y1
                                )
                            }
                        except:
                            kp = None
                    else:
                        kp = None

                    # 🔥 predict
                    label, confidence = self.predict_face(face_img, kp)

                    # 🔥 voting
                    final_label = self.voting(idx, label)

                    if final_label == "unknown":
                        name = "Unknown"
                    else:
                        name = self.inverse_label_map.get(final_label, "Unknown")

                    # 🔹 draw
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                    cv2.putText(
                        frame,
                        f"{name} ({confidence:.2f})",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2
                    )

                except:
                    continue

            cv2.imshow("Face Recognition (Stable)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()