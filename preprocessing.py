import cv2
import numpy as np
import random
import sys
from pathlib import Path

class FacePreprocessing:
    img_size = []
    augment_flag = False

    def __init__(self, img_size=(112, 112), augment=False):
        self.img_size = img_size
        self.augment_flag = augment

    # Resize
    def resize(self, img):
        return cv2.resize(img, self.img_size)

    # BGR → RGB
    def to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalisasi
    def normalize(self, img):
        return img.astype("float32") / 255.0

    # Tambah batch dimensi
    def add_batch_dim(self, img):
        return np.expand_dims(img, axis=0)

    # FACE ALIGNMENT (berdasarkan posisi mata)
    def align(self, img, keypoints):
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        # hitung sudut rotasi
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # titik tengah mata
        center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2)
        )

        # rotasi
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return aligned

    # AUGMENTASI
    def augment(self, img):
        # horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # brightness
        if random.random() > 0.5:
            factor = 0.7 + 0.6 * random.random()
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # rotation kecil
        if random.random() > 0.5:
            angle = random.randint(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))

        return img

    # PIPELINE TRAINING (ALIGN + AUGMENT)
    def process_training(self, img, keypoints=None):
        if keypoints is not None:
            img = self.align(img, keypoints)

        if self.augment_flag:
            img = self.augment(img)

        img = self.resize(img)
        img = self.to_rgb(img)
        img = self.normalize(img)

        return img

    # PIPELINE INFERENCE (TANPA AUGMENT)
    def process_inference(self, img, keypoints=None):
        if keypoints is not None:
            img = self.align(img, keypoints)

        img = self.resize(img)
        img = self.to_rgb(img)
        img = self.normalize(img)

        return img

    # Siap untuk model
    def process_for_model(self, img, keypoints=None):
        img = self.process_inference(img, keypoints)
        img = self.add_batch_dim(img)
        return img


def _find_sample_image(base_dir: Path) -> Path | None:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidate_dirs = [
        base_dir / "DATASET_FACES",
        base_dir / "DATASET",
    ]

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue

        for image_path in sorted(candidate_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                return image_path

    return None

if __name__ == "__main__":
    import cv2

    # 🔹 load gambar
    image_path = "DATASET_FACES/TRAINING/ADEL/1.jpg"
    img = cv2.imread(image_path)

    if img is None:
        print("Gagal load gambar!")
        exit()

    preprocessor = FacePreprocessing()

    # =====================================================
    # 🔹 SIMULASI KEYPOINT (kalau belum pakai MTCNN)
    # =====================================================
    h, w = img.shape[:2]

    keypoints = {
        'left_eye': (int(w*0.35), int(h*0.4)),
        'right_eye': (int(w*0.65), int(h*0.42))  # sengaja agak miring
    }

    # =====================================================
    # 🔹 SEBELUM ALIGNMENT
    # =====================================================
    before = preprocessor.process_inference(img)

    # =====================================================
    # 🔹 SESUDAH ALIGNMENT
    # =====================================================
    aligned_img = preprocessor.align(img, keypoints)
    after = preprocessor.process_inference(aligned_img)

    # =====================================================
    # 🔹 CONVERT UNTUK DISPLAY
    # =====================================================
    def to_bgr_show(x):
        x = (x * 255).astype("uint8")
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    before_show = to_bgr_show(before)
    after_show = to_bgr_show(after)

    # =====================================================
    # 🔹 GAMBAR ORIGINAL + TITIK MATA
    # =====================================================
    vis = img.copy()
    cv2.circle(vis, keypoints['left_eye'], 5, (0,255,0), -1)
    cv2.circle(vis, keypoints['right_eye'], 5, (0,0,255), -1)

    # =====================================================
    # 🔹 TAMPILKAN
    # =====================================================
    cv2.imshow("Original + Keypoints", vis)
    cv2.imshow("Before Alignment", before_show)
    cv2.imshow("After Alignment", after_show)

    cv2.waitKey(0)
    cv2.destroyAllWindows()