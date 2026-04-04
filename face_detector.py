import os
from pathlib import Path
from PIL import Image
import numpy as np
from mtcnn import MTCNN


class FaceDetector:

    def __init__(self, image_size: tuple = (112, 112), margin_ratio: float = 0.2, min_confidence: float = 0.9):
        self.image_size = image_size
        self.margin_ratio = margin_ratio
        self.min_confidence = min_confidence
        self.detector = MTCNN()

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        detections = self.detector.detect_faces(image)
        return [d for d in detections if d["confidence"] >= self.min_confidence]

    def crop_face(self, image: np.ndarray, box: list) -> Image.Image:
        x, y, w, h = box
        img_h, img_w = image.shape[:2]

        margin_x = int(w * self.margin_ratio)
        margin_y = int(h * self.margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        face = Image.fromarray(image[y1:y2, x1:x2])
        return face.resize(self.image_size, Image.LANCZOS)

    def process_image(self, image_path: str | Path) -> list[Image.Image]:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        detections = self.detect_faces(img_array)

        faces = []
        for det in detections:
            face = self.crop_face(img_array, det["box"])
            faces.append(face)

        return faces

    def process_dataset(self, input_dir: str | Path, output_dir: str | Path) -> dict:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        stats = {"total": 0, "success": 0, "no_face": 0, "error": 0}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        image_paths = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        print(f"Ditemukan {len(image_paths)} gambar di '{input_dir}'")

        for image_path in image_paths:
            stats["total"] += 1
            relative_path = image_path.relative_to(input_dir)
            save_dir = output_dir / relative_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                faces = self.process_image(image_path)

                if not faces:
                    print(f"  [SKIP] Tidak ada wajah: {relative_path}")
                    stats["no_face"] += 1
                    continue

                stem = image_path.stem
                for i, face in enumerate(faces):
                    suffix = f"_face{i}" if len(faces) > 1 else "_face"
                    save_path = save_dir / f"{stem}{suffix}.jpg"
                    face.save(save_path, "JPEG", quality=95)

                print(f"  [OK] {relative_path} -> {len(faces)} wajah disimpan")
                stats["success"] += 1

            except Exception as e:
                print(f"  [ERROR] {relative_path}: {e}")
                stats["error"] += 1

        return stats
