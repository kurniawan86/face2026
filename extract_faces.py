from pathlib import Path
from face_detector import FaceDetector


BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "DATASET"
OUTPUT_DIR = BASE_DIR / "DATASET_FACES"

def main():
    detector = FaceDetector(
        image_size=(112, 112),
        margin_ratio=0.2,
        min_confidence=0.9,
    )

    for split in ["TRAINING", "TESTING"]:
        input_split = INPUT_DIR / split
        output_split = OUTPUT_DIR / split

        if not input_split.exists():
            print(f"Folder tidak ditemukan: {input_split}")
            continue

        print(f"\n{'='*50}")
        print(f"Memproses: {split}")
        print(f"{'='*50}")

        stats = detector.process_dataset(input_split, output_split)

        print(f"\nRingkasan {split}:")
        print(f"  Total gambar : {stats['total']}")
        print(f"  Berhasil     : {stats['success']}")
        print(f"  Tidak ada wajah: {stats['no_face']}")
        print(f"  Error        : {stats['error']}")

    print(f"\nSelesai! Hasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
