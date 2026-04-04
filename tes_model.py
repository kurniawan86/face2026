from datasetloader import DatasetLoader
from cnn_face import CNNface

# =====================================================
# 🔹 CONFIG
# =====================================================
DATASET_PATH = "DATASET_FACES"
MODEL_PATH = "face_model.keras"


if __name__ == "__main__":

    # =====================================================
    # 🔹 LOAD DATASET
    # =====================================================
    print("Loading dataset...")

    loader = DatasetLoader(DATASET_PATH, augment=False)
    X_train, X_test, y_train, y_test = loader.load_data()

    print("Dataset loaded")
    print("Test shape:", X_test.shape)

    # =====================================================
    # 🔹 INIT MODEL (TANPA BUILD)
    # =====================================================
    cnn = CNNface(
        input_shape=(112,112,3),
        num_classes=len(loader.label_map)
    )

    # =====================================================
    # 🔹 EVALUATE (LOAD MODEL .KERAS)
    # =====================================================
    print("\nEvaluating model...")

    cnn.evaluate(
        [(X_test, y_test, "Test Dataset")],
        model_path=MODEL_PATH
    )

    # =====================================================
    # 🔹 VISUAL EVALUATION (20 GAMBAR)
    # =====================================================
    print("\nShowing predictions...")

    cnn.evaluate_visual(
        X_test,
        y_test,
        label_map=loader.label_map,
        model_path=MODEL_PATH,
        max_images=20
    )

    print("\nDone ✅")