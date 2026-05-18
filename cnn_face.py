import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

class CNNface:
    input_shape = None
    num_classes = 0

    model = None
    history = None

    optimizer = None
    loss = None
    metrics = None

    batch_size = None
    epochs = None
    training_time = None

    early_stopping = None
    eval_results = {}
    architecture_name = None

    def __init__(self, input_shape=(112,112,3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_vgg_like(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        self.model = model
        return model
    
    def build_lenet5(self):
        model = models.Sequential()

        model.add(layers.Conv2D(6, (5,5), activation='relu', input_shape=self.input_shape))
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(16, (5,5), activation='relu'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        self.model = model
        return model
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def get_early_stopping(self, monitor='val_loss', patience=5):
        return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
        batch_size=16, epochs=10, callbacks=None):

        start = time.time()

        if X_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )

        duration = time.time() - start
        print(f"\n⏱️ Training time: {duration:.2f} seconds")

        return self.history, duration
    
    def evaluate(self, datasets, model_path=None):

        # CEK MODEL
        if model_path is not None:
            try:
                self.load(model_path)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Gagal load model: {e}")
                return

        elif self.model is None:
            print("Model belum dibuat / di-load!")
            print("Gunakan build_model() atau load('model.keras')")
            return

        # 🔹 evaluasi
        self.eval_results = {}

        for X, y, name in datasets:
            y_pred = self.model.predict(X)
            y_pred = np.argmax(y_pred, axis=1)

            acc = accuracy_score(y, y_pred)
            report = classification_report(
                y, y_pred,
                output_dict=True,
                zero_division=0
            )

            self.eval_results[name] = {
                "accuracy": acc,
                "f1_score": report["weighted avg"]["f1-score"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"]
            }

            print(f"\n=== {name} ===")
            print(f"Accuracy : {acc:.4f}")
            print(classification_report(y, y_pred, zero_division=0))

        return self.eval_results
    
    def summary(self):
        return self.model.summary()
    
    def save(self, path="model.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path="model.keras"):
        self.model = tf.keras.models.load_model(path)

    def plot_model(self, filename="model.png"):
        plot_model(self.model, to_file=filename, show_shapes=True)
        print(f"Model plot saved: {filename}")
    
    def plot_training(self):
        if self.history is None:
            print("Belum ada history training!")
            return

        plt.figure(figsize=(10,4))

        # Loss
        plt.subplot(1,2,1)
        plt.plot(self.history.history['loss'], label='train_loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.title("Loss")
        plt.legend()

        # Accuracy
        plt.subplot(1,2,2)
        plt.plot(self.history.history['accuracy'], label='train_acc')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='val_acc')
        plt.title("Accuracy")
        plt.legend()

        plt.show()
    
    def evaluate_visual(self, X, y, label_map, model_path=None, max_images=20):

        # 🔹 load model jika ada path
        if model_path is not None:
            try:
                self.load(model_path)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Gagal load model: {e}")
                return

        elif self.model is None:
            print("⚠️ Model belum dibuat / di-load!")
            return

        # 🔹 inverse label map
        inv_map = {v: k for k, v in label_map.items()}

        # 🔹 prediksi
        y_pred = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)

        # 🔹 pilih max_images
        total = min(max_images, len(X))

        plt.figure(figsize=(12, 10))

        for i in range(total):
            plt.subplot(4, 5, i+1)

            img = X[i]

            # jika sudah normalisasi → kembalikan
            if img.max() <= 1.0:
                img = (img * 255).astype("uint8")

            actual = inv_map[y[i]]
            pred = inv_map[y_pred[i]]

            # warna
            color = "green" if actual == pred else "red"

            plt.imshow(img)
            plt.title(f"A: {actual}\nP: {pred}", color=color, fontsize=8)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from datasetloader import DatasetLoader

    # 🔹 load data
    loader = DatasetLoader(
    "DATASET_FACES",
    augment=True,
    augment_times=3)
    X_train, X_test, y_train, y_test = loader.load_data()

    # 🔹 init model
    cnn = CNNface(input_shape=(112,112,3), num_classes=len(loader.label_map))

    cnn.build_vgg_like()
    cnn.summary()

    cnn.compile()

    early_stop = cnn.get_early_stopping(patience=3)

    cnn.train(
        X_train, y_train,
        X_test, y_test,
        epochs=30,
        batch_size=10,
        callbacks=[early_stop]
    )
    
    # 🔹 save model
    cnn.save("face_model.keras")

    # 🔹 evaluate
    cnn.evaluate(
        [(X_test, y_test, "Test Dataset")],
        model_path="face_model.keras",
    )

    # 🔹 plot training
    cnn.plot_training()

    # 🔹 save model
    cnn.save("face_model.keras")

    # # 🔹 plot arsitektur
    # cnn.plot_model("model_arch.png")

    cnn.evaluate_visual(
    X_test,
    y_test,
    label_map=loader.label_map,
    model_path="face_model.keras",
    max_images=20)