import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datasetloader import DatasetLoader

def build_resnet50(input_shape, num_classes):
    # Menggunakan ResNet50 (sebagai representasi "VGG 50+" / arsitektur 50 layer) 
    # tanpa weights pre-trained (training dari awal) atau bisa juga menggunakan weights='imagenet'
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    print("=== Loading Dataset ===")
    loader = DatasetLoader("DATASET_FACES", augment=True, augment_times=3)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    num_classes = len(loader.label_map)
    print(f"Jumlah kelas: {num_classes}")
    
    print("=== Building ResNet50 Model ===")
    model = build_resnet50(input_shape=(112, 112, 3), num_classes=num_classes)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("=== Training ResNet50 Model ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=[early_stop]
    )
    
    print("=== Evaluating ResNet50 Model ===")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Plotting Training Loss and Testing Accuracy
    plt.figure(figsize=(10, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.title('ResNet50 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
    plt.title('ResNet50 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet50_loss_acc.png')
    plt.show()

if __name__ == "__main__":
    main()
