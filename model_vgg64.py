import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datasetloader import DatasetLoader

def build_vgg64_custom(input_shape, num_classes):
    """
    Membangun arsitektur mirip VGG (tanpa residual connection/ResNet) dengan total 64 layer.
    VGG murni menggunakan tumpukan layer Conv2D (3x3) diikuti MaxPooling2D.
    """
    model = models.Sequential()
    
    # Konfigurasi arsitektur untuk mencapai total 62 layer konvolusi + 2 layer dense = 64 layer (berbobot)
    # List berisi: (jumlah_filter, jumlah_conv_dalam_blok)
    # Total conv: 10 + 12 + 14 + 14 + 12 = 62 layer konvolusi
    vgg_config = [
        (64, 10),
        (128, 12),
        (256, 14),
        (512, 14),
        (512, 12)
    ]
    
    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for filters, num_convs in vgg_config:
        for _ in range(num_convs):
            # Menggunakan BatchNormalization untuk sedikit membantu gradien yang menghilang (Vanishing Gradient),
            # walaupun pada arsitektur VGG asli tidak ada.
            model.add(layers.Conv2D(filters, (3, 3), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
    model.add(layers.Flatten())
    
    # Layer 63
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Layer 64 (Output)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
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
    
    print("=== Building Custom VGG-64 Model ===")
    model = build_vgg64_custom(input_shape=(112, 112, 3), num_classes=num_classes)
    
    # Tampilkan summary untuk membuktikan kedalaman layer
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("=== Training VGG-64 Model ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=[early_stop]
    )
    
    print("=== Evaluating VGG-64 Model ===")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Plotting Training Loss and Testing Accuracy
    plt.figure(figsize=(10, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.title('VGG-64 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
    plt.title('VGG-64 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vgg64_loss_acc.png')
    plt.show()

if __name__ == "__main__":
    main()
