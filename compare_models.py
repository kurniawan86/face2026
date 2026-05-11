import tensorflow as tf
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from model_skip import build_resnet
from model_plain import build_plain

def main():
    print("=== Loading CIFAR-10 Dataset ===")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalisasi Data
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    
    # One-hot encoding label
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test, 10)
    
    # Menggunakan subset data agar pelatihan lebih cepat dan sekadar untuk demonstrasi
    subset_size = 5000
    x_train_sub, y_train_sub = x_train[:subset_size], y_train[:subset_size]
    
    print(f"Menggunakan {subset_size} sampel untuk training demi kecepatan...")
    
    models_to_test = {
        'PlainNet (Tanpa Skip)': build_plain(depth=10),
        'ResNet (Dengan Skip)': build_resnet(depth=10)
    }
    
    results = {}
    histories = {}
    
    for name, model in models_to_test.items():
        print(f"\n=== Training Model: {name} ===")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Latih model
        h = model.fit(
            x_train_sub, y_train_sub,
            batch_size=64, 
            epochs=15,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        histories[name] = h
        results[name] = {
            'train_acc': max(h.history['accuracy']),
            'val_acc':   max(h.history['val_accuracy']),
            'params':    model.count_params()
        }
    
    # === Menampilkan Perbandingan di Terminal ===
    print("\n" + "="*65)
    print("=== HASIL PERBANDINGAN PENGARUH SKIP CONNECTION ===")
    print("="*65)
    print(f"{'Model':<25} | {'Train Acc':>10} | {'Val Acc':>10} | {'Total Params':>12}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<25} | {r['train_acc']:>10.4f} | {r['val_acc']:>10.4f} | {r['params']:>12,}")
    print("="*65)
    
    # === Visualisasi Grafik Perbandingan ===
    plt.figure(figsize=(14, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for name, h in histories.items():
        plt.plot(h.history['val_accuracy'], label=f'{name} (Val Acc)')
        plt.plot(h.history['accuracy'], label=f'{name} (Train Acc)', linestyle='--')
    plt.title('Perbandingan Akurasi')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    for name, h in histories.items():
        plt.plot(h.history['val_loss'], label=f'{name} (Val Loss)')
        plt.plot(h.history['loss'], label=f'{name} (Train Loss)', linestyle='--')
    plt.title('Perbandingan Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('skip_connection_comparison.png')
    print("\nGrafik perbandingan berhasil disimpan sebagai 'skip_connection_comparison.png'.")
    plt.show()

if __name__ == "__main__":
    main()
