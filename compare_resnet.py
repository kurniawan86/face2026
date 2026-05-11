import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================
# 1. BUAT DATA SINTETIS
# ============================================
np.random.seed(42)
tf.random.set_seed(42)

# 1000 titik antara -π dan π
x_data = np.linspace(-np.pi, np.pi, 1000).astype('float32')

# Fungsi target: kombinasi sin dan cos
y_data = (np.sin(x_data) + 0.5 * np.cos(2 * x_data)).astype('float32')

# Tambah noise kecil supaya lebih realistis
y_data += np.random.normal(0, 0.05, size=y_data.shape).astype('float32')

# Reshape untuk Keras: (1000,) → (1000, 1)
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

# Split train / test
split = 800
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

print(f"Training samples : {len(x_train)}")
print(f"Test samples     : {len(x_test)}")
print(f"Input shape      : {x_train.shape}")
print(f"Output shape     : {y_train.shape}")


# ============================================
# 2. MODEL A — PLAIN (Tanpa Skip Connection)
# ============================================
# 10 layer Dense yang dalam — tanpa shortcut apapun

def build_plain_network():
    inputs = tf.keras.Input(shape=(1,), name='input')
    
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)  # layer ke-10
    
    outputs = layers.Dense(1, name='output')(x)  # output regresi
    
    return Model(inputs, outputs, name='PlainNet_10layer')


# ============================================
# 3. MODEL B — RESNET (Dengan Skip Connection)
# ============================================
# Arsitektur IDENTIK — 10 layer Dense — tapi dengan skip connection
# setiap 2 layer sekali

def residual_block_dense(x, units):
    """
    Skip connection untuk Dense layer.
    Sama persis dengan residual block CNN,
    tapi menggunakan Dense bukan Conv2D.
    """
    shortcut = x           # simpan input asli

    # Jalur utama: 2 layer Dense
    out = layers.Dense(units, activation='relu')(x)
    out = layers.Dense(units)(out)   # belum aktivasi

    # Penjumlahan: F(x) + x
    out = layers.Add()([out, shortcut])
    out = layers.Activation('relu')(out)  # aktivasi setelah +

    return out


def build_resnet_network():
    inputs = tf.keras.Input(shape=(1,), name='input')

    # Layer pertama: naikkan dimensi dari 1 → 64
    # (supaya bisa dijumlahkan di residual block)
    x = layers.Dense(64, activation='relu')(inputs)

    # 5 residual block × 2 layer = 10 layer total
    # sama persis dengan PlainNet
    x = residual_block_dense(x, 64)   # layer 2–3
    x = residual_block_dense(x, 64)   # layer 4–5
    x = residual_block_dense(x, 64)   # layer 6–7
    x = residual_block_dense(x, 64)   # layer 8–9
    x = residual_block_dense(x, 64)   # layer 10–11

    outputs = layers.Dense(1, name='output')(x)

    return Model(inputs, outputs, name='ResNet_10layer')


# ============================================
# 4. TRAINING DAN PERBANDINGAN
# ============================================

plain_model  = build_plain_network()
resnet_model = build_resnet_network()

print("\n=== Jumlah Parameter ===")
print(f"Plain Net : {plain_model.count_params():,}")
print(f"ResNet    : {resnet_model.count_params():,}")
# Hampir sama — perbedaan hanya dari layer output tambahan

histories = {}

for name, model in [('Plain (tanpa skip)', plain_model),
                    ('ResNet (dengan skip)', resnet_model)]:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',           # Mean Squared Error untuk regresi
        metrics=['mae']       # Mean Absolute Error
    )
    print(f"\n--- Training: {name} ---")
    h = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=0    # silent training
    )
    histories[name] = h

    # Evaluasi akhir
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test MSE : {loss:.6f}")
    print(f"  Test MAE : {mae:.6f}")


# ============================================
# 5. LIHAT PERBEDAAN LOSS AWAL VS AKHIR
# ============================================
print("\n=== Perbandingan Loss per 10 Epoch ===")
print(f"{'Epoch':<8} {'Plain val_loss':>16} {'ResNet val_loss':>16} {'Selisih':>10}")
print("-" * 55)

plain_loss  = histories['Plain (tanpa skip)'].history['val_loss']
resnet_loss = histories['ResNet (dengan skip)'].history['val_loss']

for epoch in [0, 9, 19, 29, 49, 74, 99]:
    p = plain_loss[epoch]
    r = resnet_loss[epoch]
    diff = p - r
    marker = ' ← ResNet menang' if diff > 0.01 else ''
    print(f"Epoch {epoch+1:<3} {p:>16.6f} {r:>16.6f} {diff:>10.6f}{marker}")


# ============================================
# 6. PREDIKSI VISUAL (teks)
# ============================================
print("\n=== Sample Prediksi (x → y_asli vs y_prediksi) ===")
print(f"{'x':>8} {'y asli':>10} {'Plain':>10} {'ResNet':>10}")
print("-" * 42)

sample_x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                     dtype='float32').reshape(-1, 1)
sample_y = np.sin(sample_x) + 0.5 * np.cos(2 * sample_x)

pred_plain  = plain_model.predict(sample_x, verbose=0)
pred_resnet = resnet_model.predict(sample_x, verbose=0)

for i in range(len(sample_x)):
    print(f"{sample_x[i,0]:>8.1f} "
          f"{sample_y[i,0]:>10.4f} "
          f"{pred_plain[i,0]:>10.4f} "
          f"{pred_resnet[i,0]:>10.4f}")