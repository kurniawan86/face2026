import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================
# 1. BUAT DATA: DUA LINGKARAN KONSENTRIS
# ============================================
# Kelas 0: titik-titik di lingkaran kecil (dalam)
# Kelas 1: titik-titik di lingkaran besar (luar)
# Ini masalah non-linear — butuh jaringan yang benar-benar belajar

np.random.seed(42)
tf.random.set_seed(42)

def make_circles(n=500, noise=0.1):
    """Buat dua kelas: lingkaran dalam dan luar."""
    # Kelas 0: lingkaran kecil, radius ~0.5
    angles_0 = np.random.uniform(0, 2 * np.pi, n // 2)
    r_0 = np.random.normal(0.5, noise, n // 2)
    x_0 = np.column_stack([r_0 * np.cos(angles_0),
                            r_0 * np.sin(angles_0)])
    y_0 = np.zeros(n // 2)

    # Kelas 1: lingkaran besar, radius ~1.0
    angles_1 = np.random.uniform(0, 2 * np.pi, n // 2)
    r_1 = np.random.normal(1.0, noise, n // 2)
    x_1 = np.column_stack([r_1 * np.cos(angles_1),
                            r_1 * np.sin(angles_1)])
    y_1 = np.ones(n // 2)

    x = np.vstack([x_0, x_1]).astype('float32')
    y = np.concatenate([y_0, y_1]).astype('float32')

    # Acak urutannya
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


x_data, y_data = make_circles(n=1000, noise=0.08)

# Split
split = 800
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

print(f"Data: {len(x_train)} train, {len(x_test)} test")
print(f"Input: koordinat (x1, x2)")
print(f"Output: 0 = lingkaran dalam, 1 = lingkaran luar")

# Cek distribusi kelas
print(f"Kelas 0: {int((y_train==0).sum())} samples")
print(f"Kelas 1: {int((y_train==1).sum())} samples")


# ============================================
# 2. MODEL A — PLAIN NETWORK (dalam, tanpa skip)
# ============================================

def build_plain_classifier():
    inputs = tf.keras.Input(shape=(2,))   # 2 fitur: koordinat x1, x2

    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)  # 8 layer dalam

    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='PlainClassifier')


# ============================================
# 3. MODEL B — RESNET CLASSIFIER (dengan skip)
# ============================================

def res_block(x, units):
    shortcut = x

    out = layers.Dense(units)(x)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    out = layers.Dense(units)(out)
    out = layers.BatchNormalization()(out)

    # F(x) + x
    out = layers.Add()([out, shortcut])
    out = layers.Activation('relu')(out)

    return out


def build_resnet_classifier():
    inputs = tf.keras.Input(shape=(2,))

    # Naikkan dimensi ke 128 dulu
    x = layers.Dense(128, activation='relu')(inputs)

    # 4 residual block = 8 layer — sama dengan PlainNet
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='ResNetClassifier')


# ============================================
# 4. TRAINING
# ============================================

plain_clf  = build_plain_classifier()
resnet_clf = build_resnet_classifier()

print("\n=== Parameter Count ===")
print(f"Plain : {plain_clf.count_params():,}")
print(f"ResNet: {resnet_clf.count_params():,}")

histories = {}

for name, model in [('Plain', plain_clf),
                    ('ResNet', resnet_clf)]:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    h = model.fit(
        x_train, y_train,
        epochs=80,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=0
    )
    histories[name] = h
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{name} — Test Accuracy: {acc:.4f} | Loss: {loss:.4f}")


# ============================================
# 5. PERBANDINGAN DETAIL
# ============================================

print("\n=== Akurasi per Epoch ===")
print(f"{'Epoch':<8} {'Plain acc':>12} {'ResNet acc':>12} {'Unggul':>10}")
print("-" * 46)

p_acc = histories['Plain'].history['val_accuracy']
r_acc = histories['ResNet'].history['val_accuracy']

for ep in [0, 4, 9, 19, 39, 59, 79]:
    p = p_acc[ep]
    r = r_acc[ep]
    winner = 'ResNet ✓' if r > p else ('Plain ✓' if p > r else 'Seri')
    print(f"Epoch {ep+1:<3} {p:>12.4f} {r:>12.4f} {winner:>10}")


# ============================================
# 6. UJI PREDIKSI MANUAL
# ============================================
print("\n=== Prediksi Manual ===")
print("Titik (0,0) = pusat → harusnya kelas 0 (lingkaran DALAM)")
print("Titik (0,1) = tepi  → harusnya kelas 1 (lingkaran LUAR)")
print()

test_points = np.array([
    [0.0,  0.0],   # tengah → kelas 0
    [0.5,  0.0],   # radius 0.5 → kelas 0
    [0.0,  1.0],   # radius 1.0 → kelas 1
    [0.7,  0.7],   # radius ~1.0 → kelas 1
    [0.3,  0.3],   # radius ~0.4 → kelas 0
], dtype='float32')

true_classes = [0, 0, 1, 1, 0]

pred_plain  = plain_clf.predict(test_points, verbose=0)
pred_resnet = resnet_clf.predict(test_points, verbose=0)

print(f"{'Titik':<14} {'Asli':>6} {'Plain':>10} {'ResNet':>10}")
print("-" * 44)
for i, (pt, true) in enumerate(zip(test_points, true_classes)):
    p_pred = int(pred_plain[i,0]  > 0.5)
    r_pred = int(pred_resnet[i,0] > 0.5)
    p_mark = '✓' if p_pred  == true else '✗'
    r_mark = '✓' if r_pred == true else '✗'
    print(f"({pt[0]:.1f}, {pt[1]:.1f}){'':<5}"
          f"{true:>6} "
          f"{p_pred:>5} {p_mark:>3}  "
          f"{r_pred:>5} {r_mark:>3}")