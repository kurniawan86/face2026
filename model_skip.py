import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(x, filters, stride=1):
    shortcut = x
    
    # === Jalur Utama ===
    out = layers.Conv2D(filters, 3, strides=stride,
                        padding='same', use_bias=False)(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    
    out = layers.Conv2D(filters, 3, padding='same', use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    
    # === Cek apakah dimensi berubah ===
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # === Penjumlahan (Skip Connection) ===
    out = layers.Add()([out, shortcut])
    out = layers.ReLU()(out)
    
    return out

def build_resnet(depth=10, input_shape=(112, 112, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial Conv & MaxPooling agar resolusi gambar (112x112) tidak terlalu berat
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x) # 112 -> 56
    
    filters = 32
    for i in range(depth // 2):
        # Setiap 2 block kita tingkatkan filter dan turunkan resolusi (stride=2) untuk efisiensi
        stride = 2 if i > 0 and i % 2 == 0 else 1
        if stride == 2:
            filters *= 2
        x = residual_block(x, filters, stride=stride)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='ResNet_Skip')
