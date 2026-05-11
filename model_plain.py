import tensorflow as tf
from tensorflow.keras import layers, Model

def build_plain(depth=10, input_shape=(112, 112, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial Conv & MaxPooling agar resolusi seimbang dengan ResNet_Skip
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x) # 112 -> 56
    
    filters = 32
    for i in range(depth // 2):
        stride = 2 if i > 0 and i % 2 == 0 else 1
        if stride == 2:
            filters *= 2
            
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='PlainNet_NoSkip')
