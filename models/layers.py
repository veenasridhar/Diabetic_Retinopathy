import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, filters, kernel_size):

    out = tf.keras.layers.Conv2D(filters, kernel_size)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Dropout(0.3)(out)
    out = tf.keras.layers.Conv2D(filters*2, kernel_size)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Dropout(0.3)(out)

    return out
