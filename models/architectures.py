import gin
import tensorflow as tf
from models.layers import *

#Accomplished by Veena Sridhar
@gin.configurable
def simple_convnet(input_shape,n_classes,kernel_size):
   inputs = tf.keras.Input(input_shape)
   out = tf.keras.layers.Conv2D(32, kernel_size,kernel_regularizer=tf.keras.regularizers.L2(1e-2),activation='silu')(inputs)
   out = tf.keras.layers.MaxPool2D((2, 2))(out)
   out = tf.keras.layers.Dropout(0.2)(out)
   out = tf.keras.layers.Conv2D(64, kernel_size,kernel_regularizer=tf.keras.regularizers.L2(1e-2),activation='silu')(out)
   out = tf.keras.layers.MaxPool2D((2, 2))(out)
   out = tf.keras.layers.Dropout(0.2)(out)
   out = tf.keras.layers.Conv2D(128, kernel_size,kernel_regularizer=tf.keras.regularizers.L2(1e-2),activation='silu')(out)
   out = tf.keras.layers.MaxPool2D((2, 2))(out)
   out = tf.keras.layers.Dropout(0.2)(out)
   out = tf.keras.layers.Conv2D(256, kernel_size,kernel_regularizer=tf.keras.regularizers.L2(1e-2),name = 'visualisation',activation='silu')(out)
   out = tf.keras.layers.MaxPool2D((2, 2))(out)
   out = tf.keras.layers.Dropout(0.2)(out)
   out = tf.keras.layers.GlobalAveragePooling2D()(out)
   out = tf.keras.layers.Dropout(0.2)(out)
   out = tf.keras.layers.Dense(512,kernel_regularizer=tf.keras.regularizers.L2(1e-2),activation=tf.nn.leaky_relu)(out)
   out = tf.keras.layers.Dropout(0.5)(out)
   outputs = tf.keras.layers.Dense(n_classes,activation='softmax')(out)
   return tf.keras.Model(inputs=inputs, outputs=outputs, name='simple_conv_net')

#Accomplished by Sefali Pradhan
@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    inputs = tf.keras.Input(input_shape)
    # establish feature extraction blocks
    out = inputs
    for i in range(1,  n_blocks[1]):
        out = vgg_block(out, base_filters * 2 ** (i))
    for i in range(2, n_blocks[1]):
        out = vgg_block(out, base_filters * 2 ** (i+1))
    out = tf.keras.layers.Activation(activation = 'linear',name = 'visualisation')(out)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes,activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')
#Accomplished by Veena Sridhar
def efficientnet_transferlearning(input_shape,n_classes):
  preprocess_input = tf.keras.applications.efficientnet.preprocess_input
  effcientnetB2 = tf.keras.applications.EfficientNetB2(input_shape = input_shape, include_top = False, weights='imagenet')
  effcientnetB2.trainable = False
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(2, activation='softmax')
  inputs = tf.keras.Input(shape=(256,256,3))
  x = preprocess_input(inputs)
  x = effcientnetB2(x)
  x = tf.keras.layers.Activation(activation = 'linear',name = 'visualisation')(x)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs, name='efficientnet_transferlearning')






