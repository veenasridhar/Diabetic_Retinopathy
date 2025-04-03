import gin
import tensorflow as tf
import random
import numpy as np
import random
import cv2
import tensorflow_addons as tfa

#Accomplished by Sefali Pradhan
norm_layer = tf.keras.layers.Rescaling(1. / 255.)
@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32)
    label = tf.cast(label,tf.int32)
    # Resize image
    image = tf.image.resize(image, (img_height, img_width))
    image = norm_layer(image)
    label = tf.cast(label >= 2, dtype=tf.int32)
    return image, label


def shear_image(image, shear_x=0.2, shear_y=0.2):
    """
    Apply shear transformation to an image using an affine transformation matrix.
    
    :param image: Input image (Tensor)
    :param shear_x: Shear factor for x-axis
    :param shear_y: Shear factor for y-axis
    :return: Sheared image
    """
    # Define shear matrix
    shear_matrix = [1, shear_x, 0,  # x-axis shear
                    shear_y, 1, 0,  # y-axis shear
                    0, 0]           # No translation

    # Apply shear using the transform method
    sheared_image = tfa.image.transform(image, shear_matrix, interpolation='BILINEAR')
    return sheared_image

rotation_layer = tf.keras.layers.RandomRotation(factor=0.0250)  # Rotate up to ~45 degrees (0.125 * 2 * π) ---0.0250 was good 
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.05, -0.05))  # Slight zoom-in
flip_layer = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")
crop_layer = tf.keras.layers.RandomCrop(0.2,0.2)

#Accomplished by Veena Sridhar
def augment(image, label):
    """Data augmentation"""
    #image = shear_image(image, shear_x=0.05, shear_y=0.05)
    #image = rotation_layer(image,training = True)
    random_rotation = random.randint(1,3)
    image = tf.image.rot90(image,k=random_rotation)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_crop(image,size= (256,256,3),seed = 42)
    return image, label

@gin.configurable
def transfer_learning_preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32)
    label = tf.cast(label,tf.int32)
    # Resize image
    image = tf.image.resize(image, (img_height, img_width))
    label = tf.cast(label >= 2, dtype=tf.int32)
    return image, label


#Balancing classes in each batch --- an alternative approach
def interleave_classes(class_0, class_1):
    """Interleave samples from class_0 and class_1."""
    features = tf.concat([class_0[0], class_1[0]], axis=0)
    labels = tf.concat([class_0[1], class_1[1]], axis=0)
    # Shuffle within the batch to mix class_0 and class_1
    indices = tf.range(tf.shape(features)[0])
    shuffled_indices = tf.random.shuffle(indices)
    return tf.gather(features, shuffled_indices), tf.gather(labels, shuffled_indices)
