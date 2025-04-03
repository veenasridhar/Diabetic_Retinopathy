import logging
import os 
import gin
import pandas as pd 
import tensorflow as tf
#import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from input_pipeline.preprocessing import *
import cv2
from sklearn.utils import resample

#Accomplished by Veena Sridhar
def build_tfrecord(data_dir,images,labels,filename):
    def crop_black_border(image):
        #Black Border Cropping
        img = tf.expand_dims(image,axis=0)
        img = tf.image.crop_and_resize(img,[[0.0,0.09,1.0,0.85]],[0],crop_size=[256,256])
        image = tf.squeeze(img)
        return image

    def image_example(image_string, label):
        #change the image into an example"""
        image_shape = tf.image.decode_jpeg(image_string).shape
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
            'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
            'img_depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def read(image_dir):
        """convert the image into a string file after processing"""

        image = cv2.imread(image_dir)
        encoded_image = crop_black_border(image)
        _,encoded_image = cv2.imencode(".jpg",encoded_image.numpy())
        # change the image to a string
        image_string = encoded_image.tostring()

        return image_string

    with tf.io.TFRecordWriter(filename) as writer:
    # read in the original unmixed image
        for i in range(len(images)):
            img_name = images[i]
            img_label = labels[i]
            img_string = read(img_name)
            tf_example = image_example(img_string, img_label)
            writer.write(tf_example.SerializeToString())

def load_idrid(data_dir, csv_path, split):
    #Creating list from the images and respective labels  
    logging.info(f"Loading IDRID {split} dataset...")
    df_label = pd.read_csv(csv_path)
    image_path = [os.path.join(data_dir, f"{fname}.jpg") for fname in df_label['Image name']]
    
    #Dropping unwanted Columns and relabeling target labels
    df_label = df_label.drop(df_label.columns[2:],axis=1)
    labels = df_label["Retinopathy grade"].values
    print(np.unique(labels,return_counts=True))
    
    return image_path,labels


@gin.configurable
def load(model_name,name, data_dir):

    #Accomplished by Sefali Pradhan
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        train_images,train_labels = load_idrid(os.path.join(data_dir, 'images', 'train'), csv_path=os.path.join(data_dir, 'labels', 'train.csv'), split='train')
        test_images,test_labels = load_idrid(os.path.join(data_dir, 'images', 'test'), csv_path=os.path.join(data_dir, 'labels', 'test.csv'), split='test')
        train_num = len(train_labels)
        test_num = len(test_labels)

        if not os.path.exists('./TFRecord/'):
            os.makedirs('./TFRecord/')
            build_tfrecord(data_dir,train_images,train_labels,'./TFRecord/IDRID_train.tfrecord')
            build_tfrecord(data_dir,test_images,test_labels,'./TFRecord/IDRID_test.tfrecord')

        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'img_height': tf.io.FixedLenFeature([], tf.int64),
            'img_width': tf.io.FixedLenFeature([], tf.int64),
            'img_depth': tf.io.FixedLenFeature([], tf.int64),
        }

        # parse data from bytes format into original image format
        def load_and_preprocess(img_label_dict):
            """parse data"""
            temp = tf.io.parse_single_example(img_label_dict, features)
            img_raw = tf.io.decode_jpeg(temp['image'], channels=3)
            image = tf.reshape(img_raw, [temp['img_height'], temp['img_width'], temp['img_depth']])
            label = temp['label']
            return image,label

        # import the original training set from TFRecord files
        raw_ds_train = tf.data.TFRecordDataset('./TFRecord/IDRID_train.tfrecord')
        ds_before_split = raw_ds_train.map(load_and_preprocess)

        # shuffle the original training set and divide it into training set and validation set
        num_before_split = sum(1 for _ in ds_before_split)
        num_train = int(num_before_split * (1 - 0.2))
        num_val = int(num_before_split * 0.2)

        # split dataset into train and validation
        ds_train = ds_before_split.take(num_train)
        ds_val = ds_before_split.skip(num_train).take(num_val)

        #Accomplished by Veena Sridhar
        class0 = []
        class1 = []
        class2 = []
        class3 = []
        class4 = []
        for idx,(images,labels) in enumerate(ds_train):
            if labels == 1:
                class1.append((images,labels))
            if labels == 0:
                class0.append((images,labels))
            if labels == 2:
                class2.append((images,labels))
            if labels == 3:
                class3.append((images,labels))
            if labels == 4:
                class4.append((images,labels))
                
        #Resampling classes with less samples
        class1_oversampled = resample(class1,replace=True,n_samples=135,random_state=42)
        class4_oversampled = resample(class4,replace=True,n_samples=50,random_state=10)

        oversampled_images = []
        oversampled_labels = []

        for i in range(len(class1_oversampled)):
            oversampled_images.append(class1_oversampled[i][0])
            oversampled_labels.append(class1_oversampled[i][1])
        for i in range(len(class4_oversampled)):
            oversampled_images.append(class4_oversampled[i][0])
            oversampled_labels.append(class4_oversampled[i][1])

        ds_new = tf.data.Dataset.from_tensor_slices((oversampled_images,oversampled_labels))
        ds_train = ds_train.concatenate(ds_new)

        ds_train = ds_train.flat_map(
            lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(2))

        ds_test = tf.data.TFRecordDataset('./TFRecord/IDRID_test.tfrecord')
        ds_test = ds_test.map(load_and_preprocess)
        
        ds_info = {
                "num_classes": 2,  # number of classes in IDRID, for instance
                "class_names": ["No_DR", "DR"],
                "num_train": tf.data.experimental.cardinality(ds_train).numpy(),  # number of training samples
                "num_val": tf.data.experimental.cardinality(ds_val).numpy(),  # number of validation samples
                "num_test": tf.data.experimental.cardinality(ds_test).numpy(),  # number of test samples
                "image_shape": (256, 256, 3),  # example image shape
                "description": "IDRID Diabetic Retinopathy dataset for image classification"
        }
        return prepare(ds_train, ds_val, ds_test, ds_info, model_name)

#Accomplished by sefali Pradhan
@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, model_name, batch_size, caching):
    # Prepare training dataset
    if model_name != 'efficientnet_transferlearning':
        ds_train = ds_train.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds_train = ds_train.map(
            transfer_learning_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()  
    ds_train = ds_train.shuffle(700,seed=42) 
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    if model_name != 'efficientnet_transferlearning':
        ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds_val = ds_val.map(
            transfer_learning_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Prepare test dataset
    if model_name != 'efficientnet_transferlearning':
        ds_test = ds_test.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds_test = ds_test.map(
           transfer_learning_preprocess , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_val, ds_test, ds_info
    
