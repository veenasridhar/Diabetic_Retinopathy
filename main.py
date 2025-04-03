import gin
import logging
from absl import app, flags
import tensorflow as tf
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
from input_pipeline.preprocessing import preprocess
import os


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False , 'Specify whether to train or evaluate a model.')

def main(argv):

    model_name = 'simple_convnet'
    folder_path = ''
    if len(argv) >1:
        model_name = argv[1]            #Model Name
        if len(argv) == 3:
            folder_path = argv[2]           #CheckPoint Path
        else:
            folder_path = ''

    # generate folder structures
    run_paths = utils_params.gen_run_folder('')

    if folder_path != '':
        run_paths = utils_params.gen_run_folder(folder_path)
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(model_name)
    # model
    if model_name ==  'simple_convnet':
        model = simple_convnet(input_shape=ds_info["image_shape"], n_classes=ds_info['num_classes'])
    elif model_name == 'modified_vgg':
        model = vgg_like(input_shape=ds_info['image_shape'], n_classes=ds_info['num_classes'])
    elif model_name == 'efficientnet_transferlearning':
        model = efficientnet_transferlearning(ds_info["image_shape"], n_classes=ds_info['num_classes'])                                  
    model.summary()
        
    #Training
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
        evaluate(model,checkpoint,ds_test,ds_info,run_paths,model_name)
        
if __name__ == "__main__":
    app.run(main)
