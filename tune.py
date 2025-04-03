import logging
import gin
import os
import ray
from ray import tune
import tensorflow as tf
from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc
from tensorboard.plugins.hparams import api as hp
from evaluation.metrics import BinaryAccuracy
from evaluation.eval import evaluate

#Accomplished by Sefali Pradhan
#@gin.configurable
def train_func(config):
    # Hyperparameters
    bindings = [f"{key}={value}" for key, value in config.items()]

    # Generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # Set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    #gin-config
    gin.parse_config_files_and_bindings(
        ['/Users/sefalipradhan/PycharmProjects/Diabetic_Retinopathy_latest/configs/config.gin'],
        bindings) # change path to absolute path of config file
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # Create TensorBoard log directory
    tensorboard_log_dir = os.path.join("C:/ray_results", "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(tensorboard_log_dir)
    print(f"TensorBoard logs are being saved to: {tensorboard_log_dir}")

    # Define hyperparameters and metrics for TensorBoard
    HP_TOTAL_STEPS = hp.HParam('Trainer.total_steps', hp.Discrete([1e4]))
    HP_BASE_FILTERS = hp.HParam('vgg_like.base_filters', hp.Discrete([8, 16]))
    HP_N_BLOCKS = hp.HParam('vgg_like.n_blocks', hp.Discrete([2, 3, 4]))
    HP_DENSE_UNITS = hp.HParam('vgg_like.dense_units', hp.Discrete([32, 64]))
    HP_DROPOUT_RATE = hp.HParam('vgg_like.dropout_rate', hp.RealInterval(0.0, 0.9))
    METRIC_ACCURACY = hp.Metric('val_accuracy', display_name='Validation Accuracy')

    # Log hyperparameter and metric configuration to TensorBoard
    with writer.as_default():
        hp.hparams_config(
            hparams=[HP_TOTAL_STEPS, HP_BASE_FILTERS, HP_N_BLOCKS, HP_DENSE_UNITS, HP_DROPOUT_RATE],
            metrics=[METRIC_ACCURACY],
        )
    # Log hyperparameters for this trial
    with writer.as_default():
        hp.hparams(config)

    # Load dataset
    ds_train, ds_val, ds_test, ds_info = load()

    # Model
    model = vgg_like(input_shape=ds_info["image_shape"], n_classes=ds_info['num_classes'])

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, total_steps=config["Trainer.total_steps"],
                      log_interval=10, ckpt_interval=500, learning_rate=0.001)

    # Training loop
    for val_accuracy in trainer.train():
        with writer.as_default():
            tf.summary.scalar("val_accuracy", val_accuracy, step=int(trainer.checkpoint.step.numpy()))
        tune.report({"val_accuracy": val_accuracy})

    writer.close()

ray.init(ignore_reinit_error=True)
analysis = tune.run(
    train_func,
    config={
        "Trainer.total_steps": tune.grid_search([1e4]),
        "vgg_like.base_filters": tune.choice([8, 16]),
        "vgg_like.n_blocks": tune.choice([2, 3, 4]),
        "vgg_like.dense_units": tune.choice([32, 64]),
        "vgg_like.dropout_rate": tune.uniform(0, 0.9),
    },
    resources_per_trial={"cpu": 6, "gpu": 0},
    num_samples=30,
    max_concurrent_trials=1,
    stop={"training_iteration": 20}
)

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
