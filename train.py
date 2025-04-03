import gin
import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrix
import numpy as np
import os
import matplotlib.pyplot as plt
from visualization import *

#Accomplished by Sefali Pradhan, Veena Sridhar
@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval,learning_rate):
       
        # Summary Writer
        train_log_dir = os.path.join(run_paths['path_model_id'],'logs','train')
        val_log_dir = os.path.join(run_paths['path_model_id'],'logs','val')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=3000,decay_rate=0.9)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
        self.train_recall = tf.keras.metrics.Recall(name='train_recall')
        self.train_precision = tf.keras.metrics.Precision(name='train_precsion')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Accuracy(name='val_accuracy')
        self.val_recall = tf.keras.metrics.Recall(name='val_recall')
        self.val_precision = tf.keras.metrics.Precision(name='val_precsion')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, run_paths['path_ckpts_train'], max_to_keep=3)
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch...")
        
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        predictions = tf.cast(tf.reshape(tf.math.argmax(predictions,axis=1),(-1,1)),tf.int32)
        self.train_accuracy(labels, predictions)
        self.train_recall(labels, predictions)
        self.train_precision(labels, predictions)

    #@tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.val_loss(t_loss)
        predictions = tf.cast(tf.reshape(tf.math.argmax(predictions,axis=1),(-1,1)),tf.int32)
        self.val_accuracy(labels, predictions)
        self.val_recall(labels, predictions)
        self.val_precision(labels, predictions)
    
    def train(self):
        max_accuracy = 0.0
        min_loss = float("inf")
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history =[] 
        self.val_acc_history =[]
        for idx, (images, labels) in enumerate(self.ds_train): 

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_state()
                self.val_accuracy.reset_state()
                self.val_recall.reset_state()
                self.val_precision.reset_state()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Training Loss: {}, Training Accuracy: {}, Training Precision: {}, Training Recall: {}, Validation Loss: {}, Validation Accuracy: {}, Val Precision: {}, Val Recall: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.train_precision.result(),self.train_recall.result(),
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100,
                                             self.val_precision.result(),self.val_recall.result()))
                
                
                # Write summary to tensorboard
                with self.train_summary_writer.as_default():
                   tf.summary.scalar('Loss', self.train_loss.result(), step=step)
                   tf.summary.scalar('Accuracy',self.train_accuracy.result()*100, step=step)
                   tf.summary.scalar('Recall', self.train_recall.result(), step=step)
                   tf.summary.scalar('Precision', self.train_precision.result(), step=step)

                with self.val_summary_writer.as_default():
                   tf.summary.scalar('Loss', self.val_loss.result(), step=step)
                   tf.summary.scalar('Accuracy',self.val_accuracy.result()*100, step=step)
                   tf.summary.scalar('Recall', self.val_recall.result(), step=step)
                   tf.summary.scalar('Precision', self.val_precision.result(), step=step)
                
                self.train_loss_history.append(self.train_loss.result())
                self.train_acc_history.append(self.train_accuracy.result())
                self.val_loss_history.append(self.val_loss.result())
                self.val_acc_history.append(self.val_accuracy.result())
                

                # Reset train metrics
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()
                self.train_recall.reset_state()
                self.train_precision.reset_state()

                yield self.val_accuracy.result().numpy()


            if step % self.ckpt_interval == 0:
              if self.val_accuracy.result() >= max_accuracy and self.val_loss.result() <= min_loss:
                min_loss = self.val_loss.result().numpy()
                max_accuracy = self.val_accuracy.result().numpy()
                # Save checkpoint
                self.checkpoint.step.assign_add(1)
                save_path = self.manager.save()
                logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path ))
              
              elif self.val_accuracy.result() >= max_accuracy:
                min_loss = self.val_loss.result().numpy()
                max_accuracy = self.val_accuracy.result().numpy()
                self.checkpoint.step.assign_add(1)
                save_path = self.manager.save()
                logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path ))
            
            if step % self.total_steps == 0:
               logging.info(f'Finished training after {step} steps.')
               # Save final checkpoint
               if self.val_accuracy.result() >= max_accuracy and self.val_loss.result() <= min_loss:
                    self.checkpoint.step.assign_add(1)
                    save_path = self.manager.save()
               logging.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))

               epochs = range(1, len(self.train_loss_history) + 1)
               plt.figure(figsize=(12, 5))

               plt.subplot(1, 2, 1)
               plt.plot(epochs, self.train_loss_history, label="Train Loss", color='blue', linestyle='-')
               plt.plot(epochs, self.val_loss_history, label="Validation Loss", color='red', linestyle='--')
               plt.xlabel("Epochs")
               plt.ylabel("Loss")
               plt.title("Training vs Validation Loss")
               plt.legend()
               plt.grid(True)

               # 🔹 Plot Accuracy
               plt.subplot(1, 2, 2)
               plt.plot(epochs, self.train_acc_history, label="Train Accuracy", color='blue', linestyle='-')
               plt.plot(epochs, self.val_acc_history, label="Validation Accuracy", color='red', linestyle='--')
               plt.xlabel("Epochs")
               plt.ylabel("Accuracy")
               plt.title("Training vs Validation Accuracy")
               plt.legend()
               plt.grid(True)

               plt.tight_layout()
               plt.show()
            
               return self.val_accuracy.result().numpy()
            
    def model_output(self):
        return self.model
