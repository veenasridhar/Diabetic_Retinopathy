import tensorflow as tf
import numpy as np
import logging

#Accomplished by  Sefali Pradhan
class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.int32
        )

    def update_state(self, labels,y_pred, *args, **kwargs):
        # Compute batch confusion matrix
        new_conf_matrix = tf.math.confusion_matrix(labels,y_pred,num_classes= self.num_classes,dtype= tf.int32)
        # Update the state variable
        self.confusion_matrix.assign_add(new_conf_matrix)

    def result(self):
      # Return the confusion matrix
      return self.confusion_matrix

    def reset_states(self):
        # Reset the confusion matrix to zeros
        self.confusion_matrix.assign(tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32))

#Accomplished by  Sefali Pradhan
class BinaryAccuracy(tf.keras.metrics.Metric):
    
    def __init__(self,name="accuracy",**kwargs):
        super(BinaryAccuracy,self).__init__(name=name)
        self.correct_predictions = self.add_weight(name="correct", initializer="zeros", dtype=tf.float32)
        self.total_predictions = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)

    def update_state(self,labels,y_pred,sample_weight=None):
        labels = tf.cast(labels,tf.int32)
        y_pred = tf.cast(y_pred,tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.math.equal(labels, y_pred), dtype=tf.float32))
        total = tf.cast(tf.size(labels), dtype=tf.float32)

        self.correct_predictions.assign_add(correct)
        self.total_predictions.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct_predictions, self.total_predictions)*100
    
    def reset_states(self):
        self.correct_predictions.assign(0)
        self.total_predictions.assign(0)
    

#Accomplished by  Veena Sridhar
class Sensitivity(tf.keras.metrics.Metric):

    def __init__(self,name='sensitivity',**kwargs):
        super(Sensitivity,self).__init__(name=name)
        self.true_positive = tf.Variable(0.0,trainable=False)
        self.true_negative = tf.Variable(0.0,trainable=False)
        self.false_positive = tf.Variable(0.0,trainable=False)
        self.false_negative = tf.Variable(0.0, trainable=False)

    def update_state(self,labels,y_pred):
        labels = tf.cast(labels,tf.int32)
        y_pred = tf.cast(y_pred,tf.int32)
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.equal(y_pred,tf.constant(0, dtype= tf.int32)), (tf.math.equal(y_pred,labels))),tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.equal(y_pred,tf.constant(1, dtype= tf.int32)), (tf.math.equal(y_pred,labels))),tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.equal(y_pred,tf.constant(0, dtype= tf.int32)), (tf.math.not_equal(y_pred,labels))),tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.math.equal(y_pred,tf.constant(1, dtype= tf.int32)), (tf.math.not_equal(y_pred,labels))),tf.float32))
        self.true_positive.assign_add(tp)
        self.true_negative.assign_add(tn)
        self.false_positive.assign_add(fp)
        self.false_negative.assign_add(fn)

    def result(self):
        sensitivity = tf.math.divide(self.true_positive,tf.math.add(self.true_positive,self.false_negative))
        specificity = tf.math.divide(self.true_negative,tf.math.add(self.true_negative,self.false_positive))
        return sensitivity.numpy(),specificity.numpy()
    
