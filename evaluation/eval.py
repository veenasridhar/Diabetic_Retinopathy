import tensorflow as tf
from tensorflow._api.v2.math import confusion_matrix
from evaluation.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from visualization import *
from input_pipeline.preprocessing import preprocess

#Accomplished by Veena Sridhar, Sefali Pradhan
def evaluate(model, checkpoint, ds_test, ds_info, run_paths,model_name):

    manager = tf.train.CheckpointManager(checkpoint,run_paths['path_ckpts_train'],max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))   #manager.checkpoints list all the checkpoints saved.
    else:
      print("Initializing from scratch.")

    confusion_mat = ConfusionMatrix(ds_info['num_classes'])
    accuracy = BinaryAccuracy()
    sensitivity_specificity = Sensitivity()

    for idx,(images,labels) in enumerate(ds_test):
       predictions = model.predict(images)
       preds = tf.math.argmax(predictions,axis=1)

       confusion_mat.update_state(labels,preds)
       accuracy.update_state(labels,preds)
       sensitivity_specificity.update_state(labels,preds)
      
       if model_name != 'efficientnet_transferlearning':
        visual_image = images[1]
        print('True Label of Visualised image',labels[1])
        visual_image = np.expand_dims(visual_image,axis=0)
        grad_cam_model(visual_image,model,'visualisation',idx)
        baseline_image = np.zeros_like(visual_image)
        integrated_gradients(model, visual_image, baseline_image,idx)
  

    print('------------Results-------------------------')
       
    print('Confusion Matrix: \n',confusion_mat.result())
    cm = confusion_mat.result()
    sns.heatmap(cm,annot=True)
    plt.show()
    print('Test Traditional Accuracy: ',accuracy.result().numpy())
    sensitivity,specificity = sensitivity_specificity.result()
    print('Sensitivity and Specificity: ',sensitivity,specificity)
    balanced_accuracy = (sensitivity + specificity)/2.0
    print("Balnced Accuracy ",balanced_accuracy)
    return accuracy.result().numpy()