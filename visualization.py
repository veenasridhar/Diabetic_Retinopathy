import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import logging


#Accomplished by Veena Sridhar
def grad_cam_model(image,model,last_conv_layer_name,index):
  #Creating a model which can access trained model's input aand the last convolutional layer
  grad_model = tf.keras.models.Model([model.inputs],[model.get_layer(last_conv_layer_name).output,model.output])
  folder_name = os.path.join(os.getcwd() , 'Visualisation')
  with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(image,training = False)
    tape.watch(last_conv_layer_output)

    #Getting the probability of the positive class (Class 1)
    idx = np.argmax(preds[0]) 
    pred_class = preds[:,idx]
    print('Predicted Class: ',idx)

  #Calculate the gradient of the predicted class with repsect to the last convoolution layer
  gradient = tape.gradient(pred_class,last_conv_layer_output)
  #Averaging the gradients over the batch, width and height dimensions gives us the avg_gradients for each channel
  #This is like the importance score for each channel
  pooled_gradients = tf.reduce_mean(gradient,axis=(0,1,2)).numpy()
  

  #Computing the heatmap

  output = last_conv_layer_output.numpy()[0]
  #Multiply these featuremap with the appropriate importance_scores of the channel
  for i,weight in enumerate(pooled_gradients):
    output[:,:,i] *= weight                             #The respective channel's feature map are multiplied with the respective channel importance
  
  cam = np.mean(output,axis=-1)
  cam =cv2.resize(cam,(image.shape[1],image.shape[2]))
  
  #normalize the cam
  cam = np.maximum(cam,0) /(cam.max() +1e-16)
  cam_image = cam *255
  cam_image = np.clip(cam_image, 0, 255).astype('uint8')
  cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)  #Applying a colormap to the heatmap JET
  
  original_image = tf.cast(image[0]*255, tf.uint8)
  superimposed_img =cv2.addWeighted(original_image.numpy(), 1, cam_image, 0.3, 0)
  
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  plt.imshow(original_image.numpy())
  plt.title('Original Image')

  plt.subplot(1, 2, 2)
  plt.imshow(superimposed_img)
  plt.title('Grad CAM Visualisation')

  plt.show()

  if not os.path.exists(folder_name):
    os.mkdir(folder_name)
  image_path = 'Grad_Cam'+str(index)+'.jpg'
  cv2.imwrite(os.path.join(folder_name,image_path),cv2.cvtColor(superimposed_img,cv2.COLOR_RGB2BGR))
  image_path = 'Cam_heatmap'+str(index)+'.jpg'
  cv2.imwrite(os.path.join(folder_name,image_path),cam_image)
  image_path = 'Original_Image'+str(index)+'.jpg'
  cv2.imwrite(os.path.join(folder_name,image_path),original_image.numpy())


def integrated_gradients(model, image, baseline, index, steps=50):
    # Generate 50 steps from the baseline to the image
    scaled_images = baseline + (image - baseline) * np.linspace(0, 1, steps)[:, None, None, None]
    scaled_images = tf.convert_to_tensor(scaled_images, dtype=tf.float32)
    folder_name = os.path.join(os.getcwd() , 'Visualisation')
    with tf.GradientTape() as tape:
        tape.watch(scaled_images)
        predictions = model(scaled_images)
    grads = tape.gradient(predictions, scaled_images)

    # Average gradients over all steps
    avg_grads = np.mean(grads.numpy(), axis=0)

    # Calculate integrated gradients
    integrated_gradients = (image - baseline) * avg_grads
    attributions =  np.abs(integrated_gradients)
    attributions = np.sum(np.abs(attributions), axis=-1)
    attributions = attributions / np.max(attributions)  # Normalize to [0, 1]

     # Convert to 8-bit heatmap
    heatmap = np.uint8(255 * attributions)
    
    heatmap = np.uint8(255 * attributions)[0]
    heatmap = cv2.resize(heatmap, (256,256))
    

    if heatmap.shape[-1] == 3:
      heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed

    attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)  # Avoid division by zero

    original_image = tf.image.resize(image[0], (256, 256)).numpy()
    original_image = np.uint8(255 * (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image) + 1e-8))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blended = cv2.addWeighted(original_image, 0.7, heatmap, 0.7, 0)

    # Plot the original image and attributions
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image,cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(blended,cmap='gray')
    plt.title('Integrated Gradients')

    plt.subplot(1, 3, 3)
    plt.imshow(heatmap,cmap='hot')
    plt.title('Integrated Gradients - heatmap')

    plt.show()
    if not os.path.exists(folder_name):
      os.mkdir(folder_name)
    image_path = 'Intergrated_Gradients'+str(index)+'.jpg'
    cv2.imwrite(os.path.join(folder_name,image_path),cv2.cvtColor(blended,cv2.COLOR_RGB2BGR))
    image_path = 'Intergrated_Gradients_heatmap'+str(index)+'.jpg'
    cv2.imwrite(os.path.join(folder_name,image_path),heatmap)



