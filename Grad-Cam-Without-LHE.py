# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:13:50 2021

@author: MSE
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
tf.executing_eagerly()
import keras as k
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



img_path = "Left-Right_Images/Covid19_L_R/Covid-32-R.jpeg"


img = cv2.imread(img_path)
img = cv2.resize(img,(224,224))
plt.imshow(img)


def prepare(img_path):
    IMG_SIZE=224
    img_array=cv2.imread(img_path)
    img_array = img_array/255.0
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)


###########################################################################################

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


#######################################################################################

model = tf.keras.models.load_model("AlexNet(Right-WLHE).h5")

last_conv_layer_name = "conv2d_4"
classifier_layer_names = ['activation_4', 'max_pooling2d_2', 'batch_normalization_4', 'flatten', 'dense', 'activation_5', 'dropout', 'batch_normalization_5', 'dense_1', 'activation_6',
                          'dropout_1', 'batch_normalization_6', 'dense_2', 'activation_7']
# activation_4   max_pooling2d_2   batch_normalization_4  flatten    dense  activation_5   dropout  batch_normalization_5  dense_1  activation_6
#  dropout_1  batch_normalization_6  dense_2  activation_7


model.summary()

layer_names=[layer.name for layer in model.layers]
print(layer_names)


#img_array = get_img_array(img_path, size=img_size)
img_array = prepare(img_path)
  

# Print what the top predicted class is
preds = model.predict_proba(img_array)

#preds = ([0],[1],[2])
#print("Predicted:", decode_predictions(preds, top=1)[2])
print("Predicted:",preds)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

# Display heatmap
plt.matshow(heatmap)
plt.show()



# We load the original image
img_array = cv2.imread(img_path)
img=cv2.resize(img_array,(224,224))
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
display(Image(save_path))

